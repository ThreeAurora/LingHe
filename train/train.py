# -*- coding: utf-8 -*-
"""FastReranker 训练脚本。

用法:
  python train.py --data train/data/gen.jsonl --epochs 3 --out train/ckpt
样本: {"code","mode","ctx","cands","gold","gold_idx","n_fixed","wlen"}
要点: 候选顺序训练时随机打乱（防位置作弊），原始 rank 以特征形式喂入。
"""
import os, sys, json, math, random, argparse, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import FastReranker

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def enc_word(word, vocab, L):
    ids = [vocab.get(ch, 1) for ch in word[:L]]
    return ids + [0] * (L - len(ids))


class LazyJsonlDS(Dataset):
    """惰性 jsonl 数据集：只存行偏移索引，按需 seek 读取。

    2026-09-08 修 MemoryError：370 万行全量 dict 驻内存 ~10GB，再加
    DataLoader worker 复制直接爆——改为 offset 索引 + 常驻 fd 按行读，
    内存 ~100MB（offset 表），OS page cache 自然热。
    """

    def __init__(self, path, offsets, vocab, max_cand=96, ctx_len=12,
                 cand_len=4, lo=0, hi=None, seed=0):
        self.path = path
        self.offsets = offsets
        self.vocab = vocab
        self.max_cand = max_cand
        self.ctx_len = ctx_len
        self.cand_len = cand_len
        self.lo = lo
        self.hi = hi if hi is not None else len(offsets)
        self.rng = random.Random(seed)
        self._fd = None

    def __len__(self):
        return self.hi - self.lo

    def __getstate__(self):
        # Windows spawn worker 会 pickle dataset——文件句柄不可序列化，
        # 丢弃后 worker 内按需重开（2026-09-08 训练提速：惰性集+多 worker）
        s = self.__dict__.copy()
        s["_fd"] = None
        return s

    def _line(self, i):
        if self._fd is None:
            self._fd = open(self.path, "rb")
        self._fd.seek(self.offsets[self.lo + i])
        return json.loads(self._fd.readline())

    def __getitem__(self, i):
        r = self._line(i)
        cands = list(r["cands"])[: self.max_cand]
        feats0 = [(math.log1p(j) / 6.0, 1.0 if j < r["n_fixed"] else 0.0,
                   len(w) / 4.0) for j, w in enumerate(cands)]
        order = list(range(len(cands)))
        self.rng.shuffle(order)
        cands = [cands[j] for j in order]
        feats = [feats0[j] for j in order]
        if r["gold_idx"] >= len(cands):
            return None
        label = order.index(r["gold_idx"])
        ctx_ids = [self.vocab.get(ch, 1) for ch in r["ctx"][-self.ctx_len:]]
        ctx_ids = [0] * (self.ctx_len - len(ctx_ids)) + ctx_ids
        cand_ids = [enc_word(w, self.vocab, self.cand_len) for w in cands]
        while len(cand_ids) < self.max_cand:
            cand_ids.append([1] * self.cand_len)
            feats.append((0.0, 0.0, 0.0))
        mask = [1.0] * len(cands) + [0.0] * (self.max_cand - len(cands))
        inv = [order[j] for j in range(len(order))] + \
              [-1] * (self.max_cand - len(order))
        return {
            "ctx": torch.tensor(ctx_ids, dtype=torch.long),
            "cand": torch.tensor(cand_ids, dtype=torch.long),
            "feats": torch.tensor(feats, dtype=torch.float),
            "mask": torch.tensor(mask, dtype=torch.float),
            "label": torch.tensor(label, dtype=torch.long),
            "inv": torch.tensor(inv, dtype=torch.long),
        }


def build_offsets_vocab(path, min_freq=2):
    """一次遍历：行偏移索引 + 流式 vocab（370 万行 ~2 分钟，零驻留）。"""
    from collections import Counter
    offs = []
    cnt = Counter()
    pos = 0
    with open(path, "rb") as f:
        for line in f:
            offs.append(pos)
            pos += len(line)
            r = json.loads(line)
            for ch in r["ctx"] + r["gold"]:
                cnt[ch] += 1
            for w in r["cands"]:
                for ch in w:
                    cnt[ch] += 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for ch, n in sorted(cnt.items(), key=lambda x: -x[1]):
        if n >= min_freq and "\u4e00" <= ch <= "\u9fff":
            vocab[ch] = len(vocab)
    return offs, vocab


class MemmapDS(Dataset):
    """memmap 数据集：预编码数组切片，零 json 解析（单进程最优路线）。

    2026-09-08 主人定案：不开子进程（worker 数=0），效率靠预编码——
    步内数据成本从 ~0.3s json 解析降到 ~2ms 数组切片。
    """

    def __init__(self, mm_dir, idx, seed=0):
        self.idx = idx                       # 全局洗牌后的样本索引子列表
        self.ctx = np.load(os.path.join(mm_dir, "ctx.npy"), mmap_mode="r")
        self.cand = np.load(os.path.join(mm_dir, "cand.npy"), mmap_mode="r")
        self.wlen = np.load(os.path.join(mm_dir, "wlen.npy"), mmap_mode="r")
        self.nfixed = np.load(os.path.join(mm_dir, "nfixed.npy"), mmap_mode="r")
        self.gold = np.load(os.path.join(mm_dir, "gold.npy"), mmap_mode="r")
        self.POOL = self.cand.shape[1]
        # 09-08 向量化：numpy 洗牌/特征/定位替代纯 Python 循环（数据准备
        # 曾占 0.3s/step 的九成，GPU 一直在等）。seed 保证 val 可复现。
        self._npr = np.random.default_rng(seed if seed else 20260908)
        self._rank_feat = (np.log1p(np.arange(self.POOL)) / 6.0).astype(np.float32)
        self._pad_row = np.ones((self.POOL, 4), np.int16)
        self._neg1 = np.full(self.POOL, -1, np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        gi = self.idx[i]
        ctx = torch.from_numpy(self.ctx[gi].astype(np.int64))
        cand_np = self.cand[gi]                       # [POOL,4] int16
        wl = self.wlen[gi]                            # [POOL] int8
        nf = int(self.nfixed[gi])
        gold = int(self.gold[gi])
        n = int((wl > 0).sum())                       # 有效候选数（pad 槽 wlen=0）
        if gold >= n:                                 # 模板词条脏行（全库2条）
            return None
        perm = self._npr.permutation(n)               # 向量化洗牌
        pad = self.POOL - n
        cand_full = np.concatenate([cand_np[perm], self._pad_row[:pad]], 0)
        feats = np.zeros((self.POOL, 3), np.float32)
        feats[:n, 0] = self._rank_feat[:n][perm]      # log1p(原rank)/6
        feats[:n, 1] = perm < nf                      # 固频区标志
        feats[:n, 2] = wl[perm] / 4.0                 # 词长
        label = int(np.flatnonzero(perm == gold)[0])  # 金标洗后位置
        mask = np.zeros(self.POOL, np.float32)
        mask[:n] = 1.0
        inv = np.concatenate([perm, self._neg1[:pad]])  # 洗后位置->原rank
        return {
            "ctx": ctx,
            "cand": torch.from_numpy(cand_full.astype(np.int64)),
            "feats": torch.from_numpy(feats),
            "mask": torch.from_numpy(mask),
            "label": torch.tensor(label, dtype=torch.long),
            "inv": torch.from_numpy(inv),
        }


def collate(batch):
    batch = [b for b in batch if b is not None]   # 视野外样本丢弃
    if not batch:
        return None
    out = {}
    for k in ("ctx", "cand", "feats", "mask", "label", "inv"):
        out[k] = torch.stack([b[k] for b in batch])
    return out


def evaluate(model, loader, dev):
    model.eval()
    n, hit = 0, 0
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(dev == "cuda")):
        for b in loader:
            if b is None:
                continue
            s = model(b["ctx"].to(dev), b["cand"].to(dev),
                      b["feats"].to(dev), b["mask"].to(dev))
            pred = s.argmax(-1).cpu()
            hit += (pred == b["label"]).sum().item()
            n += len(pred)
    return hit / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="train/data/gen.jsonl")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--nlayer", type=int, default=2)
    ap.add_argument("--out", default="ckpt")
    ap.add_argument("--val-frac", type=float, default=0.02)
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # 相对路径以 train/ 为基
    os.makedirs(a.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev, flush=True)

    # 数据源：memmap 预编码优先（单进程零解析），否则惰性 jsonl
    mm_dir = os.path.join("data", "mm")
    mm_meta = os.path.join(mm_dir, "meta.json")
    if os.path.exists(mm_meta):
        meta = json.load(open(mm_meta))
        all_idx = list(range(meta["n"]))
        random.Random(7).shuffle(all_idx)
        n_val = max(1, int(len(all_idx) * a.val_frac))
        va_idx, tr_idx = all_idx[:n_val], all_idx[n_val:]
        print("memmap: train=%d val=%d pool=%d"
              % (len(tr_idx), len(va_idx), meta["pool"]), flush=True)
        vocab = json.load(open(os.path.join("ckpt", "vocab.json"),
                               encoding="utf-8"))
        tr_ds = MemmapDS(mm_dir, tr_idx)
        va_ds = MemmapDS(mm_dir, va_idx, seed=12345)
    else:
        print("indexing+vocab (one pass)...", flush=True)
        offsets, vocab = build_offsets_vocab(a.data)
        random.Random(7).shuffle(offsets)          # 行级洗牌=样本洗牌
        n_val = max(1, int(len(offsets) * a.val_frac))
        va_off, tr_off = offsets[:n_val], offsets[n_val:]
        print("train=%d val=%d" % (len(tr_off), len(va_off)), flush=True)
        tr_ds = LazyJsonlDS(a.data, tr_off, vocab)
        va_ds = LazyJsonlDS(a.data, va_off, vocab, seed=12345)
    print("vocab=%d" % len(vocab), flush=True)
    json.dump(vocab, open(os.path.join(a.out, "vocab.json"), "w", encoding="utf-8"),
              ensure_ascii=False)

    model = FastReranker(len(vocab), d=a.d, nlayer=a.nlayer).to(dev)
    print("params=%.2fM" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    tr_dl = DataLoader(tr_ds, batch_size=a.bs, shuffle=True,
                       collate_fn=collate, num_workers=0, pin_memory=True,
                       drop_last=True)
    va_dl = DataLoader(va_ds, batch_size=256, shuffle=False, collate_fn=collate,
                       num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = len(tr_dl) * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=total, pct_start=0.06)
    ce = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda"))  # 09-08 混合精度
    best = 0.0
    t0 = time.time()
    step = 0
    for ep in range(a.epochs):
        model.train()
        for bi, b in enumerate(tr_dl):
            if b is None:
                continue
            with torch.cuda.amp.autocast(enabled=(dev == "cuda")):
                s = model(b["ctx"].to(dev), b["cand"].to(dev),
                          b["feats"].to(dev), b["mask"].to(dev))
                loss = ce(s, b["label"].to(dev))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if bi % 200 == 0:
                acc = (s.argmax(-1).cpu() == b["label"]).float().mean().item()
                print("ep%d step%d/%d loss=%.4f bacc=%.3f %.0fs"
                      % (ep, step, total, loss.item(), acc, time.time() - t0),
                      flush=True)
        va = evaluate(model, va_dl, dev)
        print("== ep%d val top1=%.4f ==" % (ep, va), flush=True)
        if va > best:
            best = va
            torch.save({"model": model.state_dict(), "vocab_size": len(vocab),
                        "d": a.d, "nlayer": a.nlayer, "val": va},
                       os.path.join(a.out, "best.pt"))
    print("BEST val top1=%.4f -> %s/best.pt" % (best, a.out))


if __name__ == "__main__":
    main()
