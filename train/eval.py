# -*- coding: utf-8 -*-
"""评估：原始池排序（现状基线） vs FastReranker 重排（memmap 版，单进程）。

指标: P@1 / P@3 / MRR（金标=主人实际会选的词）。
基线=金标在引擎原池中的 rank（什么都不做时的表现）。
模型=重排后金标的 rank。提升幅度=端侧模型的净价值。

2026-09-08 重写：
  - 数据走 data/mm memmap（零 json 解析，370万 也只有数组切片）；
  - val 划分与 train.py 完全一致（Random(7).shuffle 后前 2%）；
  - 两种序：natural（端侧部署真实形态，主指标）/ shuffled seed=12345
    （训练分布对照）；
  - margin 门控扫描与 fast_rerank.py 语义一致：
    pred!=0 且 sc[pred]-sc[0] < tau → 不动原首选（单点置顶）；
  - 深位分桶（基线 rank 0/1-4/5-9/10-19/20+）。
"""
import os, sys, json, math, time, argparse
import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))


def load_model(ckpt, dev):
    from model import FastReranker
    ck = torch.load(ckpt, map_location=dev)
    model = FastReranker(ck["vocab_size"], d=ck["d"], nlayer=ck["nlayer"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck.get("val")


def batches(mm, va_idx, bs, seed=None):
    """yield (ctx[B,12], cand[B,96,4], feats[B,96,3], mask[B,96], gold_orig[B], n[B])

    seed=None → natural 序（部署真实形态）；seed=int → 复现 MemmapDS 的乱序。
    gold_orig = 金标在引擎原池中的 rank（基线）。
    """
    import random as _r
    rng = _r.Random(seed) if seed is not None else None
    ctx_mm = np.load(os.path.join(mm, "ctx.npy"), mmap_mode="r")
    cand_mm = np.load(os.path.join(mm, "cand.npy"), mmap_mode="r")
    wlen_mm = np.load(os.path.join(mm, "wlen.npy"), mmap_mode="r")
    nfixed_mm = np.load(os.path.join(mm, "nfixed.npy"), mmap_mode="r")
    gold_mm = np.load(os.path.join(mm, "gold.npy"), mmap_mode="r")
    POOL = cand_mm.shape[1]
    buf = []
    for gi in va_idx:
        ctx = ctx_mm[gi].astype(np.int64)
        cand_np = cand_mm[gi]                       # [POOL,4] 原池序
        wl = wlen_mm[gi]
        nf = int(nfixed_mm[gi])
        gold = int(gold_mm[gi])
        n = int((wl > 0).sum())
        if gold >= n:                                 # 模板词条脏行，跳过（全库2条）
            continue
        c = cand_np[:n]
        w = wl[:n]
        gold_orig = gold                            # natural 序下=原 rank
        if rng is not None:
            order = list(range(n))
            rng.shuffle(order)
            c = cand_np[order]
            w = wl[order]
            gold_orig = order.index(gold)
        pad = POOL - n
        cand_full = np.concatenate([c, np.ones((pad, 4), np.int16)], 0)
        feats = [(math.log1p(j) / 6.0, 1.0 if j < nf else 0.0,
                  float(w[j]) / 4.0) for j in range(n)]
        feats += [(0.0, 0.0, 0.0)] * pad
        mask = [1.0] * n + [0.0] * pad
        buf.append((ctx, cand_full.astype(np.int64),
                    np.array(feats, np.float32),
                    np.array(mask, np.float32), gold_orig, n))
        if len(buf) == bs:
            yield buf
            buf = []
    if buf:
        yield buf


def forward(model, dev, buf):
    ctx = torch.from_numpy(np.stack([b[0] for b in buf])).to(dev)
    cand = torch.from_numpy(np.stack([b[1] for b in buf])).to(dev)
    feats = torch.from_numpy(np.stack([b[2] for b in buf])).to(dev)
    mask = torch.from_numpy(np.stack([b[3] for b in buf])).to(dev)
    with torch.no_grad():
        sc = model(ctx, cand, feats, mask).cpu().numpy()
    return sc


def metrics(ranks):
    ranks = np.asarray(ranks)
    return (float((ranks == 0).mean()), float((ranks < 3).mean()),
            float((1.0 / (ranks + 1.0)).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mm", default="data/mm")
    ap.add_argument("--ckpt", default="ckpt/best.pt")
    ap.add_argument("--vocab", default="ckpt/vocab.json")  # 校验一致性
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--tau-scan", default="0,0.25,0.5,0.75,1,1.25,1.5,2,2.5,3,4,5")
    ap.add_argument("--out", default="eval_full_report.txt")
    a = ap.parse_args()
    os.chdir(ROOT)
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    meta = json.load(open(os.path.join(a.mm, "meta.json")))
    all_idx = list(range(meta["n"]))
    import random as _r
    _r.Random(7).shuffle(all_idx)                    # 与 train.py 完全一致
    n_val = max(1, int(len(all_idx) * a.val_frac))
    va_idx = all_idx[:n_val]
    vocab = json.load(open(a.vocab, encoding="utf-8"))
    model, ck_val = load_model(a.ckpt, dev)
    print("device=%s val=%d vocab=%d pool=%d ckpt_val=%s"
          % (dev, len(va_idx), len(vocab), meta["pool"], ck_val), flush=True)

    def after_move(gold, pred):
        """单点置顶 pred 后金标的新 rank（fast_rerank 语义：pred==0 → 不动，
        pred!=0 → 置顶，gold==pred 即金标被选中 → 0）。"""
        if pred == 0:
            return float(gold)
        return 0.0 if gold == pred else (gold + 1.0 if gold < pred else gold)

    def run(seed):
        gold_ranks = []
        for buf in batches(a.mm, va_idx, a.bs, seed):
            sc = forward(model, dev, buf)
            for row, (_, _, _, _, gold_orig, n) in enumerate(buf):
                s = sc[row, :n]
                pred = int(s.argmax())
                gold_ranks.append(after_move(gold_orig, pred))
        return np.array(gold_ranks)

    # ---------- natural 序（部署真实形态）：基线 / 纯重排 / 门控扫描 ----------
    base_ranks, nat_ranks = [], []
    gate_rows = []                                   # (gold_orig, sc0, scpred, pred)
    for buf in batches(a.mm, va_idx, a.bs, None):
        sc = forward(model, dev, buf)
        for row, (_, _, _, _, gold_orig, n) in enumerate(buf):
            s = sc[row, :n]
            pred = int(s.argmax())
            nat_ranks.append(after_move(gold_orig, pred))
            base_ranks.append(gold_orig)
            gate_rows.append((gold_orig, float(s[0]), float(s[pred]), pred))
    base_ranks = np.array(base_ranks)
    mnp = np.array(nat_ranks)
    print("== natural 序（部署形态） ==", flush=True)
    print("基线      P@1=%.4f P@3=%.4f MRR=%.4f" % metrics(base_ranks))
    print("纯重排    P@1=%.4f P@3=%.4f MRR=%.4f" % metrics(mnp))

    taus = [float(x) for x in a.tau_scan.split(",")]
    print("tau    P@1     P@3     MRR     rank0误伤", flush=True)
    best_tau, best_p1 = None, -1.0
    for tau in taus:
        rk = base_ranks.copy()
        hurt = 0
        for i, (g, s0, sp, pred) in enumerate(gate_rows):
            if pred != 0 and (sp - s0) >= tau:       # 门控放行 → 置顶
                rk[i] = 0.0 if g == pred else (g + 1.0 if g < pred else g)
                if g == 0:
                    hurt += 1
        p1, p3, mrr = metrics(rk)
        print("%.2f  %.4f  %.4f  %.4f  %d(%.3f%%)"
              % (tau, p1, p3, mrr, hurt, 100.0 * hurt / len(rk)), flush=True)
        if p1 > best_p1:
            best_p1, best_tau = p1, tau

    # ---------- shuffled 对照（训练分布） ----------
    rk_sh = run(12345)
    print("== shuffled(seed=12345) 对照 ==", flush=True)
    print("纯重排    P@1=%.4f P@3=%.4f MRR=%.4f" % metrics(rk_sh), flush=True)

    # ---------- 深位分桶（natural 纯重排 vs 基线） ----------
    print("== 深位分桶（natural） ==", flush=True)
    buckets = [(0, 1), (1, 5), (5, 10), (10, 20), (20, 10 ** 9)]
    print("桶           n      基线P@1  重排P@1  净变", flush=True)
    for lo, hi in buckets:
        sel = (base_ranks >= lo) & (base_ranks < hi)
        if not sel.any():
            continue
        b0 = float((base_ranks[sel] == 0).mean())
        b1 = float((mnp[sel] == 0).mean())
        print("rank[%s,%s)  %6d  %.4f   %.4f   %+.4f"
              % (lo, hi if hi < 10 ** 8 else "+", sel.sum(), b0, b1, b1 - b0),
              flush=True)

    rep = ("eval done val=%d base_P@1=%.4f rerank_P@1=%.4f best_tau=%.2f -> %.4f  %.0fs"
           % (len(va_idx), float((base_ranks == 0).mean()),
              float((mnp == 0).mean()),
              best_tau, best_p1, time.time() - t0))
    print(rep, flush=True)
    open(a.out, "w", encoding="utf-8").write(rep + "\n")


if __name__ == "__main__":
    main()
