# -*- coding: utf-8 -*-
"""jsonl -> memmap 预编码（一次性 ~3 分钟，之后训练零 json 解析）。

输出 train/data/mm/:
  ctx.npy   int16 [N,12]    上文字符 id
  cand.npy  int16 [N,48,4]  候选字符 id（池宽 48 = 训练数据金标最大深度）
  wlen.npy  int8  [N,48]    候选词长
  nfixed.npy int16[N]       各样本固频数
  gold.npy  int16 [N]       金标原池 rank
  meta.json                 N / 池宽
编码用 ckpt/vocab.json（全量训练已构建的 12210 词表）。
"""
import os, sys, json
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))   # train/
DATA = os.path.join(BASE, "data", "gen_v4.jsonl")
VOCAB = os.path.join(BASE, "ckpt", "vocab.json")
OUT = os.path.join(BASE, "data", "mm")
CTX_LEN, POOL, CAND_LEN = 12, 48, 4


def enc(word, vocab):
    ids = [vocab.get(ch, 1) for ch in word[:CAND_LEN]]
    return ids + [0] * (CAND_LEN - len(ids))


def main():
    os.makedirs(OUT, exist_ok=True)
    vocab = json.load(open(VOCAB, encoding="utf-8"))
    # 先数行数
    n = sum(1 for _ in open(DATA, "rb"))
    print("rows=%d" % n, flush=True)
    ctx = np.lib.format.open_memmap(os.path.join(OUT, "ctx.npy"), mode="w+",
                                    dtype=np.int16, shape=(n, CTX_LEN))
    cand = np.lib.format.open_memmap(os.path.join(OUT, "cand.npy"), mode="w+",
                                     dtype=np.int16, shape=(n, POOL, CAND_LEN))
    wlen = np.lib.format.open_memmap(os.path.join(OUT, "wlen.npy"), mode="w+",
                                     dtype=np.int8, shape=(n, POOL))
    nfixed = np.lib.format.open_memmap(os.path.join(OUT, "nfixed.npy"), mode="w+",
                                       dtype=np.int16, shape=(n,))
    gold = np.lib.format.open_memmap(os.path.join(OUT, "gold.npy"), mode="w+",
                                     dtype=np.int16, shape=(n,))
    i = 0
    with open(DATA, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            c = [vocab.get(ch, 1) for ch in r["ctx"][-CTX_LEN:]]
            ctx[i] = [0] * (CTX_LEN - len(c)) + c
            cs = r["cands"][:POOL]
            for j, w in enumerate(cs):
                cand[i, j] = enc(w, vocab)
                wlen[i, j] = min(len(w), 20)   # int8 防溢出（模板词条 len 可达148）
            for j in range(len(cs), POOL):     # pad 槽：UNK 防全 pad NaN
                cand[i, j] = 1
                wlen[i, j] = 0
            nfixed[i] = r["n_fixed"]
            gold[i] = r["gold_idx"]
            i += 1
            if i % 500000 == 0:
                print("  %d / %d" % (i, n), flush=True)
    for arr in (ctx, cand, wlen, nfixed, gold):
        arr.flush()
    json.dump({"n": n, "pool": POOL, "ctx_len": CTX_LEN, "cand_len": CAND_LEN},
              open(os.path.join(OUT, "meta.json"), "w"))
    print("DONE %d rows -> %s" % (n, OUT), flush=True)


if __name__ == "__main__":
    main()
