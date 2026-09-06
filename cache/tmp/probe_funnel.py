# -*- coding: utf-8 -*-
"""神经引导漏斗预研：
1) 口语前缀在统计 viterbi 的真实排名（定变体窗口 K）
2) 全喂式 1 行/句打分 vs 伪似然的区分度对比（定漏斗宽度）
"""
import os
import sys
import threading

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

import numpy as np
from dict_engine import DictEngine
from rerank import StatReranker
from neural_rerank import NeuralReranker

de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))
rr = StatReranker(de)
rr.load(os.path.join(BASE, "dicts"))

nr = NeuralReranker(os.path.join(BASE, "ai_neural", "rbt3"))
ev = threading.Event()
_orig = nr._load


def _load2():
    _orig()
    if nr.ready:
        ev.set()


nr._load = _load2
nr.load_async()
if not ev.wait(60):
    sys.exit(1)


def viterbi_rank(code, want, n=96):
    out = rr.viterbi(list(code), "ini", n, ret_cost=True)
    seqs = [s for s, _, _ in out]
    r = seqs.index(want) + 1 if want in seqs else None
    return r, out


print("== 实验1: 口语串在统计 viterbi 的排名 ==")
for code, want in (("wjtxi", "我今天想吃"), ("wjtxixhcy", "我今天想吃西湖醋鱼"),
                   ("nwgngdx", "那我给你个东西"), ("nwgngdxcuyx", "那我给你个东西测试一下"),
                   ("nwgng", "那我给你")):
    r, out = viterbi_rank(code, want)
    print("  %-14s %-14s rank=%s" % (code, want, r))


def whole_logits_score(sents):
    """全喂式 1 行/句：CLS+句+SEP 定长填充，读各字位 MLM 自恢复 logP 均值。"""
    tok = nr.tok
    mask_id = tok.vocab.get("[MASK]", 103)
    SEQ = 28
    rows, meta = [], []
    for s in sents:
        ids = ([tok.cls] + tok.encode(s)[:20] + [tok.sep])
        rows.append(ids)
        meta.append(s)
    B = len(rows)
    input_ids = np.zeros((B, SEQ), dtype=np.int64)
    attn = np.zeros((B, SEQ), dtype=np.int64)
    for i, r in enumerate(rows):
        input_ids[i, :len(r)] = r
        attn[i, :len(r)] = 1
    logits = nr.sess.run(None, {"input_ids": input_ids,
                                "attention_mask": attn,
                                "token_type_ids": np.zeros((B, SEQ), dtype=np.int64)})[0]
    out = {}
    for i, s in enumerate(meta):
        ids = rows[i]
        lps = []
        for pos in range(1, len(ids) - 1):  # 跳过 CLS/SEP
            row = logits[i, pos].astype(np.float64)
            row -= row.max()
            lse = np.log(np.exp(row).sum())
            lps.append(row[ids[pos]] - lse)
        out[s] = float(np.mean(lps)) if lps else -99.0
    return out


print()
print("== 实验2: 全喂式 1 行/句 区分度 ==")
sents = ["我今天想吃西湖醋鱼", "文件体现出西湖醋鱼", "为家庭新车型后槽牙",
         "那我给你个东西测试一下", "那我给你过渡性测试一下",
         "年我国能够的乡村是一些", "我今天新车型会采用"]
t0 = __import__("time").perf_counter()
ws = whole_logits_score(sents)
ms = (__import__("time").perf_counter() - t0) * 1000
ps = nr.score("。", sents, max_cand=16)
print("  全喂式 %d句 %.0fms   伪似然同批" % (len(sents), ms))
for s in sents:
    print("  全喂 %7.2f   伪似然 %7.2f   %s" % (ws[s], ps.get(s, -99), s))
