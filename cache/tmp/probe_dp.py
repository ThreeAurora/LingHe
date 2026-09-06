# -*- coding: utf-8 -*-
"""viterbi 断点终极定位：dp[2]/dp[3]/j4 展开，一轮内全部打印。"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from dict_engine import DictEngine
from rerank import StatReranker, BEAM, MAX_SPAN, SPAN_CANDS

de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))
rr = StatReranker(de)
rr.load(os.path.join(BASE, "dicts"))

keys = ("w", "j", "t", "x", "i")
print("BEAM=%d MAX_SPAN=%d SPAN_CANDS=%d" % (BEAM, MAX_SPAN, SPAN_CANDS))

dp = [None] * 6
dp[0] = [(0.0, (), "")]
for j in range(1, 6):
    cand = []
    lo = max(0, j - MAX_SPAN)
    for i in range(lo, j):
        for c0, seq, prev in (dp[i] or []):
            words_here = rr._span_cands(keys[i:j], "ini", SPAN_CANDS)
            if not words_here:
                continue
            for w in words_here:
                cw = rr._cost(w)
                cand.append((c0 + cw - rr._bi_bonus_span(prev, w), seq + (w,), w))
    cand.sort(key=lambda t: t[0])
    dp[j] = cand[:BEAM]
    if j == 3:
        print("dp3:")
        for c, s, _ in dp[j]:
            print("   %8.2f %s" % (c, "+".join(s)))
    if j == 4:
        mine = [(round(c, 2), "+".join(s)) for c, s, _ in cand if s and s[0] == "我"]
        print("j4: 首词我 条数=%d 最优5=%s" % (len(mine), sorted(mine)[:5]))
        print("j4: cand min=%.2f  top3=%s" % (
            min(t[0] for t in cand),
            [(round(c, 2), "+".join(s)) for c, s, _ in cand[:3]]))
