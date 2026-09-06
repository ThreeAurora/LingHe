# -*- coding: utf-8 -*-
"""11 键全简拼召回诊断：beam 加宽能否召回目标句 + 整句神经裁决能否顶到第一。

主人码串：nwgngdxcsyx（打不出——试=sh，简拼声母是 u 不是 s）
正确码串：nwgngdxcuyx（那n 我w 给g 你n 个g 东d 西x 测c 试u 一y 下x）
"""
import os
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

import rerank as R
from dict_engine import DictEngine
from rerank import StatReranker
from neural_rerank import NeuralReranker
from xiaohe import decode_syllable
import threading

de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))
rr = StatReranker(de)
rr.load(os.path.join(BASE, "dicts"))

WANT = "那我给你个东西测试一下"

print("== 索引抽查 ==")
for key in ("c u", "d x", "n w", "g n", "y x"):
    hits = de.by_initial.get(key, ())
    words = [w for _, w in hits[:6]]
    print("  by_initial[%r] -> %s" % (key, words))

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

for beam in (16, 48, 96):
    R.BEAM = beam
    rr._viterbi_cache.clear()
    for code in ("nwgngdxcuyx", "nwgngdxcsyx"):
        keys = list(code)
        t0 = time.perf_counter()
        out = rr.viterbi(keys, "ini", 32, ret_cost=True)
        ms = (time.perf_counter() - t0) * 1000
        seqs = [s for s, _, _ in out]
        rank = seqs.index(WANT) + 1 if WANT in seqs else None
        print("-" * 64)
        print("beam=%d 码=%s(%d键)  %dms  目标句名次=%s" % (beam, code, len(code), ms, rank))
        for s, c, m in out[:6]:
            print("    %.1f  %s" % (c / max(1, m), s))

# 目标句进池后：整句神经裁决（beam=96 的池 + 池外强插目标句对照）
print("=" * 64)
print("== 整句神经裁决 ==")
R.BEAM = 96
rr._viterbi_cache.clear()
out = rr.viterbi(list("nwgngdxcuyx"), "ini", 24, ret_cost=True)
sents = [s for s, _, _ in out]
if WANT not in sents:
    sents.append(WANT)
scores = nr.score("。", sents, max_cand=14)
stat = {s: c / max(1, m) for s, c, m in out if s in scores}
for s in sorted(sents, key=lambda s: -scores.get(s, -99))[:12]:
    print("  神经 %7.2f  统计 %s  %s" % (scores.get(s, -99.0),
          ("%7.2f" % stat[s]) if s in stat else "  --   ", s))
print("目标句神经名次:", sorted(sents, key=lambda s: -scores.get(s, -99)).index(WANT) + 1)
