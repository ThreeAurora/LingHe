# -*- coding: utf-8 -*-
"""端到端终验 v3：主 beam + 锚定 + 最大匹配 + 神经引导 viterbi → 整句组纯神经排序。"""
import os
import sys
import threading
import time

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

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


def e2e(code, want):
    n = len(code)
    pool = {}
    vpool = list(rr.viterbi(list(code), "ini", 24, ret_cost=True))
    vpool += rr.anchor_sentences(list(code), "ini", 12)
    vpool += rr.max_match_sentences(list(code), "ini")
    t0 = time.perf_counter()
    nv = rr.neural_viterbi(list(code), "ini", 3, nr=nr)
    nv_ms = (time.perf_counter() - t0) * 1000
    vpool += nv
    for s, c, m in vpool:
        pool[s] = min(pool.get(s, 9e9), c / max(1, m))
    ranked = sorted(pool.items(), key=lambda kv: kv[1])
    top = [w for w, _ in ranked[:48]]
    ns = nr.score("。", top, max_cand=16)
    sents = sorted([(ns[w], w) for w in top if len(w) >= 5], reverse=True)
    words = sorted([(dict(ranked)[w] - ns.get(w, -12.0), w)
                    for w in top if len(w) < 5])
    merged = [w for _, w in sents] + [w for _, w in words]
    rank = merged.index(want) + 1 if want in merged else None
    print("码=%s(%d键) 期望[%s] 终排名=%s  引导切分%.0fms 产%d条"
          % (code, n, want, rank, nv_ms, len(nv)))
    print("   引导切分产出:", [s for s, _, _ in nv])
    print("   整句组top4:", [w for _, w in sents[:4]])
    return rank


r1 = e2e("nwgngdxcuyx", "那我给你个东西测试一下")
r2 = e2e("wjtxixhcy", "我今天想吃西湖醋鱼")
print()
print("验收: 11键=%s 9键=%s" % ("✓" if r1 == 1 else "✗(%s)" % r1,
                                "✓" if r2 == 1 else "✗(%s)" % r2))
