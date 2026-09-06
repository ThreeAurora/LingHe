# -*- coding: utf-8 -*-
"""融合策略诊断：bz 场景 90 池，对比 原始λ / z-score / PMI 三种神经分融合。"""
import os
import sys
import threading

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from dict_engine import DictEngine
from rerank import StatReranker
from neural_rerank import NeuralReranker
from xiaohe import decode_syllable

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


def stat_pool(code, prevs, pool_n=90):
    pool = {}

    def add(w):
        if w in pool:
            return
        pym = de.word_py.get(w) or ""
        m = max(1, len(pym.split()))
        pool[w] = rr.score_word(w, prevs) / m

    n = len(code)
    if n >= 2 and n % 2 == 0:
        py = " ".join(decode_syllable(code[i:i + 2]) for i in range(0, n, 2))
        for w in de.lookup_pinyin(py, 45):
            add(w)
    for w in de.lookup_initial(" ".join(code), pool_n):
        add(w)
    vpool = []
    if n % 2 == 0 and n >= 4:
        keys = [decode_syllable(code[i:i + 2]) for i in range(0, n, 2)]
        vpool += rr.viterbi(keys, "py", 5, ret_cost=True, prev=prevs[0] if prevs else "")
    vpool += rr.viterbi(list(code), "ini", 5, ret_cost=True, prev=prevs[0] if prevs else "")
    vpool.sort(key=lambda t: t[1] / max(1, t[2]))
    for s, c, m in vpool:
        cps = c / max(1, m)
        if s not in pool or cps < pool[s]:
            pool[s] = cps
    return sorted(pool.items(), key=lambda kv: kv[1])


def rank_of(seq, want):
    return seq.index(want) + 1 if want in seq else 99


for ctx, want in (("我吃了一个", "包子"), ("我打碎了一个", "杯子"), ("我追上了一个", "豹子")):
    prevs = ["一个", "了"]  # 近似 tail_words；诊断用途足够
    ranked = stat_pool("bz", prevs)
    top = [w for w, _ in ranked[:90]]
    ns = nr.score(ctx, top)

    # --- 策略A: 原始 logP 减法 λ=1 ---
    a = sorted(top, key=lambda w: dict(ranked)[w] - ns.get(w, -12.0))
    # --- 策略B: 批内 z-score ---
    vals = [ns[w] for w in top if w in ns]
    mu = sum(vals) / len(vals)
    sd = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5 or 1.0
    # z 越大(越接近0)越好；融合: stat - λ*z*scale, scale 取 1.0
    b = sorted(top, key=lambda w: dict(ranked)[w] - 1.0 * ((ns[w] - mu) / sd) if w in ns else dict(ranked)[w] + 12.0)
    # --- 策略C: PMI 增益 = logP(w|ctx) - logP(w|"。") ---
    ns0 = nr.score("。", top)
    gain = {w: ns.get(w, -12.0) - ns0.get(w, -12.0) for w in top}
    c = sorted(top, key=lambda w: dict(ranked)[w] - 1.0 * gain[w])

    print("=" * 64)
    print("%s  期望=%s  (池%d词, μ=%.2f σ=%.2f)" % (ctx, want, len(top), mu, sd))
    print("  神经分样本:", {w: round(ns[w], 2) for w in ("包子", "杯子", "豹子", "不在", "班子") if w in ns})
    print("  增益样本  :", {w: round(gain[w], 2) for w in ("包子", "杯子", "豹子", "不在", "班子") if w in gain})
    for name, seq in (("A原始", a), ("B zscore", b), ("C PMI ", c)):
        print("  %s: %s" % (name, seq[:8]))
    print("  期望词: A=%s B=%s C=%s  (stat名次=%s)" % (
        rank_of(a, want), rank_of(b, want), rank_of(c, want),
        rank_of([w for w, _ in ranked], want)))
