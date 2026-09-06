# -*- coding: utf-8 -*-
"""神经重排 v3 探针：批间一致性 + bz 三场景 + 主人新用例 nwgngdxcsyx。"""
import os
import sys
import time
import threading

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from dict_engine import DictEngine
from rerank import StatReranker
from neural_rerank import NeuralReranker
from xiaohe import decode_syllable
import caret_ctx

LAMBDA = 1.0

t0 = time.perf_counter()
de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))
rr = StatReranker(de)
rr.load(os.path.join(BASE, "dicts"))
print("[load] %.1fs" % (time.perf_counter() - t0))

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

print("== 一致性：同词不同批 ==")
c6 = ['包子', '杯子', '不足', '不再', '班子', '婊子']
s5 = nr.score('我吃了一个', c6[:5])
s6 = nr.score('我吃了一个', c6)
for w in c6[:5]:
    print("   %-4s b5=%7.2f b6=%7.2f drift=%+.3f" % (w, s5[w], s6[w], s6[w] - s5[w]))


def stat_pool(code, prevs):
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
    if n >= 2:
        for w in de.lookup_initial(" ".join(code), 90 if n == 2 else 45):
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


def run(ctx, code, want, window):
    prevs = caret_ctx.tail_words(ctx, de.word_py, 3) if ctx else []
    ranked = stat_pool(code, prevs)
    top = [w for w, _ in ranked[:window]]
    ns = nr.score(ctx or "。", top)
    fused = sorted(((s - LAMBDA * ns.get(w, -12.0), w) for w, s in ranked[:window]),
                   key=lambda t: t[0])
    print("-" * 62)
    print("上文=%r 码=%s(%d键) 期望=%s" % (ctx, code, len(code), want))
    print("  统计top5:", ["%s(%.2f)" % (w, s) for w, s in ranked[:5]])
    print("  融合top8:", ["%s(%.2f)" % (w, v) for v, w in fused[:8]])
    fr = [w for _, w in fused]
    print("  期望排名:", fr.index(want) + 1 if want in fr else "池外")


print("== bz 三场景 ==")
for ctx, want in (("我吃了一个", "包子"), ("我打碎了一个", "杯子"), ("我追上了一个", "豹子")):
    run(ctx, "bz", want, 90)

print("== 主人新用例 ==")
# 试=sh，小鹤简拼 sh->u；主人串里的 s 无法对应"试"，两种都测
run("", "nwgngdxcuyx", "那我给你个东西测试一下", 48)
run("", "nwgngdxcsyx", "那我给你个东西测试一下", 48)
