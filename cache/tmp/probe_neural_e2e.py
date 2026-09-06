# -*- coding: utf-8 -*-
"""神经重排端到端模拟探针：统计池 → RBT3 精排 → 融合排序。
验证主人三场景 + tune 代表用例不被破坏。
"""
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
TOP_K = 12

t0 = time.perf_counter()
de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))
rr = StatReranker(de)
rr.load(os.path.join(BASE, "dicts"))
print("[load] %.1fs  %d 词" % (time.perf_counter() - t0, de.size))

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
    print("神经加载失败")
    sys.exit(1)


def stat_pool(code, prevs):
    """镜像 Engine.compute 同池装配。"""
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


CASES = [
    ("我吃了一个", "bz", "包子"),
    ("我打碎了一个", "bz", "杯子"),
    ("我追上了一个", "bz", "豹子"),
    ("", "nihao", "你好"),
    ("", "jintiantianqibucuo", "今天天气不错"),
]

for ctx, code, want in CASES:
    prevs = caret_ctx.tail_words(ctx, de.word_py, 3) if ctx else []
    ranked = stat_pool(code, prevs)
    dict_side = ranked[:90]
    top = [w for w, _ in dict_side[: (90 if len(code) == 2 else TOP_K)]]
    ns = nr.score(ctx or "。", top)
    fused = []
    for w, s in dict_side:
        v = s - LAMBDA * ns[w] if w in ns else s
        fused.append((v, w))
    fused.sort()
    print("=" * 62)
    print("上文=%r 码=%s 期望=%s" % (ctx, code, want))
    print("  统计池top6:", ["%s(%.2f)" % (w, s) for w, s in dict_side[:6]])
    print("  融合后top8:", ["%s(%.2f)" % (w, v) for v, w in fused[:8]])
    fr = [w for _, w in fused]
    pos = fr.index(want) + 1 if want in fr else -1
    print("  期望词融合后排名: %d" % pos)
