# -*- coding: utf-8 -*-
"""bz 上下文场景探针 v2 —— 多上文衰减打分路径验证。
    我吃了一个   + bz -> 期望 包子
    我打碎了一个 + bz -> 期望 杯子
    我追上了一个 + bz -> 期望 豹子
"""
import os
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from dict_engine import DictEngine
from rerank import StatReranker
from xiaohe import decode_syllable
import caret_ctx
import rerank

t0 = time.perf_counter()
de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))
rr = StatReranker(de)
rr.load(os.path.join(BASE, "dicts"))
print("[load] %.1fs  %d 词  DECAY=%.2f B_PRE=%.2f B_PRE_CHAR=%.2f P=%.1f L=%.1f"
      % (time.perf_counter() - t0, de.size, rerank.DECAY, rerank.B_PRE,
         rerank.B_PRE_CHAR, rerank.P, rerank.L))

SCENARIOS = [
    ("我吃了一个", "bz", "包子"),
    ("我打碎了一个", "bz", "杯子"),
    ("我追上了一个", "bz", "豹子"),
]


def build_pool(code, prevs):
    """镜像 linghe.Engine.compute 的同池装配（2键简拼召回 90）。"""
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


for ctx, code, want in SCENARIOS:
    prevs = caret_ctx.tail_words(ctx, de.word_py, 3)
    ranked = build_pool(code, prevs)
    print("=" * 64)
    print("上文=%r  tail=%r  码=%s  期望=%s" % (ctx, prevs, code, want))
    for i, (w, s) in enumerate(ranked[:10]):
        mark = " <== 期望" if w == want else ""
        print("  %2d. %-6s %.2f%s" % (i + 1, w, s, mark))
    if want not in [w for w, _ in ranked[:10]]:
        for i, (w, s) in enumerate(ranked):
            if w == want:
                print("  ... 期望词 %s 排第 %d（%.2f）" % (want, i + 1, s))
                break
        else:
            print("  ... 期望词 %s 不在池内" % want)
    if want in de.word_py:
        c = rr._cost(want)
        b = rr._bi_bonus_multi(prevs, want)
        print("  [分项] %s: cost=%.2f multi_bonus=%.2f（%s）"
              % (want, c, b, " + ".join(
                  "%s*%.2f^%d=%.2f" % (p, rerank.DECAY, i, rr._bi_bonus(p, want) * rerank.DECAY ** i)
                  for i, p in enumerate(prevs) if rr._bi_bonus(p, want))))
