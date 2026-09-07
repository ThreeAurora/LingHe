# -*- coding: utf-8 -*-
"""探针：RBT3 粗筛（prune_keep=20）里「包子」能否进送裁名单。"""
import os, sys, time, threading
sys.path.insert(0, r'e:\CCSpace\projects\2026\09\linghe')
from neural_rerank import NeuralReranker
from mabiao import MaBiao

BASE = r'e:\CCSpace\projects\2026\09\linghe'
mb = MaBiao()
mb.load_dir(BASE + r'\mabiao')
ctx = "我吃了一个"

# 复刻 compute 的 dynamic pool（bz 简拼 90 词 + viterbi ini）
import itertools
from dict_engine import DictEngine
de = DictEngine(log=lambda *a: None)
de.load_dir(BASE + r'\dicts')
words = de.lookup_initial("b z", 90)
words = list(dict.fromkeys(words))

nr = NeuralReranker(os.path.join(BASE, "ai_neural", "rbt3"))
loaded = threading.Event()
orig = nr._load

def _l2():
    orig()
    if nr.ready:
        loaded.set()
nr._load = _l2
t0 = time.perf_counter()
nr.load_async()
loaded.wait(120)
print("RBT3 加载 %.0fs" % (time.perf_counter() - t0))

t0 = time.perf_counter()
s = nr.score(ctx, words, max_cand=16, max_ctx=16)
print("打分 %.0fms" % ((time.perf_counter() - t0) * 1000))
ranked = sorted(((v, w) for w, v in s.items()), reverse=True)
for i, (v, w) in enumerate(ranked[:30]):
    mark = "  <<< 包子" if w == "包子" else ""
    print("%2d  %-6s %.3f%s" % (i + 1, w, v, mark))
print()
print("『包子』RBT3 粗筛排名: %d / %d （prune_keep=20 时 %s）" %
      ([w for _, w in ranked].index("包子") + 1, len(ranked),
       "送裁 ✓" if [w for _, w in ranked].index("包子") + 1 <= 20 else "被挡 ✗"))