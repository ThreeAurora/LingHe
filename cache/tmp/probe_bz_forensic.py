# -*- coding: utf-8 -*-
"""取证：bz 场景三层病灶的根因数据
1) 单字权重 vs 二字词权重（吧/做/子/不 vs 包子/杯子）——Viterbi 字序列为何碾压词
2) cooc_pre 词级共现里有没有动宾对：(吃,包子)(打碎,杯子)(追,豹子)(一个,包子)...
3) 豹子在 by_initial['b z'] 里的排名（召回深度问题）
4) by_initial 重复条目统计
"""
import os
import sys
import math

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from dict_engine import DictEngine
from rerank import StatReranker
import marshal

de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))
rr = StatReranker(de)
rr.load(os.path.join(BASE, "dicts"))

print("== 1) 权重与代价对比 ==")
words = ["吧", "做", "被", "子", "不", "在", "包子", "杯子", "豹子", "鼻子",
         "不足", "不再", "并在", "一个", "吃了"]
for w in words:
    wt = de.weight(w)
    c = rr._cost(w)
    print("  %-4s weight=%-12s cost=%8.2f" % (w, wt, c))

print("== 2) 预训练词级共现（SIGHAN）==")
pairs = [("吃", "包子"), ("吃", "杯子"), ("打碎", "杯子"), ("追", "豹子"),
         ("追上", "豹子"), ("一个", "包子"), ("一个", "杯子"), ("一个", "豹子"),
         ("吃", "了"), ("吃", "东西"), ("吃", "饭"), ("喝", "水"), ("打", "球")]
for a, b in pairs:
    print("  (%s,%s) = %s" % (a, b, rr.pre_word.get((a, b), 0)))

print("== 2b) 用户 bigram 里有没有相关记录 ==")
rel = [(k, v) for k, v in rr.bigram.items()
       if any(t in k for t in ("包子", "杯子", "豹子", "吃", "碎", "追"))]
for k, v in rel[:15]:
    print("  %r = %d" % (k, v))

print("== 3) 豹子 in by_initial['b z'] ==")
lst = de.by_initial.get("b z", [])
print("  池内条目数=%d（含重复）" % len(lst))
seen = {}
for i, (wt, w) in enumerate(lst):
    seen.setdefault(w, []).append((i + 1, wt))
for w in ("包子", "杯子", "豹子", "鼻子", "班子"):
    print("  %-4s -> %s" % (w, seen.get(w, "未收录!")))

print("== 4) by_initial 重复率 ==")
from collections import Counter
cnt = Counter(w for _, w in lst)
dup = sum(1 for w, c in cnt.items() if c > 1)
print("  'b z': 唯一词 %d，其中重复 %d 个" % (len(cnt), dup))

print("== 5) 字级共现（备查）==")
for a, b in [("吧", "做"), ("吃", "包"), ("一", "个"), ("了", "一")]:
    print("  (%s,%s) = %s" % (a, b, rr.pre_char.get((a, b), 0)))
