# -*- coding: utf-8 -*-
"""解剖 viterbi top1 路径的逐段代价，与目标路径对齐比较。"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from dict_engine import DictEngine
from rerank import StatReranker

de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))
rr = StatReranker(de)
rr.load(os.path.join(BASE, "dicts"))


def decompose(words):
    total = 0.0
    prev = ""
    parts = []
    for w in words:
        c = rr._cost(w)
        b = rr._bi_bonus_span(prev, w)
        parts.append("%s[%.1f%+.1f]" % (w, c, -b))
        total += c - b
        prev = w
    return total, parts


TOP1 = ["年", "我国", "能够", "的", "乡村", "是", "一些"]
t, parts = decompose(TOP1)
print("竞争路径 总%.2f" % t, " ".join(parts))

WANT = ["那", "我", "给你", "个", "东西", "测试", "一下"]
t2, parts2 = decompose(WANT)
print("目标路径 总%.2f" % t2, " ".join(parts2))

print()
print("== 竞争路径词的 weight ==")
for w in set(TOP1):
    print("  %-4s weight=%s cost=%.2f" % (w, de.weight(w), rr._cost(w)))

print()
print("== 关键词对共现查询 ==")
pairs = [("年", "我国"), ("我国", "能够"), ("能够", "的"), ("的", "乡村"),
         ("乡村", "是"), ("是", "一些"), ("我", "给你"), ("给你", "个"),
         ("个", "东西"), ("东西", "测试"), ("测试", "一下"), ("那", "我")]
for a, b in pairs:
    c1 = rr.bigram.get((a, b), 0)
    c2 = rr.pre_word.get((a, b), 0) if rr.pre_word else 0
    c3 = rr.pre_char.get((a[-1], b[0]), 0) if rr.pre_char else 0
    print("  (%s,%s) user=%s pre_word=%s pre_char=%s" % (a, b, c1, c2, c3))
