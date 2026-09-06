# -*- coding: utf-8 -*-
"""wjtxixhcy 深度诊断：boost 是否生效、目标路径死在哪层。"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from dict_engine import DictEngine
from rerank import StatReranker, LONG_SPAN_BOOST

de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))
rr = StatReranker(de)
rr.load(os.path.join(BASE, "dicts"))

out = rr.viterbi(list("wjtxixhcy"), "ini", 24, ret_cost=True)
print("== viterbi top24 ==")
for s, c, m in out[:24]:
    print("  %8.2f 每字%6.2f  %s" % (c, c / max(1, m), s))

print()
print("== 手动目标路径（含 boost）==")
for words in (["我", "今天", "想吃", "西湖醋鱼"],
              ["我", "今天", "想吃", "西湖", "醋鱼"],
              ["我", "今天", "想吃", "西湖", "醋", "鱼"]):
    total = 0.0
    prev = ""
    parts = []
    for w in words:
        c = rr._cost(w)
        b = rr._bi_bonus_span(prev, w)
        tag = ""
        if len(w) >= 3:
            py = de.word_py.get(w) or ""
            key = " ".join(rr._sm_key(s) if hasattr(rr, "_sm_key") else s for s in py.split()) if False else None
        parts.append("%s[%.1f%+.1f]" % (w, c, -b))
        total += c - b
        prev = w
    # 手动补 boost：西湖醋鱼 span4 rank1
    n_boost = sum(1 for w in words if len(de.word_py.get(w) or "".split()) >= 3)
    print("  总%8.2f  %s" % (total, " ".join(parts)))

print()
print("== span 候选池检查 ==")
for span in (("x h c y",), ("x i",), ("j t",)):
    k = span[0]
    lst = de.by_initial.get(k, ())
    print("  %-10r -> %s" % (k, [w for _, w in lst[:6]]))

print()
print("== 西湖醋鱼 boost 检查 ==")
print("  LONG_SPAN_BOOST =", LONG_SPAN_BOOST)
lst = de.by_initial.get("x h c y", ())
print("  xhcy top3:", [w for _, w in lst[:3]])
