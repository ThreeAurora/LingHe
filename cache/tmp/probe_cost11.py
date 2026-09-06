# -*- coding: utf-8 -*-
"""11 键代价解剖：手动切分目标句逐段算代价，对比 viterbi 头部路径。"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

import rerank as R
from dict_engine import DictEngine
from rerank import StatReranker

de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))
rr = StatReranker(de)
rr.load(os.path.join(BASE, "dicts"))

CODE = "nwgngdxcuyx"
WANT = "那我给你个东西测试一下"

paths = [
    ["那", "我", "给你", "个", "东西", "测试", "一下"],
    ["那我", "给你", "个", "东西", "测试", "一下"],
    ["那", "我给你", "个", "东西", "测试", "一下"],
    ["那", "我给你个", "东西", "测试", "一下"],
    ["那", "我", "给你", "个东西", "测试一下"],
    ["那", "我", "给你", "个", "东西", "测试一下"],
    ["那我", "给你", "个东西", "测试", "一下"],
]

print("== 手动路径代价分解（viterbi top1 每字均值 -14.1/11≈-1.28）==")
for words in paths:
    total = 0.0
    prev = ""
    parts = []
    ok = True
    for w in words:
        py = de.word_py.get(w) or ""
        inis = " ".join(R._sm_key(s) if hasattr(R, "_sm_key") else s[0] for s in py.split())
        # 校验该词的声母串确实匹配码段（非必须，仅供展示）
        c = rr._cost(w)
        b = rr._bi_bonus_span(prev, w)
        step = c - b
        parts.append("%s:%.1f%+.1f" % (w, c, -b))
        total += step
        prev = w
    n = len("".join(words))
    print("  总%8.2f 每字%6.2f  %s" % (total, total / n, "  ".join(parts)))

print()
print("== viterbi 头部路径（beam=16）==")
out = rr.viterbi(list(CODE), "ini", 16, ret_cost=True)
for s, c, m in out[:10]:
    print("  %8.2f 每字%6.2f  %s" % (c, c / max(1, m), s))

print()
print("== 目标句各词 cost 单查 ==")
for w in ("那", "我", "给你", "个", "东西", "测试", "一下", "那我", "我给你个", "个东西", "测试一下"):
    print("  %-6s cost=%7.2f weight=%s user=%s" % (w, rr._cost(w), de.weight(w), de.user_count(w)))
