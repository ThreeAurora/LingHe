# -*- coding: utf-8 -*-
"""锚点串接诊断：cores 收集了什么、关键键串的排名、串接每步走到哪。"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from dict_engine import DictEngine
from rerank import StatReranker, ANCHOR2_FREQ

de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))
rr = StatReranker(de)
rr.load(os.path.join(BASE, "dicts"))

for keys_str in ["c u y x", "d x", "x i", "x h", "x h c y", "n w g n g",
                 "n w g n g d x", "w j t", "w j t x i", "c u", "y x"]:
    lst = de.by_initial.get(keys_str) or []
    top = [w for _, w in lst[:5]]
    print("%-14s top5=%s" % (keys_str, top))

print()
# 复制 anchor_sentences 的 cores 收集逻辑（打印版）
for code, want in [("nwgngdxcuyx", "那我给你个东西测试一下"),
                   ("wjtxixhcy", "我今天想吃西湖醋鱼")]:
    keys = list(code)
    m = len(keys)
    cores = []
    for i in range(m):
        for L in (2, 3, 4):
            if i + L > m:
                continue
            lst = de.by_initial.get(" ".join(keys[i:i + L]))
            if not lst:
                continue
            if L >= 4:
                cores.append((i, i + L, lst[0][1]))
            elif L == 3:
                if len(lst) <= 8:
                    cores.append((i, i + L, lst[0][1]))
            else:
                for r in (0, 1):
                    if r < len(lst) and de.weight(lst[r][1]) >= ANCHOR2_FREQ:
                        cores.append((i, i + L, lst[r][1]))
    cores.sort(key=lambda a: a[0])
    print("码=%s 期望[%s]" % (code, want))
    print("  cores(%d):" % len(cores))
    for ai, aj, aw in cores:
        mark = " ◀◀◀" if aw in want else ""
        print("    [%d:%d) %s%s" % (ai, aj, aw, mark))
    print("  viterbi(nwgng) top2:", [s for s, _, _ in rr.viterbi(list("nwgng"), "ini", 2, ret_cost=True)] if code.startswith("n") else [s for s, _, _ in rr.viterbi(list("wjt"), "ini", 2, ret_cost=True)])
    print("  anchor产出:", [s for s, _, _ in rr.anchor_sentences(keys, "ini", 12)])
    print()
