# -*- coding: utf-8 -*-
"""定位 11 键目标句的 viterbi 断点：逐跨度词 + 逐单字查索引排名。"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from dict_engine import DictEngine

de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))

# 目标句的所有可能切分跨度（含单字兜底）
spans = [
    ("n w", "那我"), ("n", "那"), ("w", "我"),
    ("g n", "给你"), ("g", "给"), ("n", "你"),
    ("g", "个"),
    ("d x", "东西"), ("d", "东"), ("x", "西"),
    ("c u", "测试"), ("c", "测"), ("u", "试"),
    ("y x", "一下"), ("y", "一"), ("x", "下"),
    ("w g n", "我给你"), ("w g n g", "我给你个"),
    ("g d x", "个东西"), ("g d x c u", "个东西测试"),
    ("d x c u", "东西测试"), ("c u y x", "测试一下"),
    ("n g", "那个"), ("n g d", "那个的"), ("d", "的"),
]

print("== 跨度词索引抽查 ==")
for key, word in spans:
    lst = de.by_initial.get(key, ())
    rank = None
    for i, (_, w) in enumerate(lst):
        if w == word:
            rank = i + 1
            break
    total = len(lst)
    flag = "OK " if rank else "MISS"
    print("  %s %-10s in %-10r rank=%s/%s" % (flag, word, key, rank, total))

# 单字在各自声母池的排名（fallback 通道 limit=4，正常通道 48）
print("== 单字排名（fallback 只取前4！）==")
for sm, ch in (("n", "那"), ("w", "我"), ("g", "给"), ("n", "你"), ("g", "个"),
               ("d", "东"), ("x", "西"), ("c", "测"), ("u", "试"),
               ("y", "一"), ("x", "下"), ("u", "是"), ("s", "所"), ("s", "四")):
    lst = de.by_initial.get(sm, ())
    rank = None
    for i, (_, w) in enumerate(lst):
        if w == ch:
            rank = i + 1
            break
    print("  %-2s 声母池 %-3s rank=%s/%s" % (sm, ch, rank, len(lst)))
