# -*- coding: utf-8 -*-
"""生成拼音位置约束的 GBNF 语法：第 k 字的声母必须等于键码第 k 位。

每个位置 = 该声母所有常用汉字的枚举字符类。llama-cli --grammar-file 消费。
字集来源：底库单字表（word_py 里长度 1 的词条）+ 声母提取（pypinyin）。
"""
import os
import sys
import re

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from dict_engine import DictEngine

de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))

# 声母 -> 汉字集（底库单字，按词频排序取前 N）
# 小鹤简拼键位：zh→v、ch→i、sh→u——键码字母不等于拼音首字母，需映射
KEY2SM = {"v": "zh", "i": "ch", "u": "sh"}
SM2HZ = {}
for w, py in de.word_py.items():
    if len(w) != 1 or not ("\u4e00" <= w <= "\u9fff"):
        continue
    sm = py.split()[0][:2] if py.split()[0][:2] in ("zh", "ch", "sh") \
        else py.split()[0][0]
    SM2HZ.setdefault(sm, []).append(w)
# 底库 word_py 无序——按该字在词库中的总词频近似排序太重，直接保留全部；
# GBNF 枚举顺序影响 tie-break（先出现的优先），把高频字排前面：
# 用 SUBTLEX 字频表排序
SUB = os.path.join(BASE, "cache", "tmp", "subtlex", "SUBTLEX-CH-WF")
freq = {}
raw = open(SUB, "rb").read().decode("gb18030")
for line in raw.splitlines()[3:]:
    p = line.split("\t")
    if len(p) >= 2 and p[0]:
        try:
            freq[p[0]] = int(p[1])
        except ValueError:
            pass
for sm in SM2HZ:
    SM2HZ[sm] = sorted(set(SM2HZ[sm]), key=lambda c: -freq.get(c, 0))[:40]

KEYS = sys.argv[1] if len(sys.argv) > 1 else "nwgngdxcuyx"
out = os.path.join(BASE, "cache", "tmp", "subtlex", "cons_%s.gbnf" % KEYS)
parts = []
for k in KEYS:
    sm = KEY2SM.get(k, k)
    hzs = SM2HZ.get(sm, [])
    if not hzs:
        print("声母无字:", k, "->", sm)
        sys.exit(1)
    chars = "|".join('"%s"' % hz for hz in hzs)
    parts.append('(%s)' % chars)
gbnf = "root ::= %s\n" % " ".join(parts)
with open(out, "w", encoding="utf-8") as f:
    f.write(gbnf)
print("grammar:", out)
print("每位置字数:", [len(SM2HZ.get(KEY2SM.get(k, k), ())) for k in KEYS])
