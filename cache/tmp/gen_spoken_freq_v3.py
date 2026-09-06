# -*- coding: utf-8 -*-
"""口语词频表 v3：SUBTLEX-CH 词级频次 + 字幕 2-gram 组合词频 融合。

v2 教训：wordfreq 的 top_n_list 分词口径不含「想吃」类 V+V 组合，查询接口
给的 zipf 是 n-gram 平滑值。v3 数据源：
  1. SUBTLEX-CH-WF（Cai & Brysbaert 2010，影视字幕 33.5M 词次，99k 词条）
     —— 词级真实口语频次，书面词（县城=2、薪酬=0）被正确压制
  2. spoken_2gram.txt（字幕原文 2-gram）—— 分词器切不开的组合词
     （想吃/想出/想和）的真实字对频次
融合规则：SUBTLEX 优先，2-gram 兜底，value=真实频次。
"""
import io
import os

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUB = os.path.join(BASE, "cache", "tmp", "subtlex", "SUBTLEX-CH-WF")
NG2 = os.path.join(BASE, "dicts", "spoken_2gram.txt")
OUT = os.path.join(BASE, "dicts", "spoken_freq.txt")

tbl = {}
# 1) SUBTLEX 词级
raw = open(SUB, "rb").read().decode("gb18030")
for line in raw.splitlines()[3:]:  # 前 2 行元信息 + 1 行表头
    p = line.split("\t")
    if len(p) >= 3 and p[0] and "\u4e00" <= p[0][0] <= "\u9fff":
        try:
            c = int(p[1])
        except ValueError:
            continue
        if c >= 2:
            tbl[p[0]] = c
print("SUBTLEX words:", len(tbl))

# 2) 2-gram 组合词兜底（SUBTLEX 没有的才补）
# 跨词字对污染：2-gram 是「汉字相邻对」，包含大量跨词组合——「过的」（走过
# 的）、「跟你」、「想和」（想+和）频次虚高，会把真词（想吃）挤出口语序。
# 剔除含虚词字的组合（虚词跨词搭配无穷尽，无法从频次区分）。
VIRT = set("的了是在有和就都也还才跟你我这那它她他个把被往朝")
n2 = 0
for line in io.open(NG2, encoding="utf-8"):
    p = line.rstrip("\n").split("\t")
    if len(p) >= 2:
        w, c = p[0], int(p[1])
        if w in tbl or c < 5 or len(w) != 2:
            continue
        if any(ch in VIRT for ch in w):
            continue
        tbl[w] = c
        n2 += 1
print("2gram 补录:", n2)

with io.open(OUT, "w", encoding="utf-8") as f:
    f.write("# 口语词频表 v3（SUBTLEX-CH 词级 + 字幕 2-gram 组合词，词\\t频次）\n")
    for w, c in sorted(tbl.items(), key=lambda kv: -kv[1]):
        f.write("%s\t%d\n" % (w, c))
print("written", len(tbl), "->", OUT)

for probe in ["想吃", "想出", "想和", "东西", "大学", "形成", "宣传", "消除",
              "选出", "相处", "县城", "薪酬", "测试一下", "西湖醋鱼"]:
    print("%-6s -> %s" % (probe, tbl.get(probe)))
