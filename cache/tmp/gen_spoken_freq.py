# -*- coding: utf-8 -*-
"""生成口语词频表 v2：底库词条驱动 × wordfreq zipf 查询 → dicts/spoken_freq.txt。

v1 教训：wordfreq 的 top_n_list 是分词口径词表，「想吃」类 V+V 组合词不在
其中（查询接口给的 4.50 是 n-gram 平滑值，没写进表）——导致口语表对最关键
的口语组合词集体失明。v2 直接遍历底库全部词条逐个查询，底库有的词就有口语值。
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from dict_engine import DictEngine
from wordfreq import zipf_frequency

de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))

out = os.path.join(BASE, "dicts", "spoken_freq.txt")
n = 0
with open(out, "w", encoding="utf-8") as f:
    f.write("# 口语词频表 v2（底库词条 × wordfreq zipf×100，词\\tzipf100）\n")
    for w in de.word_py:
        if not (1 <= len(w) <= 8):
            continue
        if not all("\u4e00" <= c <= "\u9fff" for c in w):
            continue
        z = zipf_frequency(w, 'zh')
        if z < 2.0:
            continue
        f.write("%s\t%d\n" % (w, int(z * 100)))
        n += 1
print("written %d words -> %s" % (n, out))
# 自检
for probe in ['想吃', '东西', '选出', '相处', '形成', '测试一下', '西湖醋鱼']:
    v = None
    try:
        with open(out, encoding="utf-8") as f:
            for line in f:
                if line.startswith(probe + "\t"):
                    v = line.strip()
                    break
    except OSError:
        pass
    print(probe, "->", v)
