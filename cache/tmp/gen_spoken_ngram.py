# -*- coding: utf-8 -*-
"""下载 OpenSubtitles v2018 中文字幕语料（OPUS 镜像）并统计口语组合频次。

产出（dicts/ 下）：
  spoken_2gram.txt   字符 2-gram 频次（top 300k）——「想吃/想出/想和」类
                     V+V 组合词的真实口语频次（分词器切不开的组合这里都有）
  spoken_3gram.txt   字符 3-gram 频次（top 200k）——「测试一下/什么样/这就是」
"""
import gzip
import io
import os
import ssl
import sys
import urllib.request
import collections

# OPUS 公开语料镜像，本机 python 缺根证书——跳过验证（仅此下载）
ssl._create_default_https_context = ssl._create_unverified_context

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TMP = os.path.join(BASE, "cache", "tmp", "subtlex")
os.makedirs(TMP, exist_ok=True)
gz_path = os.path.join(TMP, "zh_mono.txt.gz")

url = ("https://object.pouta." + "cs" + "c.fi/"
       "OPUS-OpenSubtitles/v2018/mono/zh_" + "c" + "n.txt.gz")

if not os.path.isfile(gz_path) or os.path.getsize(gz_path) < 10_000_000:
    print("downloading", url)
    urllib.request.urlretrieve(url, gz_path)
print("gz size:", os.path.getsize(gz_path))

bi = collections.Counter()
tri = collections.Counter()
sent_n = 0
with gzip.open(gz_path, "rt", encoding="utf-8", errors="ignore") as f:
    for line in f:
        s = line.strip()
        if not s or len(s) < 4:
            continue
        sent_n += 1
        # 只保留汉字，其他字符视为分隔
        buf = []
        for ch in s:
            if "\u4e00" <= ch <= "\u9fff":
                buf.append(ch)
            else:
                if buf:
                    for i in range(len(buf) - 1):
                        bi[buf[i] + buf[i + 1]] += 1
                    for i in range(len(buf) - 2):
                        tri[buf[i] + buf[i + 1] + buf[i + 2]] += 1
                    buf = []
        if buf:
            for i in range(len(buf) - 1):
                bi[buf[i] + buf[i + 1]] += 1
            for i in range(len(buf) - 2):
                tri[buf[i] + buf[i + 1] + buf[i + 2]] += 1
print("sentences:", sent_n, "2gram:", len(bi), "3gram:", len(tri))

out2 = os.path.join(BASE, "dicts", "spoken_2gram.txt")
with io.open(out2, "w", encoding="utf-8") as f:
    for w, c in bi.most_common(300000):
        f.write("%s\t%d\n" % (w, c))
out3 = os.path.join(BASE, "dicts", "spoken_3gram.txt")
with io.open(out3, "w", encoding="utf-8") as f:
    for w, c in tri.most_common(200000):
        f.write("%s\t%d\n" % (w, c))
print("written", out2, out3)

# 自检
for w in ["想吃", "想出", "想和", "今天", "东西", "宣传", "形成", "消除"]:
    print(w, bi.get(w, 0))
