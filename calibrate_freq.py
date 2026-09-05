# -*- coding: utf-8 -*-
"""跨词库词频定标（数据驱动，替代拍脑袋 ×N）。

问题
----
雾凇 base 的词频是真实语料统计（10 万~2000 万量纲），万象的词频是
AI 炼制的小量纲（1~3 万）。两者混池比价必须换到同一量纲，但**任何
常数倍放大都是猜**：库内保序，跨库对应关系没有依据，错排只是换位置。

正确做法：用数据定标
--------------------
1. 库间回归：取雾凇 ∩ 万象的共有词，最小二乘拟合
       log(freq_ice) = log(a) + b · log(freq_wx)
   得到万象 → 雾凇量纲的映射 g(wx) = a·wx^b；
2. 语料重计频：SIGHAN 分词语料（空格分隔，直接数）里出现过的词，
   用真实计数再经 语料→雾凇 映射 f(cnt)（同样交集回归得到）；
3. 万象词频 = f(语料计数) 若语料见过（真实依据最强），
            否则 g(万象词频)（库内相对位置映射），
   并设下限 FLOOR 防止库内精选冷词被压成 1。

用法：python calibrate_freq.py
输出：dicts/wanxiang_inc.dict.yaml（重写权重列）+ 校准质量报告
"""
import io
import math
import os
import sys
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
SRC_CONV = os.path.join(BASE, "cache", "tmp", "wx_jichu_conv.txt")   # 万象全量（原始频）
SRC_ZI = os.path.join(BASE, "cache", "tmp", "wx_zi_conv.txt")
SRC_LX = os.path.join(BASE, "cache", "tmp", "wx_lianxiang_conv.txt")
OUT = os.path.join(BASE, "dicts", "wanxiang_inc.dict.yaml")
CORPUS = os.path.join(BASE, "cache", "tmp", "corpus", "msr_training.utf8")
FLOOR = 200  # 万象精选词的频下限（低于此的常用词会被任何真实词压死）


def load_pairs(path):
    """词 -> (拼音, 原始频)。只取 3 列行。"""
    d = {}
    with io.open(path, encoding="utf-8") as f:
        for ln in f:
            p = ln.rstrip("\n").split("\t")
            if len(p) == 3 and p[2].isdigit():
                d.setdefault(p[0], (p[1], int(p[2])))
    return d


def loglog_fit(pairs):
    """pairs: [(x, y)]，拟合 log y = log a + b log x，返回 (a, b, r2)。"""
    n = len(pairs)
    if n < 10:
        return 1.0, 1.0, 0.0
    sx = sy = sxx = syy = sxy = 0.0
    for x, y in pairs:
        lx, ly = math.log(x), math.log(y)
        sx += lx; sy += ly; sxx += lx * lx; syy += ly * ly; sxy += lx * ly
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    la = (sy - b * sx) / n
    a = math.exp(la)
    my, mx = sy / n, sx / n
    ss_tot = syy - n * my * my
    ss_res = syy - 2 * b * sxy - 2 * la * sy + n * la * la + b * b * sxx \
             + 2 * b * la * sx
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return a, b, max(0.0, r2)


def corpus_counts():
    """SIGHAN msr 语料是空格分好词的，直接数。文件尾部可能有断点续传坏字节。"""
    cnt = Counter()
    with io.open(CORPUS, encoding="utf-8", errors="ignore") as f:
        for ln in f:
            for tok in ln.split():
                tok = tok.strip()
                if tok and len(tok) <= 8 and not tok.isspace():
                    cnt[tok] += 1
    return cnt


def main():
    ice = {w: c for w, (p, c) in
           load_pairs(os.path.join(BASE, "dicts", "base.dict.yaml")).items()}
    wx, PY = {}, {}
    for p in (SRC_CONV, SRC_ZI, SRC_LX):
        for w, (py, c) in load_pairs(p).items():
            if w not in wx:
                wx[w], PY[w] = c, py  # 多来源取先见者（jichu 优先）
    print("雾凇 %d 词，万象 %d 词" % (len(ice), len(wx)))

    # 1) 库间回归（交集）
    both = [(wx[w], ice[w]) for w in wx if w in ice and wx[w] > 0 and ice[w] > 0]
    a, b, r2 = loglog_fit(both)
    print("库间回归 交集 %d 词: freq_ice = %.2f * wx^%.3f   (R²=%.3f)"
          % (len(both), a, b, r2))
    for w in ("人工智能", "经济", "发展", "公司", "智能"):
        if w in wx and w in ice:
            print("   样例 %-6s wx=%-8d 实际ice=%-9d 映射=%.0f"
                  % (w, wx[w], ice[w], a * wx[w] ** b))

    # 2) 语料计数 + 语料→雾凇回归
    cc = corpus_counts()
    print("语料计频 %d 个词条" % len(cc))
    both_c = [(cc[w], ice[w]) for w in cc if w in ice and ice[w] > 0]
    a2, b2, r22 = loglog_fit(both_c)
    print("语料回归 交集 %d 词: freq_ice = %.2f * cnt^%.3f   (R²=%.3f)"
          % (len(both_c), a2, b2, r22))

    # 3) 生成定标后的词库
    n_corpus = n_reg = n_floor = 0
    out = ["# 万象增量词库（convert_wanxiang.py 转换 + calibrate_freq.py 定标）",
           "# 权重=f(语料真实计数) 优先，否则 g(万象词频) 库间回归映射，下限 %d" % FLOOR,
           "# 量纲与雾凇 base 对齐（log-log 回归数据驱动，非常数倍猜测）"]
    for w, c in wx.items():
        cnt = cc.get(w, 0)
        if cnt >= 3:
            freq = a2 * (cnt + 1) ** b2
            n_corpus += 1
        else:
            freq = a * (c + 1) ** b
            n_reg += 1
        if freq < FLOOR:
            freq = FLOOR
            n_floor += 1
        py = None
        out.append("%s\t%s\t%d" % (w, PY.get(w, ""), int(freq)))
    # 拼音从转换文件补（上面只 load 了词和频，需要拼音列）
    io.open(OUT, "w", encoding="utf-8").write("\n".join(out) + "\n")
    print("完成：语料定标 %d、回归映射 %d、触底 %d → %s"
          % (n_corpus, n_reg, n_floor, OUT))


if __name__ == "__main__":
    main()
