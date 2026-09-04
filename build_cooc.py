# -*- coding: utf-8 -*-
"""从已分词语料统计词级 bigram 共现表（预训练判别式重排的燃料）。

用途：rerank.py 的上下文打分原先只有「用户在线学习」一个来源，冷启动为零，
简拼整句（wilygbz 类）在 unigram 歧义面前没有裁决力。本脚本把公开分词语料
（SIGHAN Bakeoff 2005：微软研究院 msr_training + 北京大学 pku_training，
非商业许可）统计成词级共现表，作为预训练先验注入 rerank——**判别式**路线：
对候选打分，不生成 token，端侧毫秒级。

输入：cache/tmp/corpus/msr_training.utf8 与 pku_training_utf8.txt
      （空格分词、UTF-8；标点 token 视为句界，不统计跨句共现）
输出：dicts/cooc_pre.bin（marshal 的 {(上词,下词): 次数}，rerank 直接加载）
      同时落一份可读头部 dicts/cooc_pre.txt 说明来源（防丢哲学：能再生）。

裁剪策略：只留次数>=2 的对（单次共现多为噪声），纯汉字词才收——与底库
收录口径一致。

用法：python build_cooc.py
"""

import marshal
import os
import sys
from collections import Counter

BASE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(BASE, "cache", "tmp", "corpus")
OUT_BIN = os.path.join(BASE, "dicts", "cooc_pre.bin")
OUT_TXT = os.path.join(BASE, "dicts", "cooc_pre.txt")

MIN_COUNT = 2  # 只留出现两次以上的共现（单次多为噪声）


def is_hanzi_word(tok):
    return bool(tok) and all(0x4E00 <= ord(ch) <= 0x9FFF for ch in tok)


def is_punct(tok):
    return not is_hanzi_word(tok)


def iter_pairs(lines):
    """产出 (上词, 下词) 与 (上尾字, 下首字)。标点断句：跨标点不连。

    词级对是主表；字级对（跨词边界的尾字→首字）是补充——它对词级表没覆盖
    的词组合有泛化力（语料里没有「包子」这个词，但「子」跟在名词后的模式在）。
    """
    for line in lines:
        toks = line.split()
        prev = None
        for tok in toks:
            if is_punct(tok):
                prev = None  # 句界
                continue
            if not is_hanzi_word(tok):
                prev = None
                continue
            if prev:
                yield prev, tok
                yield prev[-1], tok[0]  # 字级：跨词边界
            prev = tok


def main():
    files = [
        ("msr_training.utf8", "utf-8"),
        ("pku_training_utf8.txt", "utf-8"),
    ]
    pair_count = Counter()
    char_count = Counter()
    total_pairs = 0
    for name, enc in files:
        path = os.path.join(CORPUS, name)
        if not os.path.isfile(path):
            print("[跳过] 缺 %s" % path)
            continue
        with open(path, "r", encoding=enc, errors="ignore") as f:
            n = c = 0
            for a, b in iter_pairs(f):
                if len(a) == 1 and len(b) == 1:
                    char_count[(a, b)] += 1
                    c += 1
                else:
                    pair_count[(a, b)] += 1
                    n += 1
            total_pairs += n + c
            print("[语料] %s: 词级 %d / 字级 %d（累计唯一 词%d 字%d）"
                  % (name, n, c, len(pair_count), len(char_count)))

    kept = {k: v for k, v in pair_count.items() if v >= MIN_COUNT}
    kept_char = {k: v for k, v in char_count.items() if v >= MIN_COUNT}
    print("词级唯一 %d -> 保留 %d；字级唯一 %d -> 保留 %d"
          % (len(pair_count), len(kept), len(char_count), len(kept_char)))
    if not kept:
        print("没有可用共现，退出")
        return 1

    with open(OUT_BIN, "wb") as f:
        marshal.dump({"word": kept, "char": kept_char}, f)
    top = sorted(kept.items(), key=lambda kv: -kv[1])
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("# 预训练共现表（SIGHAN Bakeoff 2005 msr+pku 分词语料统计，非商业许可）\n")
        f.write("# 真数据在 cooc_pre.bin（marshal：{word:{}, char:{}}）。本文件仅 top 千条供审阅。\n")
        for (a, b), c in top[:1000]:
            f.write("%s\t%s\t%d\n" % (a, b, c))
    print("已写 %s（词%d 字%d）与 %s" % (OUT_BIN, len(kept), len(kept_char), OUT_TXT))

    # 自检：整句切分关键共现是否在表里
    checks = [("我", "吃"), ("吃", "了"), ("了", "一个"), ("一个", "包子"), ("我", "是"), ("今天", "天气")]
    for a, b in checks:
        cw = kept.get((a, b), 0)
        cc = kept_char.get((a, b), 0)
        print("  %s -> %s : 词%d 字%d" % (a, b, cw, cc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
