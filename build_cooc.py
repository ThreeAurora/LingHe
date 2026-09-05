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


# ---------- 短语挖共现 ----------
# 词库本身就是一份「微语料」：吃包子 / 一个包子 / 看电影 这类多字短语在
# 库里存在，就蕴含其子词的相邻共现——(吃,包子) / (一个,包子) / (看,电影)。
# SIGHAN 是 1998 年新闻语料，动词-食物/器物这类口语搭配覆盖极差
# （(吃,饭)=0、(一个,包子)=0），短语挖矿恰好补上这块。
# 切分用「贪心 ≤2 字块」（与 caret_ctx.tail_words 同方案），保证挖出的
# 对与运行时多上文切词的粒度对得上。

def load_dict_words(dicts_dir):
    """收集词库全部纯汉字词条（挖矿的短语来源 + 切分的块校验集）。"""
    words = set()
    for name in sorted(os.listdir(dicts_dir)):
        if not name.endswith(".yaml"):
            continue
        path = os.path.join(dicts_dir, name)
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.rstrip("\r\n")
                    if not line or line[0] in "#.":
                        continue
                    w = line.split("\t")[0].strip()
                    if len(w) >= 2 and is_hanzi_word(w):
                        words.add(w)
        except OSError:
            continue
    return words


def mine_phrase_pairs(words):
    """把 ≥3 字短语切成 ≤2 字块，产出相邻 (上块, 下块)。"""
    for ph in words:
        if len(ph) < 3:
            continue
        blocks = []
        pos = 0
        ok = True
        while pos < len(ph):
            if pos + 2 <= len(ph) and ph[pos:pos + 2] in words:
                blocks.append(ph[pos:pos + 2])
                pos += 2
            elif is_hanzi_word(ph[pos]):
                blocks.append(ph[pos])
                pos += 1
            else:
                ok = False
                break
        if ok:
            for a, b in zip(blocks, blocks[1:]):
                yield a, b


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

    # 短语挖矿：词库多字短语 → 相邻子词对（出现即收，词条是人工炼过的）
    # 计数语义修正（2026-09-05）：挖矿数的是「含该相邻对的短语个数」，是
    # 构词产出率，不是文本频率——腾讯库 19 个「一个不X」成语家族会把
    # (一个,不) 灌到 19，压过真实文本对 (一个,包子)=1，整句切分因此跑偏
    # （ilygbz→成立一个不在 案）。乘 0.25 并入：短语存在=可能性证据，
    # 语料共现才是频次证据，两者不再同权。
    words = load_dict_words(os.path.join(BASE, "dicts"))
    mined = Counter()
    for a, b in mine_phrase_pairs(words):
        if len(a) > 1 or len(b) > 1:  # 至少一侧是多字块，纯字对留给字级表
            mined[(a, b)] += 1
    for k, v in mined.items():
        kept[k] = kept.get(k, 0) + v * 0.25  # 浮点：0.25 取整会把 (一个,包子)=1 这类关键小对抹成 0
    print("短语挖矿：%d 词条 -> %d 对（x0.25 并入词级表）" % (len(words), len(mined)))
    for a, b in (("吃", "包子"), ("一个", "包子"), ("看", "电影"), ("一个", "不")):
        print("  (%s,%s) = %s" % (a, b, kept.get((a, b), 0)))
    if not kept:
        print("没有可用共现，退出")
        return 1

    with open(OUT_BIN, "wb") as f:
        marshal.dump({"word": kept, "char": kept_char}, f)
    top = sorted(kept.items(), key=lambda kv: -kv[1])
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("# 预训练共现表（SIGHAN Bakeoff 2005 msr+pku 分词语料统计，非商业许可）\n")
        f.write("# + 词库短语挖矿（多字短语的相邻子词对，calibrate 后词库 2026-09-05）\n")
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
