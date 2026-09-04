# -*- coding: utf-8 -*-
"""统计重排器参数调优（数据驱动）。

rerank.py 里的 A / L / P / U / B 五个常数**不要凭直觉改**——它们的作用高度
非直觉：

- P（单字惩罚）要同时补偿两件事：①8105 字频是「字在语料中的总出现次数」，
  包含了该字在「包子/桌子/孩子」里的出现，而词的词频只统计独立成词，两者
  口径不一致、字频被系统性高估；②unigram 模型天然偏好「词数少」的切分。
- L（长词奖励）与 P 是联动的：L 大则偏好长词（词数少），L 小则退化为全单字。

本脚本在真实用例上扫参，选出准确率最高的组合。改 rerank.py 前先跑它。

用法：
    python tune.py            # 扫参并给出最优组合
    python tune.py --report   # 只用当前参数跑一遍，看每个用例的切分结果
"""

import argparse
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from dict_engine import DictEngine  # noqa: E402
import rerank  # noqa: E402

# 用例：(音节序列, 模式, 期望结果)。期望结果以底库实际收录为准。
CASES = [
    (list("wilygbz"), "ini", "我吃了一个包子"),
    ("zhan mu si".split(), "py", "詹姆斯"),
    ("wo shi zhong guo ren".split(), "py", "我是中国人"),
    ("ni hao".split(), "py", "你好"),
    ("wo chi le yi ge bao zi".split(), "py", "我吃了一个包子"),
    ("jin tian tian qi bu cuo".split(), "py", "今天天气不错"),
    ("xie xie ni".split(), "py", "谢谢你"),
    ("zhong guo".split(), "py", "中国"),
    ("wo men yi qi qu chi fan".split(), "py", "我们一起去吃饭"),
    ("ta shi yi ge hao ren".split(), "py", "他是一个好人"),
]


def make_rr():
    de = DictEngine(log=lambda *a: None)
    de.load_dir(os.path.join(BASE, "dicts"))
    return rerank.StatReranker(de, log=lambda *a: None), de


def score(rr, topn=1):
    ok = 0
    for keys, mode, want in CASES:
        got = rr.viterbi(keys, mode, topn)
        if want in got:
            ok += 1
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="只用当前参数跑一遍")
    args = ap.parse_args()

    rr, de = make_rr()
    print("底库 %d 词条\n" % de.size)

    if args.report:
        for keys, mode, want in CASES:
            got = rr.viterbi(keys, mode, 3)
            flag = "OK " if want in got else "MISS"
            print("%s %-24s %-3s 期望=%-14s 实得=%s" % (flag, "".join(keys), mode, want, got))
        print("\n当前参数 A=%.1f L=%.1f P=%.1f U=%.1f B=%.1f  命中 %d/%d"
              % (rerank.A, rerank.L, rerank.P, rerank.U, rerank.B, score(rr), len(CASES)))
        return

    best = None
    print("扫参中（%d 组 x %d 用例）..." % (6 * 5, len(CASES)))
    for L in (0.0, 3.0, 6.0, 9.0, 12.0):
        for P in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0):
            rerank.L, rerank.P = L, P
            rr._viterbi_cache.clear()
            s1 = score(rr, 1)
            s3 = score(rr, 3)
            print("  L=%-5.1f P=%-5.1f  top1=%2d/%d  top3=%2d/%d"
                  % (L, P, s1, len(CASES), s3, len(CASES)))
            key = (s1, s3, -abs(P - 4.0))  # 同分时取 P 接近 4 的（更平滑）
            if best is None or key > best[0]:
                best = (key, (L, P), s1, s3)
    (_, (L, P), s1, s3) = best
    print("\n最优：L=%.1f  P=%.1f   (top1 %d/%d, top3 %d/%d)" % (L, P, s1, len(CASES), s3, len(CASES)))
    print("把这两个值写回 rerank.py 的 L / P。")

    rerank.L, rerank.P = L, P
    rr._viterbi_cache.clear()
    print("\n===== 最优参数下的逐用例结果 =====")
    for keys, mode, want in CASES:
        got = rr.viterbi(keys, mode, 3)
        flag = "OK " if want in got else "MISS"
        print("%s %-24s %-3s 期望=%-14s 实得=%s" % (flag, "".join(keys), mode, want, got))


if __name__ == "__main__":
    main()
