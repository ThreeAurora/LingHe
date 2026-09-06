# -*- coding: utf-8 -*-
"""骨架通道产出审计：viterbi/anchor/max_match 三通道各自产出什么，目标句在不在。"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from dict_engine import DictEngine
from rerank import StatReranker

de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))
rr = StatReranker(de)
rr.load(os.path.join(BASE, "dicts"))

for code, want in [
    ("nwgngdxcuyx", "那我给你个东西测试一下"),
    ("wjtxixhcy", "我今天想吃西湖醋鱼"),
]:
    print("=" * 60)
    print("码=%s 期望[%s]" % (code, want))
    vv = rr.viterbi(list(code), "ini", 24, ret_cost=True)
    print("viterbi top6:", [s for s, _, _ in vv[:6]])
    an = rr.anchor_sentences(list(code), "ini", 12)
    print("anchor %d条:" % len(an), [s for s, _, _ in an])
    mm = rr.max_match_sentences(list(code), "ini")
    print("max_match %d条:" % len(mm), [s for s, _, _ in mm])
    hit = [ch for ch in (an + mm) if ch[0] == want]
    print("目标句在骨架通道: %s" % ("✓" if hit else "✗"))
    if not hit:
        # 逐段诊断：目标句的切分每段在各跨度键串的排名
        print("  逐段排名诊断:")
        import xiaohe
        segs = [("nw", "那"), ("g", "给"), ("n", "你"), ("g", "个"),
                ("dx", "东西"), ("cyx", "测试一下")]
        for keys, w in segs:
            lst = de.by_initial.get(" ".join(keys))
            rank = next((i for i, (_, x) in enumerate(lst) if x == w), None) if lst else None
            print("    %-4s %-6s rank=%s (候选数%d)" % (
                keys, w, rank, len(lst) if lst else 0))
