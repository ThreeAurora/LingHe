# -*- coding: utf-8 -*-
"""主人三个用例的诊断：ilygbz/hubgvn/hutil + 联想链覆盖检查。"""
import sys, os
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
os.chdir(BASE)

from dict_engine import DictEngine
from rerank import StatReranker, MAX_SPAN, SPAN_CANDS

de = DictEngine(log=lambda *a: None)
de.load_dir("dicts")
rr = StatReranker(de, log=lambda *a: None)
rr.load("dicts")

# --- 0) 关键字的拼音/简拼索引情况 ---
print("=== 单字注音检查 ===")
for ch in ["差", "智", "能", "还", "是", "太", "吃", "了", "包", "子", "够", "不"]:
    py = de.word_py.get(ch, "??")
    ini = " ".join(w[0] for w in py.split()) if py != "??" else "??"
    print("  %s: py=%-10s ini=%s  weight=%s" % (ch, py, ini, de.weight(ch)))

print("\n=== 词的简拼索引检查 ===")
for w in ["还是", "不够", "智能", "一个", "包子", "吃了", "太差"]:
    py = de.word_py.get(w, "??")
    ini = " ".join(x[0] for x in py.split()) if py != "??" else "??"
    hit = de.by_initial.get(ini)
    top = [x[1] for x in hit[:8]] if hit else None
    rank = next((i for i, x in enumerate(hit) if x[1] == w), -1) if hit else -1
    print("  %-4s ini=%-8s span内排名=%d  top8=%s" % (w, ini, rank, top))

print("\n=== 共现对覆盖 ===")
pairs = [("吃","了"),("了","一个"),("一个","包子"),("吃","什么"),("什么","时候"),
         ("还是","不"),("还是","不够"),("不够","智能"),("不","够"),("是","不"),
         ("还是","太"),("太","差"),("差","了"),("是","太"),("智能","还")]
for a, b in pairs:
    cu = rr.bigram.get((a, b), 0)
    cp = rr.pre_word.get((a, b), 0)
    cc = rr.pre_char.get((a[-1], b[0]), 0)
    print("  (%s,%s): user=%d pre_word=%d pre_char(%s,%s)=%d" % (a, b, cu, cp, a[-1], b[0], cc))

print("\n=== Viterbi 三用例（简拼） ===")
cases = [("ilygbz", "吃了一个包子"), ("hubgvn", "还是不够智能"), ("hutil", "还是太差了"),
         ("wilygbz", "我吃了一个包子")]
for code, want in cases:
    got = rr.viterbi(list(code), "ini", 5)
    mark = "✓" if got and got[0] == want else "✗"
    print("  %s %s -> %s" % (mark, code, got))

print("\n=== 联想链检查 ===")
for w in ["吃", "还是", "了", "一个"]:
    print("  next_word(%s) = %s" % (w, rr.next_word(w, 5)))
