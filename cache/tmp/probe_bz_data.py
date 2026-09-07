# -*- coding: utf-8 -*-
"""探针：查看 bz 键位各词权重 / 包子共现证据，确认排序瓶颈。"""
import sys
sys.path.insert(0, r'e:\CCSpace\projects\2026\09\linghe')
from dict_engine import DictEngine
from rerank import StatReranker

BASE = r'e:\CCSpace\projects\2026\09\linghe'
de = DictEngine(log=lambda *a: None)
de.load_dir(BASE + r'\dicts')
rr = StatReranker(de, log=lambda *a: None)
rr.load(BASE + r'\dicts')

print("== bz 键位 top40 权重 ==")
cands = de.lookup_initial("b z", 90)
for i, w in enumerate(cands[:40]):
    print("%2d %-6s weight=%-9s user=%-3s spoken=%s" % (
        i + 1, w, de.weight(w), de.user_count(w), rr.spoken.get(w, "-")))
print()
print("包子 weight =", de.weight("包子"), " 口语频 =", rr.spoken.get("包子"))

print()
print("== 包子相关共现证据 ==")
for k in [("一个", "包子"), ("吃了", "包子"), ("吃", "包子"), ("了", "包子")]:
    print("pre_word[%r] = %s" % (k, rr.pre_word.get(k, 0)))
for k in [("了", "包"), ("吃", "包"), ("个", "包"), ("一", "包")]:
    print("pre_char[%r] = %s" % (k, rr.pre_char.get(k, 0)))

print()
print("== 我吃了的 tail_words 变体 ==")
import caret_ctx
for txt in ["我吃了一个", "吃了一个", "吃了一个包子", "我吃了"]:
    print(repr(txt), "->", caret_ctx.tail_words(txt, de.word_py, 3))

print()
print("== 候选各词在 bz 的 score_word（有/无上文对比）==")
prevs = ["一个", "吃了", "我"]
for w in ["不再", "不足", "不在", "不止", "不吃", "被迫", "包子"]:
    with_c = rr.score_word(w, prevs)
    without = rr.score_word(w, [])
    print("%-6s 有上文=%9.2f  无上文=%9.2f  diff=%+.2f" % (w, with_c, without, with_c - without))