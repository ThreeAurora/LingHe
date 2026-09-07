# -*- coding: utf-8 -*-
"""探针：复刻 compute('bz') 链路，看上文「我吃了一个」时包子排第几。

场景 = 用户原话：打完「我吃了一个」，输入 bz，包子应到固定词库后第一位。
"""
import sys
sys.path.insert(0, r'e:\CCSpace\projects\2026\09\linghe')
from dict_engine import DictEngine
from rerank import StatReranker
from mabiao import MaBiao
from xiaohe import decode_syllable
import caret_ctx

BASE = r'e:\CCSpace\projects\2026\09\linghe'

de = DictEngine(log=lambda *a: None)
de.load_dir(BASE + r'\dicts')
rr = StatReranker(de, log=lambda *a: None)
rr.load(BASE + r'\dicts')
mb = MaBiao()
mb.load_dir(BASE + r'\mabiao')

ctx_text = "我吃了一个"
prevs = caret_ctx.tail_words(ctx_text, de.word_py, 3)
print("tail_words(%r) = %r" % (ctx_text, prevs))

code = "bz"
# ---- 底库动态区（compute 的 dict_cands 路径）----
pool = {}
dict_cands = de.lookup_initial(" ".join(code), 90)
for w in dict_cands:
    pym = de.word_py.get(w) or ""
    m = max(1, len(pym.split()))
    pool[w] = rr.score_word(w, prevs) / m
# ---- 整句池（viterbi ini）----
vpool = rr.viterbi(list(code), "ini", 8, ret_cost=True, prev=prevs[0] if prevs else "")
for s, c, m in vpool:
    cps = c / max(1, m)
    if s not in pool or cps < pool[s]:
        pool[s] = cps

ranked = sorted(pool.items(), key=lambda kv: kv[1])
for i, (w, s) in enumerate(ranked[:30]):
    mark = "  <<< 包子" if w == "包子" else ""
    print("%2d  %s %.2f%s" % (i + 1, w, s, mark))
print()
print("包子排名: ", [w for w, _ in ranked].index("包子") + 1 if "包子" in [w for w, _ in ranked] else "不在池内")

# 分解包子得分，看是哪个环节没加分
print()
print("包子 score_word 拆解:")
print("  预训练词级 (一个,包子)=", rr.pre_word.get(("一个", "包子"), 0),
      " (吃了,包子)=", rr.pre_word.get(("吃了", "包子"), 0))
print("  score_word(包子, prevs) =", rr.score_word("包子", prevs))
print("  score_word(包子, [])    =", rr.score_word("包子", []))