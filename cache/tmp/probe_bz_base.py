# -*- coding: utf-8 -*-
"""探针：复刻 compute('bz') 的 base 装配全链路，定位包子在哪个环节被切。"""
import sys
sys.path.insert(0, r'e:\CCSpace\projects\2026\09\linghe')
from dict_engine import DictEngine
from rerank import StatReranker
from mabiao import MaBiao
import caret_ctx

BASE = r'e:\CCSpace\projects\2026\09\linghe'
de = DictEngine(log=lambda *a: None)
de.load_dir(BASE + r'\dicts')
rr = StatReranker(de, log=lambda *a: None)
rr.load(BASE + r'\dicts')
mb = MaBiao()
mb.load_dir(BASE + r'\mabiao')

code = "bz"
n_pool = 45
ctx_text = "我吃了一个"
prevs = caret_ctx.tail_words(ctx_text, de.word_py, 3)
prev = prevs[0] if prevs else ""

print("== mb_part（码表固频区）==")
fused = []
mb_exact = list(mb.exact(code))
seen = set(fused) | set(mb_exact)
print("mb_exact =", mb_exact)
mb_prefix = []
for w in mb.prefix(code, n_pool * 2):
    if w not in seen:
        mb_prefix.append(w)
        seen.add(w)
print("mb_prefix 前几个 =", mb_prefix[:8])
mb_part = (fused + mb_exact + mb_prefix)[:24]
print("n_mb =", len(mb_part), " mb_part =", mb_part[:12])
n_mb = len(mb_part)

print()
print("== 动态区 dict_side ==")
pool = {}
dict_cands = de.lookup_initial(" ".join(code), 90)
for w in dict_cands:
    if w in seen or w in pool:
        continue
    pym = de.word_py.get(w) or ""
    m = max(1, len(pym.split()))
    pool[w] = rr.score_word(w, prevs) / m
ranked_pool = sorted(pool.items(), key=lambda kv: kv[1])
dict_side = [w for w, _ in ranked_pool if len(w) < 5][: n_pool - n_mb]
print("dict_side 数量 =", len(dict_side), " 包子在 dict_side 里 =", "包子" in dict_side)
print("dict_side 前 5 =", dict_side[:5])
for i, (w, s) in enumerate(ranked_pool):
    if w == "包子":
        print("包子在 ranked_pool 第 %d 位，截断线 = %d" % (i + 1, n_pool - n_mb))
        break

print()
print("== 关键参数核对 ==")
print("n_pool 默认 =", n_pool, "（config.json 里的实际值需另查）")
import json
cfg = json.load(open(BASE + r'\config.json', encoding='utf-8'))
print("config candidate.pool =", cfg.get("candidate", {}).get("pool", "?"))
print("neural 段 =", cfg.get("neural", {}))