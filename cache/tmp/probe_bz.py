# -*- coding: utf-8 -*-
"""主人的三场景验证：同一个 bz，不同上文 → 包子/杯子/豹子。"""
import sys, os, time
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
os.chdir(BASE)

from dict_engine import DictEngine
import rerank

de = DictEngine(log=lambda *a: None); de.load_dir("dicts")
rr = rerank.StatReranker(de, log=lambda *a: None); rr.load("dicts")

print("=== bigram 表覆盖检查（上文词 → 目标词）===")
for pair in [("一个", "包子"), ("一个", "杯子"), ("一个", "豹子"),
             ("了", "包子"), ("碎", "杯子")]:
    u = rr.bigram.get(pair, 0)
    pw = rr.pre_word.get(pair, 0)
    pc = rr.pre_char.get((pair[0][-1], pair[1][0]), 0)
    print("  (%s,%s): user=%d pre_word=%d pre_char=%d" % (pair[0], pair[1], u, pw, pc))

def pool_for(code, ctx):
    """复刻 compute 的同池核心：底库词 + 整句，每音节均代价，带上文。"""
    n = len(code)
    pool = {}
    if n >= 2:  # 主引擎已放宽到 2 键（bz→包子 是最高频场景）
        for w in de.lookup_initial(" ".join(code), 45):
            if w in pool:
                continue
            pym = de.word_py.get(w) or ""
            m = max(1, len(pym.split()))
            pool[w] = rr.score_word(w, ctx) / m
    for s, c, m in rr.viterbi(list(code), "ini", 5, ret_cost=True, prev=ctx):
        cps = c / max(1, m)
        if s not in pool or cps < pool[s]:
            pool[s] = cps
    return [w for w, _ in sorted(pool.items(), key=lambda kv: kv[1])][:6]

print("\n=== 同键 bz，四种上文（上下文条件预测）===")
for ctx in ["", "一个", "了", "碎了"]:
    print("  上文=%-4s bz -> %s" % (ctx or "(无)", pool_for("bz", ctx)))

print("\n=== 用户学习演示（打一次就准）===")
print("  学习前     上文=一个 ->", pool_for("bz", "一个")[:4])
rr.learn("一个", "豹子")
rr._cache_ver += 1  # bigram 变了，整句缓存失效
print("  学(一个→豹子)后 ->", pool_for("bz", "一个")[:4])
