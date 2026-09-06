# -*- coding: utf-8 -*-
"""整句神经裁决端到端验证：主人两个用例 + 词库覆盖检查。"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

import threading
from dict_engine import DictEngine
from rerank import StatReranker
from neural_rerank import NeuralReranker
from xiaohe import decode_syllable

de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))
rr = StatReranker(de)
rr.load(os.path.join(BASE, "dicts"))

print("== 词库覆盖抽查 ==")
for key, word in (("x h c y", "西湖醋鱼"), ("x h", "西湖"), ("j t", "今天"),
                  ("x i", "想吃"), ("c y", "醋鱼"), ("w j", "我们")):
    lst = de.by_initial.get(key, ())
    rank = next((i + 1 for i, (_, w) in enumerate(lst) if w == word), None)
    print("  %-8r %-6s rank=%s/%s" % (key, word, rank, len(lst)))

print("== 音节反解抽查 ==")
for code in ("wj", "jt", "xi", "xc"):
    print("  %-3s -> %s" % (code, decode_syllable(code)))


def stat_pool(code, prevs, pool_n=90):
    pool = {}

    def add(w):
        if w in pool:
            return
        pym = de.word_py.get(w) or ""
        m = max(1, len(pym.split()))
        pool[w] = rr.score_word(w, prevs) / m

    n = len(code)
    if n >= 2 and n % 2 == 0:
        py = " ".join(decode_syllable(code[i:i + 2]) for i in range(0, n, 2))
        for w in de.lookup_pinyin(py, 45):
            add(w)
    for w in de.lookup_initial(" ".join(code), pool_n):
        add(w)
    vpool = []
    if n % 2 == 0 and n >= 4:
        keys = [decode_syllable(code[i:i + 2]) for i in range(0, n, 2)]
        vpool += rr.viterbi(keys, "py", 5, ret_cost=True, prev=prevs[0] if prevs else "")
    vpool += rr.viterbi(list(code), "ini", 24, ret_cost=True, prev=prevs[0] if prevs else "")
    vpool += rr.anchor_sentences(list(code), "ini", 12, prev=prevs[0] if prevs else "")
    vpool.sort(key=lambda t: t[1] / max(1, t[2]))
    for s, c, m in vpool:
        cps = c / max(1, m)
        if s not in pool or cps < pool[s]:
            pool[s] = cps
    return sorted(pool.items(), key=lambda kv: kv[1])


nr = NeuralReranker(os.path.join(BASE, "ai_neural", "rbt3"))
ev = threading.Event()
_orig = nr._load


def _load2():
    _orig()
    if nr.ready:
        ev.set()


nr._load = _load2
nr.load_async()
if not ev.wait(60):
    sys.exit(1)

WANT_FISH = "我今天想吃西湖醋鱼"
WANT_TEST = "那我给你个东西测试一下"

for code, want in (("wjtxixhcy", WANT_FISH), ("wjtxcxhcy", WANT_FISH),
                   ("nwgngdxcuyx", WANT_TEST)):
    ranked = stat_pool(code, [])
    seqs = [w for w, _ in ranked]
    rank = seqs.index(want) + 1 if want in seqs else None
    print("-" * 64)
    print("码=%s(%d键) 目标句在池? %s（第 %s 名）" % (code, len(code), rank is not None, rank))
    if rank is not None and rank <= 90:
        ns = nr.score("。", [w for w, _ in ranked[:48]], max_cand=16)
        fused = sorted(((s - ns.get(w, -12.0), w) for w, s in ranked[:48]), key=lambda t: t[0])
        fr = [w for _, w in fused]
        print("  神经裁决后目标名次: %s / 48" % (fr.index(want) + 1 if want in fr else "48外"))
        print("  融合top5:", ["%s(%.1f)" % (w, v) for v, w in fused[:5]])
        print("  目标神经分: %.2f" % ns.get(want, -99))
