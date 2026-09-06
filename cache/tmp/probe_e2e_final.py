# -*- coding: utf-8 -*-
"""端到端终验：主 beam + 锚定池合并 → 神经整句裁决，看含目标锚词句的排名。"""
import os
import sys
import threading

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from dict_engine import DictEngine
from rerank import StatReranker
from neural_rerank import NeuralReranker
from xiaohe import decode_syllable

de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))
rr = StatReranker(de)
rr.load(os.path.join(BASE, "dicts"))

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


def e2e(code, want_sub):
    """模拟 compute + _on_neural：主池 + 锚定 → 神经裁决。"""
    n = len(code)
    pool = {}
    if n % 2 == 0 and n >= 4:
        keys = [decode_syllable(code[i:i + 2]) for i in range(0, n, 2)]
        for s, c, m in rr.viterbi(keys, "py", 5, ret_cost=True):
            pool[s] = min(pool.get(s, 9e9), c / max(1, m))
    vpool = list(rr.viterbi(list(code), "ini", 24, ret_cost=True))
    vpool += rr.anchor_sentences(list(code), "ini", 12)
    for s, c, m in vpool:
        pool[s] = min(pool.get(s, 9e9), c / max(1, m))
    ranked = sorted(pool.items(), key=lambda kv: kv[1])
    top = [w for w, _ in ranked[:48]]
    ns = nr.score("。", top, max_cand=16)
    fused = sorted(((s - ns.get(w, -12.0), w) for w, s in ranked[:48]), key=lambda t: t[0])
    seq = [w for _, w in fused]
    hits = [(i + 1, w) for i, w in enumerate(seq) if want_sub in w]
    print("码=%s(%d键) 含[%s]的句子:" % (code, n, want_sub))
    for rk, w in hits[:4]:
        print("   第%2d名 %s" % (rk, w))
    if not hits:
        print("   48 名外")
    print("   融合top5:", seq[:5])


e2e("wjtxixhcy", "西湖醋鱼")
e2e("nwgngdxcuyx", "给你")
e2e("nwgngdxcuyx", "测试一下")
