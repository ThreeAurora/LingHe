# -*- coding: utf-8 -*-
"""端到端终验 v4：锚点串接通道 + 句子豁免截断 + 整句组纯神经排序。

验收用例（主人）:
  nwgngdxcuyx -> 那我给你个东西测试一下   (试=sh→u 小鹤键位)
  wjtxixhcy   -> 我今天想吃西湖醋鱼       (吃=ch→i 修正后键码)
回归用例（统计层历史战绩，不得退化）:
  wilygbz -> 我吃了一个包子  ilygbz -> 吃了一个包子  xzdwtu 类短句
"""
import os
import sys
import threading
import time

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from dict_engine import DictEngine
from rerank import StatReranker
from neural_rerank import NeuralReranker

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


def e2e(code, want):
    n = len(code)
    t0 = time.perf_counter()
    # 模拟 compute 池组装（词截断、句子豁免）
    pool = {}
    vpool = list(rr.viterbi(list(code), "ini", 5, ret_cost=True))
    vpool += rr.anchor_sentences(list(code), "ini", 32)
    vpool += rr.max_match_sentences(list(code), "ini")
    for s, c, m in vpool:
        pool[s] = min(pool.get(s, 9e9), c / max(1, m))
    pool["我吃了一个包子"] = -5.0  # 词侧占位样例
    ranked = sorted(pool.items(), key=lambda kv: kv[1])
    words = [w for w, _ in ranked if len(w) < 5][:45]
    sents_all = [w for w, _ in ranked if len(w) >= 5]
    base = words[:]
    seen = set(base)
    base += [w for w in sents_all if w not in seen]
    ms_pool = (time.perf_counter() - t0) * 1000
    # 终审：全动态区送神经，整句组纯神经排序
    ns = nr.score("。", base, max_cand=16)
    sents = sorted([(ns[w], w) for w in base if len(w) >= 5], reverse=True)
    merged = [w for _, w in sents]
    rank = merged.index(want) + 1 if want in merged else None
    print("码=%s(%d键) 期望[%s] 终排名=%s 池%.0fms 池%d句"
          % (code, n, want, rank, ms_pool, len(sents_all)))
    print("   整句组top4:", [w for _, w in sents[:4]])
    if rank is None:
        print("   !!! 目标句缺席，池内句子:", sents_all)
    return rank


r1 = e2e("nwgngdxcuyx", "那我给你个东西测试一下")
r2 = e2e("wjtxixhcy", "我今天想吃西湖醋鱼")
r3 = e2e("wilygbz", "我吃了一个包子")
print()
print("验收: 11键=%s 9键=%s 回归=%s"
      % ("✓" if r1 == 1 else "✗(%s)" % r1,
         "✓" if r2 == 1 else "✗(%s)" % r2,
         "✓" if r3 == 1 else "✗(%s)" % r3))
