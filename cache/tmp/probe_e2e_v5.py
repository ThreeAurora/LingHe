# -*- coding: utf-8 -*-
"""端到端终验 v5：LLM 裁判（Qwen 整句判别）+ 口语字链通道。

验收（主人）:
  nwgngdxcuyx -> 那我给你个东西测试一下
  wjtxixhcy   -> 我今天想吃西湖醋鱼
回归（统计层历史战绩，wilygbz 曾被三通道集体漏召，字链通道修复案）:
  wilygbz -> 我吃了一个包子
"""
import os
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from dict_engine import DictEngine
from rerank import StatReranker
from ai_llm_judge import QwenJudge

print("loading engine...", flush=True)
de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))
rr = StatReranker(de)
rr.load(os.path.join(BASE, "dicts"))

print("loading judge...", flush=True)
qj = QwenJudge(os.path.join(BASE, "ai_llm", "qwen25-05b-hf"))
qj.load_async()
t0 = time.perf_counter()
while not qj.ready and time.perf_counter() - t0 < 180:
    time.sleep(0.5)
if not qj.ready:
    sys.exit("judge 加载失败")


def e2e(code, want):
    t0 = time.perf_counter()
    pool = {}
    vpool = list(rr.viterbi(list(code), "ini", 8, ret_cost=True))
    vpool += rr.anchor_sentences(list(code), "ini", 12)
    vpool += rr.max_match_sentences(list(code), "ini")
    vpool += rr.char_chains(list(code), "ini", 6)
    for s, c, m in vpool:
        pool[s] = min(pool.get(s, 9e9), c / max(1, m))
    sents_all = [w for w, _ in sorted(pool.items(), key=lambda kv: kv[1])
                 if len(w) >= 5]
    ms_pool = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    scored = qj.score(sents_all)
    ms_llm = (time.perf_counter() - t1) * 1000
    ranked = sorted(((v, s) for s, v in scored.items()), reverse=True)
    rank = next((i + 1 for i, (_, s) in enumerate(ranked) if s == want), None)
    print("码=%s(%d键) 期望[%s] 终排名=%s  池%.0fms(%d句) 裁判%.0fms"
          % (code, len(code), want, rank, ms_pool, len(sents_all), ms_llm),
          flush=True)
    print("   top4:", ["%s(%.2f)" % (s, v) for v, s in ranked[:4]], flush=True)
    if rank is None:
        print("   !!! 目标句缺席:", sents_all, flush=True)
    return rank


r1 = e2e("nwgngdxcuyx", "那我给你个东西测试一下")
r2 = e2e("wjtxixhcy", "我今天想吃西湖醋鱼")
r3 = e2e("wilygbz", "我吃了一个包子")
print()
print("验收: 11键=%s 9键=%s 回归7键=%s"
      % ("✓" if r1 == 1 else "✗(%s)" % r1,
         "✓" if r2 == 1 else "✗(%s)" % r2,
         "✓" if r3 == 1 else "✗(%s)" % r3))
