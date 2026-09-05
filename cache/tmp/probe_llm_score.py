# -*- coding: utf-8 -*-
"""LLM 裁判探针：候选句整句 logP 打分（教师强制，非生成）。

论点：0.5B 生成不行，但判别行——整句 logP 不带 RBT3 的新闻语料偏差。
做法：复用 e2e 池组装（锚点串接+最大匹配+viterbi），句子组送 Qwen
按 口语引子 上下文打 sumLogP（按字长归一），看目标句能否登顶。
"""
import os
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dict_engine import DictEngine
from rerank import StatReranker

MODEL = os.path.join(BASE, "ai_llm", "qwen25-05b-hf")

print("loading engine...", flush=True)
de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))
rr = StatReranker(de)
rr.load(os.path.join(BASE, "dicts"))

print("loading qwen...", flush=True)
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True,
                                             torch_dtype=torch.float32)
model.eval()

PRIMER = ("<|endoftext|>我们明天去公园吧。你吃饭了吗？这个东西多少钱？"
          "我不知道他在哪里。一起去看电影吧。今天天气真好。 "
          "你昨天说的那件事我想了一下。周末我们出门走走。")
PRIMER_IDS = tok.encode(PRIMER, add_special_tokens=False)


@torch.no_grad()
def llm_score(s):
    """P(候选句 | 口语引子)，含首字，按字数归一。"""
    cand = tok.encode(s, add_special_tokens=False)
    ids = torch.tensor([PRIMER_IDS + cand])
    logits = model(ids).logits[0]
    lp = torch.log_softmax(logits, -1)
    p0 = len(PRIMER_IDS)
    tot = 0.0
    for j, tid in enumerate(cand):
        tot += lp[p0 + j - 1, tid].item()
    return tot / max(1, len(cand))


def e2e(code, want):
    t0 = time.perf_counter()
    pool = {}
    vpool = list(rr.viterbi(list(code), "ini", 5, ret_cost=True))
    vpool += rr.anchor_sentences(list(code), "ini", 32)
    vpool += rr.max_match_sentences(list(code), "ini")
    for s, c, m in vpool:
        pool[s] = min(pool.get(s, 9e9), c / max(1, m))
    sents_all = [w for w, _ in sorted(pool.items(), key=lambda kv: kv[1])
                 if len(w) >= 5]
    ms_pool = (time.perf_counter() - t0) * 1000

    t1 = time.perf_counter()
    scored = sorted(((llm_score(s), s) for s in sents_all), reverse=True)
    ms_llm = (time.perf_counter() - t1) * 1000
    rank = next((i + 1 for i, (_, s) in enumerate(scored) if s == want), None)
    print("码=%s(%d键) 期望[%s] LLM终排名=%s  池%.0fms 打分%.0fms(%d句)"
          % (code, len(code), want, rank, ms_pool, ms_llm, len(sents_all)),
          flush=True)
    print("   top5:", ["%s(%.2f)" % (s, v) for v, s in scored[:5]], flush=True)
    if rank is None:
        print("   !!! 目标句缺席:", sents_all, flush=True)
    return rank


r1 = e2e("nwgngdxcuyx", "那我给你个东西测试一下")
r2 = e2e("wjtxixhcy", "我今天想吃西湖醋鱼")
r3 = e2e("wilygbz", "我吃了一个包子")
print()
print("验收: 11键=%s 9键=%s 回归=%s"
      % ("✓" if r1 == 1 else "✗(%s)" % r1,
         "✓" if r2 == 1 else "✗(%s)" % r2,
         "✓" if r3 == 1 else "✗(%s)" % r3))
