# -*- coding: utf-8 -*-
"""批量整句打分：全部候选一次（分批）前向，测耗时 vs 逐句。

另附诊断：查 吃了 在词库/口语2gram 里的状态（wilygbz 召回缺失根源）。
"""
import os
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.path.join(BASE, "ai_llm", "qwen25-05b-hf")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True,
                                             torch_dtype=torch.float32)
model.eval()
torch.set_grad_enabled(False)

PRIMER = ("<|endoftext|>我们明天去公园吧。你吃饭了吗？这个东西多少钱？"
          "我不知道他在哪里。一起去看电影吧。今天天气真好。 "
          "你昨天说的那件事我想了一下。周末我们出门走走。")
P_IDS = tok.encode(PRIMER, add_special_tokens=False)

CANDS = ["那我给你个东西测试一下", "那我给你过渡性测试一下",
         "那我给你个东西曾说一些", "那我给你个东西才是一些",
         "那我给你更多想从事业下", "我今天想吃西湖醋鱼",
         "我今天想出西湖醋鱼", "我今天宣传西湖醋鱼", "我今天宣传小孩参与"]


@torch.no_grad()
def score_batched(cands, bs=9):
    """右填充批量，逐行按注意力位置取 logP，字长归一。"""
    seqs = [P_IDS + tok.encode(s, add_special_tokens=False) for s in cands]
    out = {}
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]
        L = max(len(s) for s in chunk)
        pad = tok.pad_token_id or tok.eos_token_id
        ids = torch.full((len(chunk), L), pad, dtype=torch.long)
        att = torch.zeros((len(chunk), L), dtype=torch.long)
        for r, s in enumerate(chunk):
            ids[r, :len(s)] = torch.tensor(s)
            att[r, :len(s)] = 1
        logits = model(ids, attention_mask=att).logits
        lsm = torch.log_softmax(logits, -1)
        for r, s in enumerate(chunk):
            n_cand = len(s) - len(P_IDS)
            tot = 0.0
            for j in range(n_cand):
                pos = len(P_IDS) + j - 1  # 预测第 j 个候选字的位置
                tot += lsm[r, pos, s[len(P_IDS) + j]].item()
            out[cands[i + r]] = tot / max(1, n_cand)
    return out


# 预热一次（首次前向有初始化开销）
score_batched(CANDS[:1])
for bs in (1, 9):
    t0 = time.perf_counter()
    sc = score_batched(CANDS, bs=bs)
    ms = (time.perf_counter() - t0) * 1000
    print("bs=%d: %.0fms (%d句)" % (bs, ms, len(CANDS)))
    if bs == 9:
        for s, v in sorted(sc.items(), key=lambda kv: -kv[1]):
            print("   %.2f %s" % (v, s))

# ---- 诊断：吃了 的召回 ----
print()
from dict_engine import DictEngine
de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))
for w in ("吃了", "我吃了", "一个包子", "吃了一个包子"):
    py = de.word_py.get(w)
    wt = getattr(de, "word_w", {}).get(w) if hasattr(de, "word_w") else None
    print("词库[%s] py=%s weight=%s" % (w, py, wt))
g2 = os.path.join(BASE, "dicts", "spoken_2gram.txt")
if os.path.exists(g2):
    for line in open(g2, encoding="utf-8"):
        p = line.split()
        if p and p[0] in ("吃了", "我吃", "了个"):
            print("2gram:", line.strip())
