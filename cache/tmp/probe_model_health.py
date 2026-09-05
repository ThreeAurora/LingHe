# -*- coding: utf-8 -*-
"""模型体检：束搜索选出「望米亿笑」而非「我们永远」，验证是权重坏/分词坏/还是约束实现坏。"""
import os, sys
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.path.join(BASE, "ai_llm", "qwen25-05b-hf")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True,
                                             torch_dtype=torch.float32)
model.eval()

# 1) 文档起始后裸续写 top10（无任何约束）
ids = torch.tensor([[tok.eos_token_id]])
with torch.no_grad():
    logits = model(ids).logits[0, -1]
top = logits.topk(10)
print("裸续写 top10:", [(tok.decode([i]), round(v.item(), 2)) for i, v in zip(top.indices, top.values)])

# 2) 关键 token 的裸 logit 对比（w/m/y/x 各取代表字）
for hz in "我望们米永远亿笑你宁就回家":
    tid = tok.encode(hz, add_special_tokens=False)
    if len(tid) == 1:
        print("logit[%s] = %.2f" % (hz, logits[tid[0]].item()))

# 3) 教师强制整句 logprob 对比（干净上下文）
def score(s):
    ids = tok(s, return_tensors="pt")
    with torch.no_grad():
        lg = model(ids["input_ids"]).logits
    lp = torch.log_softmax(lg[0, :-1], -1)
    tgt = ids["input_ids"][0, 1:]
    return lp[torch.arange(len(tgt)), tgt].sum().item()

for s in ["我们永远", "望米亿笑", "你就回家", "宁进核进"]:
    print("sumLogP(%s) = %.2f" % (s, score(s)))
