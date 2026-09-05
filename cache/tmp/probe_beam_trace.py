# -*- coding: utf-8 -*-
"""束搜索逐步追踪：打印每步约束后的 top 分布 + greedy 对照。"""
import os, sys
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList, LogitsProcessor
from dict_engine import DictEngine

MODEL = os.path.join(BASE, "ai_llm", "qwen25-05b-hf")
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True,
                                             torch_dtype=torch.float32)
model.eval()

de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))
SUB = os.path.join(BASE, "cache", "tmp", "subtlex", "SUBTLEX-CH-WF")
freq = {}
for line in open(SUB, "rb").read().decode("gb18030").splitlines()[3:]:
    p = line.split("\t")
    if len(p) >= 2 and p[0]:
        try:
            freq[p[0]] = int(p[1])
        except ValueError:
            pass

KEY2SM = {"v": "zh", "i": "ch", "u": "sh"}
SM2HZ = {}
for w, py in de.word_py.items():
    if len(w) == 1 and "\u4e00" <= w <= "\u9fff":
        try:
            w.encode("gb2312")
        except UnicodeEncodeError:
            continue
        parts = py.split()
        if not parts:
            continue
        sm = parts[0][:2] if parts[0][:2] in ("zh", "ch", "sh") else parts[0][0]
        SM2HZ.setdefault(sm, set()).add(w)
SM2HZ = {k: sorted(v, key=lambda c: -freq.get(c, 0))[:80] for k, v in SM2HZ.items()}

ID2HZ = {}
HZ2ID = {}
for sm, hzs in SM2HZ.items():
    for hz in hzs:
        ids = tok.encode(hz, add_special_tokens=False)
        if len(ids) == 1:
            HZ2ID.setdefault(sm, []).append(ids[0])
            ID2HZ[ids[0]] = hz


class PinyinMask(LogitsProcessor):
    def __init__(self, keys, prompt_len, trace=False):
        self.keys = keys
        self.prompt_len = prompt_len
        self.trace = trace
        self.step = 0

    def __call__(self, input_ids, scores):
        gi = input_ids.shape[1] - self.prompt_len
        if gi >= len(self.keys):
            mask = torch.full_like(scores, float("-inf"))
            mask[:, tok.eos_token_id] = 0.0
            return mask
        legal = HZ2ID.get(KEY2SM.get(self.keys[gi], self.keys[gi]), [])
        mask = torch.full_like(scores, float("-inf"))
        for tid in legal or []:
            mask[:, tid] = 0.0
        out = scores + mask
        if self.trace and gi < len(self.keys):
            row = out[0]
            top = row.topk(6)
            ctx_tail = tok.decode(input_ids[0, -4:])
            print("  step%d ctx=[...%s] key=%s top: %s" % (
                gi, ctx_tail, self.keys[gi],
                " ".join("%s:%.1f" % (ID2HZ.get(int(i), "?"), v)
                         for i, v in zip(top.indices, top.values))), flush=True)
        self.step += 1
        return out


KEYS = "wmyx"
# 文档起始符 <|endoftext|>=151643（tok.eos_token_id 是 instruct 的 <|im_end|>=151645，不能用）
DOC_START = 151643
enc = {"input_ids": torch.tensor([[DOC_START]]),
       "attention_mask": torch.ones(1, 1, dtype=torch.long)}
plen = enc["input_ids"].shape[1]

common = dict(max_new_tokens=len(KEYS) + 1, do_sample=False,
              pad_token_id=tok.eos_token_id,
              repetition_penalty=1.0, temperature=1.0, top_k=0, top_p=1.0)

print("== greedy (beam=1) ==", flush=True)
out = model.generate(**enc, num_beams=1,
                     logits_processor=LogitsProcessorList(
                         [PinyinMask(KEYS, plen, trace=True)]), **common)
print("greedy ->", repr(tok.decode(out[0][plen:], skip_special_tokens=True)), flush=True)

print("== beam=8 ==", flush=True)
out = model.generate(**enc, num_beams=8, num_return_sequences=3, early_stopping=True,
                     logits_processor=LogitsProcessorList(
                         [PinyinMask(KEYS, plen, trace=True)]), **common)
for o in out:
    print("beam ->", repr(tok.decode(o[plen:], skip_special_tokens=True)), flush=True)
