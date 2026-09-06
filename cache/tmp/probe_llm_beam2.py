# -*- coding: utf-8 -*-
"""约束 beam search 探针 v2：诊断 prompt 格式对束搜索质量的影响。

核心假设：few-shot 指令模式污染 0.5B 的 LM 分布 → 改用干净续写上下文，
只靠拼音位置约束（每步只放行声母匹配的 GB2312 常用字）+ 固定句长。

对比三种 prompt：A=空(纯续写) B="句子：" C=few-shot(旧)。
输出每变体 top3 beam。
"""
import os
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from dict_engine import DictEngine

MODEL = os.path.join(BASE, "ai_llm", "qwen25-05b-hf")

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True,
                                             torch_dtype=torch.float32)
model.eval()
print("model loaded", flush=True)

# 声母 -> 汉字候选集：底库单字 ∩ SUBTLEX 字频排序，GB2312 过滤繁体，top 80
de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))
SUB = os.path.join(BASE, "cache", "tmp", "subtlex", "SUBTLEX-CH-WF")
freq = {}
raw = open(SUB, "rb").read().decode("gb18030")
for line in raw.splitlines()[3:]:
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
            w.encode("gb2312")  # 繁体/生僻字会被这里挡掉
        except UnicodeEncodeError:
            continue
        sm = py.split()[0][:2] if py.split()[0][:2] in ("zh", "ch", "sh") \
            else py.split()[0][0]
        SM2HZ.setdefault(sm, set()).add(w)
SM2HZ = {k: sorted(v, key=lambda c: -freq.get(c, 0))[:80]
         for k, v in SM2HZ.items()}

HZ2ID = {}
for sm, hzs in SM2HZ.items():
    for hz in hzs:
        ids = tok.encode(hz, add_special_tokens=False)
        if len(ids) == 1:
            HZ2ID.setdefault(sm, []).append(ids[0])

from transformers import LogitsProcessorList, LogitsProcessor


class PinyinMask(LogitsProcessor):
    def __init__(self, keys, prompt_len):
        self.keys = keys
        self.prompt_len = prompt_len

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
        return mask


FEWSHOT = ("用户用拼音首字母打字。键码 wmyx → 我们永远。"
           "键码 njhj → 你就回家。键码 tmdxl → 他们先来。键码 ")


def probe(keys, want, prompt, beam=8, tag=""):
    if prompt:
        ids = tok(prompt, return_tensors="pt")
    else:
        # 空上下文：用文档起始符 <|endoftext|> 占位，等价纯续写
        ids = {"input_ids": torch.tensor([[tok.eos_token_id]]),
               "attention_mask": torch.ones(1, 1, dtype=torch.long)}
    plen = ids["input_ids"].shape[1]
    t0 = time.perf_counter()
    out = model.generate(
        **ids, max_new_tokens=len(keys) + 1, num_beams=beam, do_sample=False,
        logits_processor=LogitsProcessorList([PinyinMask(keys, plen)]),
        pad_token_id=tok.eos_token_id, early_stopping=True,
        num_return_sequences=min(3, beam))
    ms = (time.perf_counter() - t0) * 1000
    tops = [tok.decode(o[plen:], skip_special_tokens=True) for o in out]
    ok = "✓" if want in tops[0] else ("~" if want in tops else "✗")
    print("%s [%s] %s -> %s  %.0fms" % (ok, tag, keys, " | ".join(tops), ms),
          flush=True)


CASES = [("wmyx", "我们永远"), ("njhj", "你就回家")]
VARIANTS = [
    ("A-empty", ""),
    ("B-colon", "句子："),
    ("C-fewshot", FEWSHOT),
]

for keys, want in CASES:
    for tag, prompt in VARIANTS:
        probe(keys, want, prompt, tag=tag)
