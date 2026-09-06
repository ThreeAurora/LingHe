# -*- coding: utf-8 -*-
"""约束 beam search 探测：transformers + Qwen2.5-0.5B，拼音位置约束。

每步生成时 mask 掉声母不匹配当前键位的汉字 token（自定义
LogitsProcessor），num_beams 保持整句一致性假设。
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
print("model loaded")

# 声母 -> 汉字集（底库单字 + SUBTLEX 字频排序前 60）
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
        sm = py.split()[0][:2] if py.split()[0][:2] in ("zh", "ch", "sh") \
            else py.split()[0][0]
        SM2HZ.setdefault(sm, set()).add(w)
SM2HZ = {k: sorted(v, key=lambda c: -freq.get(c, 0))[:60]
         for k, v in SM2HZ.items()}

# 汉字 -> token id（Qwen 单字 token）
HZ2ID = {}
for sm, hzs in SM2HZ.items():
    for hz in hzs:
        ids = tok.encode(hz, add_special_tokens=False)
        if len(ids) == 1:
            HZ2ID.setdefault(sm, []).append(ids[0])

from transformers import LogitsProcessorList, LogitsProcessor


class PinyinMask(LogitsProcessor):
    """第 gen_i 个生成 token 必须是声母=keys[gen_i] 的汉字。"""

    def __init__(self, keys, prompt_len):
        self.keys = keys
        self.prompt_len = prompt_len
        self.masked = set()

    def __call__(self, input_ids, scores):
        # batch 内每条已生成长度相同（beam 内一致）
        gi = input_ids.shape[1] - self.prompt_len
        if gi >= len(self.keys):
            # 键位耗尽：只允许 EOS
            eos = tok.eos_token_id
            mask = torch.full_like(scores, float("-inf"))
            mask[:, eos] = 0.0
            return mask
        legal = HZ2ID.get(KEY2SM.get(self.keys[gi], self.keys[gi]), [])
        mask = torch.full_like(scores, float("-inf"))
        if legal:
            for tid in legal:
                mask[:, tid] = 0.0
        return mask


def probe(keys, want, beam=8):
    prompt = ("用户用拼音首字母打字。键码 wmyx → 我们永远。"
              "键码 njhj → 你就回家。键码 tmdxl → 他们先来。"
              "键码 %s →" % keys)
    ids = tok(prompt, return_tensors="pt")
    plen = ids["input_ids"].shape[1]
    t0 = time.perf_counter()
    out = model.generate(
        **ids, max_new_tokens=len(keys) + 1, num_beams=beam, do_sample=False,
        logits_processor=LogitsProcessorList([PinyinMask(keys, plen)]),
        pad_token_id=tok.eos_token_id, early_stopping=True)
    ms = (time.perf_counter() - t0) * 1000
    gen = tok.decode(out[0][plen:], skip_special_tokens=True)
    ok = "✓" if gen == want else ("~" if want in gen else "✗")
    print("%s 键码=%s -> [%s]  %.0fms" % (ok, keys, gen, ms))
    return gen


for keys, want in [("wmyx", "我们永远"), ("njhj", "你就回家"),
                   ("nwgngdxcuyx", "那我给你个东西测试一下"),
                   ("wjtxixhcy", "我今天想吃西湖醋鱼")]:
    probe(keys, want)
