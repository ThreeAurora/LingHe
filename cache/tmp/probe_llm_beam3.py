# -*- coding: utf-8 -*-
"""探针 v3：0.5B 最后两张牌——chat few-shot(D) vs 口语引子(E)。

D: 标准 im_start 模板，system+10 组键码→句子示例，beam 约束生成。
E: <|endoftext|>+几行自然口语句子，然后约束续写新句（纯续写分布校准）。
共同修正：杀掉 generation_config 的采样残留(rep_penalty/top_k)；
末步只放行句末符/结束符；束宽 8。
"""
import os, sys, time
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
print("model loaded", flush=True)

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

HZ2ID = {}
for sm, hzs in SM2HZ.items():
    for hz in hzs:
        ids = tok.encode(hz, add_special_tokens=False)
        if len(ids) == 1:
            HZ2ID.setdefault(sm, []).append(ids[0])
END_IDS = [i for i in (tok.encode(c, add_special_tokens=False)[0]
                       for c in "。！？") if isinstance(i, int)]


class PinyinMask(LogitsProcessor):
    """gi<len(keys): 只放行声母匹配字；gi==len(keys): 只放行句末/结束符。"""

    def __init__(self, keys, prompt_len, end_ids):
        self.keys = keys
        self.prompt_len = prompt_len
        self.end_ids = end_ids

    def __call__(self, input_ids, scores):
        gi = input_ids.shape[1] - self.prompt_len
        mask = torch.full_like(scores, float("-inf"))
        if gi >= len(self.keys):
            for tid in self.end_ids:
                mask[:, tid] = 0.0
            return mask
        for tid in HZ2ID.get(KEY2SM.get(self.keys[gi], self.keys[gi]), []):
            mask[:, tid] = 0.0
        return scores + mask


def run(keys, prompt_ids, beam=8, nbest=3):
    plen = prompt_ids.shape[1]
    t0 = time.perf_counter()
    out = model.generate(
        input_ids=prompt_ids, attention_mask=torch.ones_like(prompt_ids),
        max_new_tokens=len(keys) + 1, num_beams=beam, do_sample=False,
        early_stopping=True, num_return_sequences=nbest,
        logits_processor=LogitsProcessorList(
            [PinyinMask(keys, plen, END_IDS)]),
        pad_token_id=tok.eos_token_id,
        repetition_penalty=1.0, temperature=1.0, top_k=0, top_p=1.0)
    ms = (time.perf_counter() - t0) * 1000
    return [tok.decode(o[plen:], skip_special_tokens=True) for o in out], ms


# ---------- D: chat few-shot ----------
PAIRS = [("wmyg", "我们有个"), ("nmtl", "你明天来"), ("tmxl", "他们先来"),
         ("dsbj", "都是北京"), ("myrw", "没有人问"), ("xhcy", "西湖醋鱼"),
         ("wjtc", "我今天吃"), ("yqwl", "一起玩了"), ("hjsj", "很久时间"),
         ("bxjd", "别想戒指")]
msgs = [{"role": "system",
         "content": "把拼音首字母键码还原成一句通顺的中文口语。"}]
for k, s in PAIRS:
    msgs += [{"role": "user", "content": "键码 %s" % k},
             {"role": "assistant", "content": s}]


def chat_ids(keys):
    seq = msgs + [{"role": "user", "content": "键码 %s" % keys},
                  {"role": "assistant", "content": ""}]
    text = tok.apply_chat_template(seq, tokenize=False, continue_final_message=True)
    return tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]


# ---------- E: 口语引子 ----------
PRIMER = ("<|endoftext|>我们明天去公园吧。你吃饭了吗？这个东西多少钱？"
          "我不知道他在哪里。一起去看电影吧。今天天气真好。 "
          "你昨天说的那件事我想了一下。周末我们出门走走。\n")


def primer_ids(keys):
    ids = tok.encode(PRIMER, add_special_tokens=False)
    return torch.tensor([ids])


CASES = [("wmyx", "我们永远"), ("njhj", "你就回家"),
         ("nwgngdxcuyx", "那我给你个东西测试一下"),
         ("wjtxixhcy", "我今天想吃西湖醋鱼")]

for keys, want in CASES:
    for tag, builder in [("D-chat", chat_ids), ("E-primer", primer_ids)]:
        tops, ms = run(keys, builder(keys))
        ok = "✓" if want in tops[0] else ("~" if want in tops else "✗")
        print("%s [%s] %s -> %s  %.0fms" % (ok, tag, keys, " | ".join(tops), ms),
              flush=True)
