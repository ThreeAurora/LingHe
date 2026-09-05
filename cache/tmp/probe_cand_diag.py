# -*- coding: utf-8 -*-
"""候选集诊断：「我」到底在不在 HZ2ID['w'] 里？模型身份核对。"""
import os, sys, json
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
from dict_engine import DictEngine

MODEL = os.path.join(BASE, "ai_llm", "qwen25-05b-hf")

# 模型身份
cfg = json.load(open(os.path.join(MODEL, "config.json"), encoding="utf-8"))
print("arch:", cfg.get("architectures"), "| vocab:", cfg.get("vocab_size"), "| hidden:", cfg.get("hidden_size"))
gc = os.path.join(MODEL, "generation_config.json")
if os.path.exists(gc):
    print("generation_config:", json.load(open(gc, encoding="utf-8")))

# 候选集构建过程逐步打印
de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))
print("底库词条:", len(de.word_py))

# 看 word_py 里单字条目长啥样
samples = [(w, py) for w, py in list(de.word_py.items()) if len(w) == 1][:8]
print("单字样例:", samples)

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
            w.encode("gb2312")
        except UnicodeEncodeError:
            continue
        parts = py.split()
        if not parts:
            continue
        sm = parts[0][:2] if parts[0][:2] in ("zh", "ch", "sh") else parts[0][0]
        SM2HZ.setdefault(sm, set()).add(w)
print("声母组:", {k: len(v) for k, v in sorted(SM2HZ.items())})

for k in ("w", "m", "y", "x", "n", "j"):
    top = sorted(SM2HZ.get(k, set()), key=lambda c: -freq.get(c, 0))[:15]
    print("SM2HZ[%s] top15: %s" % (k, "".join(top)))

# 模型侧单 token 检查
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
for hz in "我你他":
    ids = tok.encode(hz, add_special_tokens=False)
    print("tok(%s) -> %s (%d tokens)" % (hz, ids, len(ids)))
