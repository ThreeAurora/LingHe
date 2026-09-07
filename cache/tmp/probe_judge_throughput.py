# -*- coding: utf-8 -*-
"""探针：Qwen 判别打分稳态吞吐（第二轮起不预热），评估 2 键简拼全池送裁可行性。"""
import os, sys, time, threading
sys.path.insert(0, r'e:\CCSpace\projects\2026\09\linghe')
from ai_llm_judge import QwenJudge
from dict_engine import DictEngine

BASE = r'e:\CCSpace\projects\2026\09\linghe'
de = DictEngine(log=lambda *a: None)
de.load_dir(BASE + r'\dicts')
words = list(dict.fromkeys(de.lookup_initial("b z", 90)))

qj = QwenJudge(os.path.join(BASE, "ai_llm", "qwen25-05b-hf"), log=lambda *a: None)
loaded = threading.Event()
orig = qj._load

def _l2():
    orig()
    if qj.ready:
        loaded.set()
qj._load = _l2
qj.load_async()
loaded.wait(180)
print("模型加载完成 device=%s" % getattr(qj, "device", "?"))

ctx = "我吃了一个"
# 两轮：第一轮算预热，第二轮测稳态
for rnd in (1, 2):
    t0 = time.perf_counter()
    s = qj.score([ctx + w for w in words], ctx)
    ms = (time.perf_counter() - t0) * 1000
    idx = sorted(((v, w) for w, v in s.items()), reverse=True)
    bz = [w for _, w in idx].index(ctx + "包子") + 1
    print("round%d: %d词 %.0fms  包子排名=%d" % (rnd, len(s), ms, bz))
# 72 词档位
words72 = words[:72]
t0 = time.perf_counter()
s = qj.score([ctx + w for w in words72], ctx)
print("72词档: %.0fms  包子在列=%s" % ((time.perf_counter() - t0) * 1000, ctx + "包子" in s))