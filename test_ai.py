# -*- coding: utf-8 -*-
"""端到端验证：本地模型真实预测质量与耗时。
用法：先确保 ollama serve 在跑且模型已拉取，然后 python test_ai.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ai_engine import AIEngine

cfg = {"mode": "auto", "local_urls": ["http://localhost:11434"],
       "model": "qwen2.5:1.5b", "timeout": 4.0}
done = []
ai = AIEngine(cfg, log=print)
ai.on_result = lambda seq: done.append(seq)
ai.probe()
time.sleep(1.2)
print("端点:", ai.endpoint_desc, " ready:", ai.ready)
if not ai.ready:
    sys.exit("AI 未就绪")

# (上下文, 键入码, 说明)
cases = [
    ("今天天气真", "hen", "二字词 he? 实为 hen(hf)"),
    ("", "aiz", "爱 单字音形"),
    ("", "ulpb", "shuang pin 二字词"),
    ("我们要尽快完成这个", "renwu", "接词：任务"),
    ("", "xn", "小 简码"),
]
for ctx, code, note in cases:
    t0 = time.perf_counter()
    ai.request(code, ctx)
    deadline = time.time() + 8
    while time.time() < deadline and not done:
        time.sleep(0.05)
    dt = (time.perf_counter() - t0) * 1000
    got = ai.peek(ctx[-32:], code)
    print("%-4s (%s) %6.0fms -> %s" % (code, note, dt, got))
    done.clear()
