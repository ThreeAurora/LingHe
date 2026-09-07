# -*- coding: utf-8 -*-
"""探针：判别式 LLM 对「我吃了一个+X」的对句打分，验证 2 键简拼可否靠
纯裁判分把「包子」顶到固定词库后第一位（方案：词组纯 LLM 裁判裁决）。"""
import os, sys, time, threading
sys.path.insert(0, r'e:\CCSpace\projects\2026\09\linghe')
from ai_llm_judge import QwenJudge

BASE = r'e:\CCSpace\projects\2026\09\linghe'
ctx = "我吃了一个"
cands = [
    "不再", "不足", "不在", "不知", "不准", "帮助", "部长", "保证",
    "不做", "不止", "保障", "不住", "比重", "标准", "不只", "标志",
    "报纸", "本站", "班子", "包装", "标注", "不走", "本周", "本质",
    "不正", "不争", "版主", "并在", "步骤", "编制", "包子", "杯子",
    "豹子", "被子", "脖子",
]

qj = QwenJudge(os.path.join(BASE, "ai_llm", "qwen25-05b-hf"), log=lambda *a: None)
loaded = threading.Event()
orig = qj._load

def _l2():
    orig()
    if qj.ready:
        loaded.set()
qj._load = _l2
t0 = time.perf_counter()
qj.load_async()
loaded.wait(180)
print("加载 %.0fs, device=%s" % (time.perf_counter() - t0, getattr(qj, "device", "?")))

full = [ctx + w for w in cands]
t0 = time.perf_counter()
s = qj.score(full, ctx)
ms = (time.perf_counter() - t0) * 1000
ranked = sorted(((v, w) for w, v in s.items()), reverse=True)
print("---- 判别分（越大越自然，条件='" + ctx + "'），%.0fms ——" % ms)
for i, (v, w) in enumerate(ranked):
    print("%2d  %-6s %s" % (i + 1, w[len(ctx):], round(v, 3)))
print()
print("『包子』排名: %d / %d" % ([x[1] for x in ranked].index(ctx + "包子") + 1, len(ranked)))
print("『包子』分 - 『不再』分 = %.3f (需≥6才谈得上翻盘)" %
      (s.get(ctx + "包子", 0) - s.get(ctx + "不再", 0)))