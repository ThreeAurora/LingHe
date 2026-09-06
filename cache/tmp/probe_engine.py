# -*- coding: utf-8 -*-
"""综合实测：主人全部用例走真实 Engine 链路（含同池比价/短码整句/上下文）。"""
import json
import sys, os, time
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
os.chdir(BASE)

from linghe import Engine

with open("config.json", "r", encoding="utf-8") as f:
    cfg = json.load(f)
cfg["ai"] = {"enabled": False}  # 测静态侧，不探测端点

eng = Engine(".", cfg, log=lambda *a: None)
t0 = time.perf_counter()
while not eng.de.loaded and time.perf_counter() - t0 < 180:
    time.sleep(0.4)
print("[底库] %d 词条 加载耗时 %.1fs\n" % (eng.de.size, time.perf_counter() - t0))

# (码串, 期望, 说明)。期望为 None 只看输出。
CASES = [
    ("ibz",        "吃包子",        "3键简拼→3字词（同池/短码整句新增能力）"),
    ("iibczi",     "吃包子",        "6键全拼→3字词（回归）"),
    ("vingti",     "智能体",        "6键全拼→新词（万象词库）"),
    ("xzdwtu",     "现在的问题是",   "6键纯声母简拼"),
    ("ilygbz",     "吃了一个包子",   "7键简拼（回归）"),
    ("hubgvn",     "还是不够智能",   "6键简拼（回归）"),
    ("hutil",      "还是太差了",     "5键简拼（回归）"),
    ("yidvg",      "一堆",          "5键音+辅码（底库辅码路径修复回归）"),
    ("dnynkcltydqd", "但你也能看出来它有多强大", "12键圣杯"),
    ("zccizcju",   None,           "主人例：灵活组合"),
    ("vjmusi",     "詹姆斯",        "6键全拼（回归）"),
]
for code, want, note in CASES:
    t = time.perf_counter()
    cands, n_mb = eng.compute(code)
    ms = (time.perf_counter() - t) * 1000
    top = cands[:8]
    mark = ""
    if want:
        mark = "OK " if want in cands[:3] else ("IN " if want in cands else "MISS")
    print("%-4s %-14s %6.1fms mb=%d %s\n      %s" % (mark, code, ms, n_mb, note, top))

# 上下文条件预测验证：同一键串在不同上文下的表现
print("\n=== 上下文条件预测（ctx_prev 效果）===")
for ctx in ["", "豆包", "输入法"]:
    eng.ctx_prev = ctx
    cands, _ = eng.compute("sjhm")
    print("  上文=%-6s sjhm -> %s" % (ctx or "(无)", cands[:5]))
