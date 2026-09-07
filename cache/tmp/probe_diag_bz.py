# -*- coding: utf-8 -*-
"""诊断：b/z 键字库序 + 字对频，定位 包子 进不了变体的原因"""
import os, sys
BASE = r"E:/CCSpace/projects/2026/09/linghe"
sys.path.insert(0, BASE)
os.chdir(BASE)
from dict_engine import DictEngine
from rerank import StatReranker

de = DictEngine(); de.load_dir(os.path.join(BASE, "dicts"))
rr = StatReranker(de); rr.load(os.path.join(BASE, "dicts"))
inv = rr._char_inv()
bc = inv.get("b", [])[:20]
zc = inv.get("z", [])[:20]
print("b 键 top20 字:", "".join(bc), flush=True)
print("z 键 top20 字:", "".join(zc), flush=True)
print("包 在 b 键位次:", bc.index("包") if "包" in bc else "不在前20", flush=True)
print("子 在 z 键位次:", zc.index("子") if "子" in zc else "不在前20", flush=True)
pairs = sorted(((rr.sp2.get(a + b2, 0), a, b2) for a in bc[:16] for b2 in zc[:16]), reverse=True)[:10]
print("字对频 top10:", [(a + b2, f) for f, a, b2 in pairs], flush=True)
print("sp2[(包,子)] =", rr.sp2.get("包子", 0), flush=True)
print("lookup_initial('b z', 90) 里 包子 位次:",
      next((i + 1 for i, w in enumerate(de.lookup_initial("b z", 90)) if w == "包子"), None), flush=True)
