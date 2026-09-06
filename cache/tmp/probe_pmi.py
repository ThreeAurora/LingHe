# -*- coding: utf-8 -*-
"""终审排序 v2：伪似然 − γ×无条件字频偏差（PMI 式）。

垃圾句（文件他想查询缓存一）字字高频、局部平滑，字平均 logP 不低；
目标句（我今天想吃西湖醋鱼）含低频实义字（醋/鹤）被字平均稀释。
无条件字 logP 用 SUBTLEX-CH-CHR 字幕字频近似，γ 扫参验证方向。
"""
import os
import sys
import threading

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from dict_engine import DictEngine
from rerank import StatReranker
from neural_rerank import NeuralReranker
import math

de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))
rr = StatReranker(de)
rr.load(os.path.join(BASE, "dicts"))

nr = NeuralReranker(os.path.join(BASE, "ai_neural", "rbt3"))
ev = threading.Event()
_orig = nr._load


def _load2():
    _orig()
    if nr.ready:
        ev.set()


nr._load = _load2
nr.load_async()
if not ev.wait(60):
    sys.exit(1)

# 无条件字 logP（SUBTLEX-CH-CHR 字幕字频）
CHR = os.path.join(BASE, "cache", "tmp", "subtlex", "SUBTLEX-CH-CHR")
raw = open(CHR, "rb").read().decode("gb18030")
total = 0
cf = {}
for line in raw.splitlines()[2:]:
    p = line.split("\t")
    if len(p) >= 2 and p[0]:
        try:
            c = int(p[1])
        except ValueError:
            continue
        cf[p[0]] = c
        total += c
print("chars:", len(cf), "total:", total)
UNCOND = {ch: math.log(c / total) for ch, c in cf.items()}
DEFAULT_UNCOND = math.log(1.0 / total)  # 表外字（低频）取最小档


def uncond_of(w):
    vals = [UNCOND.get(ch, DEFAULT_UNCOND) for ch in w]
    return sum(vals) / len(vals)


def e2e(code, want, gamma):
    pool = {}
    vpool = list(rr.viterbi(list(code), "ini", 5, ret_cost=True))
    vpool += rr.anchor_sentences(list(code), "ini", 32)
    vpool += rr.max_match_sentences(list(code), "ini")
    for s, c, m in vpool:
        pool[s] = min(pool.get(s, 9e9), c / max(1, m))
    pool["我吃了一个包子"] = -5.0
    ranked = sorted(pool.items(), key=lambda kv: kv[1])
    words = [w for w, _ in ranked if len(w) < 5][:45]
    sents_all = [w for w, _ in ranked if len(w) >= 5]
    base = words[:]
    seen = set(base)
    base += [w for w in sents_all if w not in seen]
    ns = nr.score("。", base, max_cand=16)
    # PMI 式：条件 logP − γ×无条件字 logP
    scored = sorted(((ns[w] - gamma * uncond_of(w), w)
                     for w in base if len(w) >= 5), reverse=True)
    merged = [w for _, w in scored]
    rank = merged.index(want) + 1 if want in merged else None
    print("γ=%.2f 码=%s 排名=%s top4: %s" % (
        gamma, code, rank, [w for _, w in scored[:4]]))
    return rank


for gamma in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
    print("---")
    e2e("nwgngdxcuyx", "那我给你个东西测试一下", gamma)
    e2e("wjtxixhcy", "我今天想吃西湖醋鱼", gamma)
    e2e("wilygbz", "我吃了一个包子", gamma)
