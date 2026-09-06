# -*- coding: utf-8 -*-
"""anchor_sentences 内部过程打点：每个核心的 前缀/串接/产出 逐条打印。"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from dict_engine import DictEngine
from rerank import StatReranker, ANCHOR2_FREQ

de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))
rr = StatReranker(de)
rr.load(os.path.join(BASE, "dicts"))

for code, want in [("nwgngdxcuyx", "那我给你个东西测试一下"),
                   ("wjtxixhcy", "我今天想吃西湖醋鱼")]:
    keys = tuple(code)
    m = len(keys)
    cores = []
    for i in range(m):
        for L in (2, 3, 4):
            if i + L > m:
                continue
            lst = de.by_initial.get(" ".join(keys[i:i + L]))
            if not lst:
                continue
            if L >= 4:
                cores.append((i, i + L, lst[0][1]))
            elif L == 3:
                if len(lst) <= 8:
                    cores.append((i, i + L, lst[0][1]))
            else:
                uniq, seen_w = [], set()
                for _, w in lst:
                    if w in seen_w:
                        continue
                    seen_w.add(w)
                    uniq.append(w)
                    if len(uniq) >= 4:
                        break
                for w in uniq:
                    if de.weight(w) >= ANCHOR2_FREQ:
                        cores.append((i, i + L, w))
    cores.sort(key=lambda a: (-(a[1] - a[0]), -de.weight(a[2]), a[0]))
    print("=" * 70)
    print("码=%s 期望[%s] 核心%d个" % (code, want, len(cores)))
    out, seen_sent, seen_core = [], set(), set()
    for (ai, aj, aw) in cores:
        if aw in seen_core:
            continue
        seen_core.add(aw)
        pres = rr.viterbi(list(keys[:ai]), "ini", 1, ret_cost=True, prev="") \
            if ai else [("", 0.0, 0)]
        if not pres:
            continue
        parts = ([pres[0][0]] if pres[0][0] else []) + [aw]
        pos = aj
        chain = [aw]
        while pos < m:
            nxt = next((a for a in cores if a[0] >= pos), None)
            if nxt:
                gap = rr.viterbi(list(keys[pos:nxt[0]]), "ini", 1) \
                    if nxt[0] > pos else [""]
                g = gap[0] if gap else ""
                parts.append(g + nxt[2])
                chain.append("%s+%s" % (g, nxt[2]) if g else nxt[2])
                pos = nxt[1]
            else:
                gap = rr.viterbi(list(keys[pos:]), "ini", 1)
                if gap:
                    parts.append(gap[0])
                    chain.append(gap[0])
                pos = m
        sent = "".join(parts)
        dup = " (dup)" if sent in seen_sent else ""
        if sent not in seen_sent:
            seen_sent.add(sent)
            stop = " ★STOP" if len(out) >= 24 else ""
            out.append(sent)
            mark = "  ◀◀◀目标" if sent == want else ""
            print("  核心[%d:%d)%s -> %s%s%s%s" % (
                ai, aj, aw, sent, dup, mark, stop))
        if len(out) >= 24:
            print("  …… n=24 截断，后续核心不再轮询")
            break
