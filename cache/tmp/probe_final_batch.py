# -*- coding: utf-8 -*-
"""主人第三批样本（2026-09-06 晚）：效果终验。

tjz          接龙：上文尾词「一直」→ 统计中（3键3字词）
wbzdlszmsd   整句：我不知道老师怎么说的（10键，全高频词）
ahhjtll      整句：阿哈哈鸡汤来咯（7键网络梗，主人说不需要首选，看能不能匹配到）
"""
import os
import sys
import time

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from dict_engine import DictEngine
from rerank import StatReranker
from ai_llm_judge import QwenJudge

print("loading engine...", flush=True)
de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))
rr = StatReranker(de)
rr.load(os.path.join(BASE, "dicts"))

print("loading judge...", flush=True)
qj = QwenJudge(os.path.join(BASE, "ai_llm", "qwen25-05b-hf"))
qj.load_async()
t0 = time.perf_counter()
while not qj.ready and time.perf_counter() - t0 < 180:
    time.sleep(0.5)
if not qj.ready:
    sys.exit("judge 加载失败")


def whole(code, want, tag=""):
    t0 = time.perf_counter()
    pool = {}
    vpool = list(rr.viterbi(list(code), "ini", 8, ret_cost=True))
    vpool += rr.anchor_sentences(list(code), "ini", 20)
    vpool += rr.max_match_sentences(list(code), "ini")
    vpool += rr.char_chains(list(code), "ini", 6)
    for s, c, m in vpool:
        pool[s] = min(pool.get(s, 9e9), c / max(1, m))
    sents_all = [w for w, _ in sorted(pool.items(), key=lambda kv: kv[1])
                 if len(w) >= 5]
    ms_pool = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    scored = qj.score(sents_all)
    ms_llm = (time.perf_counter() - t1) * 1000
    ranked = sorted(((v, s) for s, v in scored.items()), reverse=True)
    rank = next((i + 1 for i, (_, s) in enumerate(ranked) if s == want), None)
    hit = "✓" if rank == 1 else ("~" if rank else "✗缺席")
    print("%s [%s] %s(%d键) 排名=%s 池%.0fms(%d句) 裁判%.0fms"
          % (hit, tag, code, len(code), rank, ms_pool, len(sents_all), ms_llm),
          flush=True)
    print("    top3:", ["%s(%.2f)" % (s, v) for v, s in ranked[:3]], flush=True)
    hitfrag = [s for s in sents_all if want[:2] in s or want[-2:] in s]
    if rank is None:
        print("    池内含目标片段的句:", hitfrag or "无", flush=True)
        print("    池内(全%d句):" % len(sents_all), sents_all, flush=True)
    return rank


def longjie(code, prevs, want, ctx="", tag=""):
    t0 = time.perf_counter()
    pool = {}
    for w in de.lookup_initial(" ".join(code), 90 if len(code) == 2 else 45):
        pym = de.word_py.get(w) or ""
        m = max(1, len(pym.split()))
        pool[w] = rr.score_word(w, prevs) / m
    prev = prevs[0] if prevs else ""
    for s, c, m in rr.viterbi(list(code), "ini", 8, ret_cost=True, prev=prev):
        pool[s] = min(pool.get(s, 9e9), c / max(1, m))
    ranked = sorted(pool.items(), key=lambda kv: kv[1])
    ms = (time.perf_counter() - t0) * 1000
    rank = next((i + 1 for i, (w, _) in enumerate(ranked) if w == want), None)
    hit = "✓" if rank == 1 else ("~" if rank else "✗缺席")
    print("%s [%s] %s+%s(%d键) 统计排名=%s %.0fms"
          % (hit, tag, ctx, code, len(code), rank, ms), flush=True)
    print("    top6:", [w for w, _ in ranked[:6]], flush=True)
    if rank != 1:
        names = [w for w, _ in ranked[:10]]
        if want in pool:
            names.append(want)
        sc = qj.score(names, ctx)
        rr2 = sorted(((v, s) for s, v in sc.items()), reverse=True)
        r2 = next((i + 1 for i, (_, s) in enumerate(rr2) if s == want), None)
        print("    裁判复核(带上文): 排名=%s top4: %s"
              % (r2, ["%s(%.2f)" % (s, v) for v, s in rr2[:4]]), flush=True)
    return rank


print("\n===== 第三批 =====", flush=True)
longjie("tjz", ["一直"], "统计中", ctx="文件夹大小一直在", tag="接龙")
whole("wbzdlszmsd", "我不知道老师怎么说的", tag="整句")
whole("ahhjtll", "阿哈哈鸡汤来咯", tag="网络梗")
