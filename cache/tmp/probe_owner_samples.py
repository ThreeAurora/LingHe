# -*- coding: utf-8 -*-
"""主人新样本批测（2026-09-06）：双拼简模式效果+速度。

A 整句组：主人原码 + 双拼简修正码（ch→i/sh→u）双口径各测一遍，
  池装配(锚点+字链+匹配) → LLM 裁判终审，报排名与耗时。
B 接龙组：上文尾词 prevs + 短码，词句同池按每音节平均代价比价
  （复刻 compute 统计层）；目标词若非 top1，LLM 裁判对 top10 复核。
C 双拼全码加餐：武媚娘传奇 wumwnlirqi（每字2键 py 模式）。
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
    """整句：池装配 + LLM 裁判终审。"""
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
    if rank is None:
        print("    池内(全%d句):" % len(sents_all), sents_all, flush=True)
    return rank


def longjie(code, prevs, want, ctx="", tag=""):
    """接龙短码：词句同池统计比价（compute 词路径复刻）。"""
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


print("\n===== A 整句组 =====", flush=True)
whole("wmncq", "武媚娘传奇", tag="原码")
whole("wmniq", "武媚娘传奇", tag="修正码(传=ch→i)")
whole("wztwsmsh", "我昨天晚上没睡好", tag="原码")
whole("wztwsmuh", "我昨天晚上没睡好", tag="修正码(睡=sh→u)")
whole("bydxwhm", "不要丢下我好吗", tag="原码")

print("\n===== B 接龙组 =====", flush=True)
longjie("bz", ["一个"], "包子", ctx="我今天吃了一个", tag="接龙")
longjie("lb", ["诗仙"], "李白", ctx="诗仙", tag="接龙")
longjie("df", ["诗圣"], "杜甫", ctx="诗圣", tag="接龙")
longjie("lyz", ["一只"], "老鹰", ctx="草原上奔跑了一只", tag="接龙")

print("\n===== C 双拼全码加餐 =====", flush=True)
whole("wumwnlirqi", "武媚娘传奇", tag="py全码")
