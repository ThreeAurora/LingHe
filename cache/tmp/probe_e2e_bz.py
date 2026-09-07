# -*- coding: utf-8 -*-
"""端到端验证：模拟 Engine 的 bz 链路（compute → judge.score → _on_neural 融合回流），
断言「我吃了一个」+ bz 场景动态区（码表固频后）首位是「包子」。"""
import sys, os, time, threading
sys.path.insert(0, r'e:\CCSpace\projects\2026\09\linghe')

from dict_engine import DictEngine
from rerank import StatReranker
from mabiao import MaBiao
from ai_llm_judge import QwenJudge
import caret_ctx

BASE = r'e:\CCSpace\projects\2026\09\linghe'
n_pool = 45

de = DictEngine(log=lambda *a: None)
de.load_dir(BASE + r'\dicts')
rr = StatReranker(de, log=lambda *a: None)
rr.load(BASE + r'\dicts')
mb = MaBiao()
mb.load_dir(BASE + r'\mabiao')

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
print("device =", getattr(qj, "device", "?"))

ctx_text = "我吃了一个"
prevs = caret_ctx.tail_words(ctx_text, de.word_py, 3)
prev = prevs[0] if prevs else ""


def compute(code):
    """精简复刻 Engine.compute：返回 (base, n_mb, dyn_all, pool_scores)。"""
    n = len(code)
    mb_exact = list(mb.exact(code))
    seen = set(mb_exact)
    mb_prefix = [w for w in mb.prefix(code, n_pool * 2) if w not in seen]
    seen |= set(mb_prefix)
    mb_part = (mb_exact + mb_prefix)[:24]
    n_mb = len(mb_part)
    pool = {}
    if de.loaded:
        dict_cands = []
        if n >= 2:
            dict_cands += de.lookup_initial(" ".join(code), 90)
        for w in dict_cands:
            if w in seen or w in pool:
                continue
            pym = de.word_py.get(w) or ""
            m = max(1, len(pym.split()))
            pool[w] = rr.score_word(w, prevs) / m
    if n >= 2:
        vpool = []
        vpool += rr.viterbi(list(code), "ini", 8, ret_cost=True, prev=prev)
        vpool.sort(key=lambda t: t[1] / max(1, t[2]))
        for s, c, m in vpool:
            if s in seen:
                continue
            cps = c / max(1, m)
            if s not in pool or cps < pool[s]:
                pool[s] = cps
    ranked_pool = sorted(pool.items(), key=lambda kv: kv[1])
    pool_scores = dict(ranked_pool)
    dyn_all = [w for w, _ in ranked_pool]
    sents_all = [w for w, _ in ranked_pool if len(w) >= 5]
    dict_side = [w for w, _ in ranked_pool if len(w) < 5][: n_pool - n_mb]
    base = mb_part + dict_side
    sb = set(base)
    base += [w for w in sents_all if w not in sb]
    return base, n_mb, dyn_all, pool_scores


def on_neural(base, n_mb, dyn_all, pool_scores, scores, judge=True,
              judge_ctx=ctx_text, neural_lambda=1.0):
    """复刻 Engine._on_neural（含回流 + λ 放大）。"""
    lam = 1.5 if (judge and judge_ctx and len(judge_ctx) >= 2) else neural_lambda
    tail = base[n_mb:]
    tail_set = set(tail)
    sents = [(scores[w], w) for w in tail
             if w in scores and w in pool_scores and len(w) >= 5]
    fused = {}
    for w in tail:
        if len(w) < 5 and w in scores and w in pool_scores:
            fused[w] = pool_scores[w] - lam * scores[w]
    for w, s in scores.items():
        if len(w) >= 5 or w in tail_set or w not in pool_scores:
            continue
        fused[w] = pool_scores[w] - lam * s
    words = sorted(fused.items(), key=lambda kv: kv[1])
    cand_set = {w for w, _ in words}
    rest = [w for w in tail if w not in cand_set]
    if judge:
        sents.sort(reverse=True)
        sents_ranked = [w for _, w in sents]
    else:
        sents_ranked = [w for w in tail if w in pool_scores and len(w) >= 5]
    cap = max(0, n_pool - n_mb - len(sents_ranked))
    merged = base[:n_mb] + sents_ranked + [w for w, _ in words[:cap]] + rest
    return merged, n_mb


code = "bz"
base, n_mb, dyn_all, pool_scores = compute(code)
print("\n[送裁前] n_mb=%d  包子在 base=%s  动态区统计首位=%s" % (
    n_mb, "包子" in base, (base[n_mb] if len(base) > n_mb else "-")))
print("  包子被截断于统计第 %d 位（> 截断线 %d）%s" % (
    dyn_all.index("包子") + 1, n_pool - n_mb,
    "—— 旧版就死于此处，LLM 根本评不到包子" if "包子" not in base else ""))

# LLM 全量评分（dyn_all 送裁）——注意 key 是「词」本身（Engine 的 request 送词表）
t0 = time.perf_counter()
scores = qj.score(dyn_all, ctx_text)
ms = (time.perf_counter() - t0) * 1000
print("\n[裁判] %d 词 %.0fms（含首轮预热）" % (len(scores), ms))
ranked_s = sorted(((v, w) for w, v in scores.items()), reverse=True)
print("  包子判别排名 = %d / %d" % (ranked_s.index((scores["包子"], "包子")) + 1, len(scores)))

merged, n_mb_out = on_neural(base, n_mb, dyn_all, pool_scores, scores)
print("\n[回流后] 动态区（码表固频后）前 12：")
for i, w in enumerate(merged[n_mb_out: n_mb_out + 12]):
    mark = "  <<< 包子" if w == "包子" else ""
    print("  %2d  %s%s" % (i + 1, w, mark))
dyn = merged[n_mb_out:]
print("\n结果：动态区首位 = %r %s" % (dyn[0], "✓ PASS" if dyn[0] == "包子" else "✗ FAIL"))

# 对照：无上文时不应被 LLM 乱序（回归检查 λ=1.0 场景）
print("\n[对照-无上文] bz 无上文 λ=1.0：")
base0, n_mb0, dyn0, ps0 = compute("bz")
s0 = qj.score(["bz 无上文" + w for w in dyn0], "")  # 空条件（模拟无上文）
_ = None
# 用空 ctx 重新评分：score(ctx=) 会给空 ctx，显示顺序可能不同，此处仅确认 λ=1.0 路径不崩
merged0, _ = on_neural(base0, n_mb0, dyn0, ps0,
                       {w: qj.score([w], "")[w] for w in dyn0}, judge=True,
                       judge_ctx="", neural_lambda=1.0)
print("  动态区首位 = %r（无上文 λ=1.0，融合仍以统计为主，不保证包子）" % merged0[n_mb0])