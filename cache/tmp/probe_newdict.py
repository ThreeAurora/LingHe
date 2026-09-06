# -*- coding: utf-8 -*-
"""验证万象增量词库集成效果：现代新词命中 + 整句切分。"""
import sys, os, time
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
os.chdir(BASE)

from dict_engine import DictEngine
import rerank

de = DictEngine(log=lambda *a: None)
t0 = time.perf_counter()
de.load_dir("dicts")
print("底库 %d 词条，加载 %.1fs\n" % (de.size, time.perf_counter() - t0))

print("=== 新词全拼命中（vingti 等要走的索引）===")
for py, want in [("zhi neng ti", "智能体"), ("jue ming du shi", "绝命毒师"),
                 ("da mo xing", "大模型"), ("ren gong zhi neng", "人工智能"),
                 ("shen jing wang luo", "神经网络")]:
    hits = [w for _, w in de.by_pinyin.get(py, [])[:6]]
    print("  %-18s -> %-30s %s" % (py, hits, "OK" if want in hits else "MISS"))

print("\n=== 简拼命中 ===")
for ini, want in [("v n t", "智能体"), ("j m d s", "绝命毒师")]:
    hits = [w for _, w in de.by_initial.get(ini, [])[:6]]
    print("  %-10s -> %-30s %s" % (ini, hits, "OK" if want in hits else "MISS"))

print("\n=== Viterbi 整句（全拼）===")
rr = rerank.StatReranker(de, log=lambda *a: None)
rr.load("dicts")
for keys, want in [("zhi neng ti".split(), "智能体"),
                   ("jue ming du shi".split(), "绝命毒师"),
                   ("wo chi le yi ge bao zi".split(), "我吃了一个包子")]:
    got = rr.viterbi(keys, "py", 3)
    print("  %-22s -> %-28s %s" % ("".join(keys), got, "OK" if want in got else "MISS"))

print("\n=== 主人三个简拼用例（回归）===")
for code, want in [("ilygbz", "吃了一个包子"), ("hubgvn", "还是不够智能"), ("hutil", "还是太差了")]:
    got = rr.viterbi(list(code), "ini", 3)
    print("  %-8s -> %-30s %s" % (code, got, "OK" if want in got else "MISS"))
