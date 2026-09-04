# -*- coding: utf-8 -*-
"""联想(next_word)功能探针：验证预训练共现表驱动上屏联想。"""
import sys, os, time
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
os.chdir(BASE)

from dict_engine import DictEngine
from rerank import StatReranker

de = DictEngine(log=print)
t0 = time.perf_counter()
de.load_dir("dicts")
print("[load] %.2fs, words=%d" % (time.perf_counter() - t0, len(de.word_py)))

rr = StatReranker(de, log=print)
t0 = time.perf_counter()
rr.load("dicts")
print("[rr.load] %.3fs" % (time.perf_counter() - t0))

# 1) 预训练表联想：常见双字/三字词的常见后继
cases = ["我们", "今天", "中国", "一个", "因为", "如果", "已经", "可以", "没有", "他们",
         "这个", "什么", "自己", "现在", "知道", "所以", "但是", "还是", "就是", "觉得"]
hits = 0
for w in cases:
    t0 = time.perf_counter()
    nxt = rr.next_word(w, 3)
    dt = (time.perf_counter() - t0) * 1000
    if nxt:
        hits += 1
    print("  %-4s -> %s  (%.2fms)" % (w, nxt, dt))
print("[pretrained liaison] %d/%d 有联想结果" % (hits, len(cases)))

# 2) 用户表联想（在线学习）：模拟用户连打
rr.learn("我", "想吃")
rr.learn("想吃", "火锅")
print("[user liaison] 想吃 ->", rr.next_word("想吃", 3))
print("[user override] 我们 ->", rr.next_word("我们", 3), "(用户学过的优先)")

# 3) 空结果边界
print("[edge] '' ->", rr.next_word("", 3), "| '不存在词' ->", rr.next_word("不存在词xyz", 3))

# 4) 查询延迟分布（热路径要求 <5ms）
t0 = time.perf_counter()
for _ in range(100):
    rr.next_word("我们", 1)
print("[latency] 100 次 next_word 平均 %.3fms" % ((time.perf_counter() - t0) * 10))
