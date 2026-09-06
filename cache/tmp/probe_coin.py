# -*- coding: utf-8 -*-
"""自动造词纯函数 coin_pick 的场景验证。"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from dict_engine import DictEngine
from linghe import coin_pick

de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))

print("== 存在性抽查 ==")
for w in ("我想", "张布斯", "试作古华", "试作", "东西测试一下", "我吃", "测试一下", "古华"):
    print("  %-8s word_py=%r weight=%s" % (w, de.word_py.get(w), de.weight(w)))


def run(name, streak, seen=None):
    seen = {} if seen is None else seen
    r = coin_pick(streak, seen, de)
    print("  %-22s -> %s" % (name, r))
    return r


print("== 场景 ==")
run("张|布|斯(人名)", [("张", "zhang"), ("布", "bu"), ("斯", "si")])
run("试|作|古|华(全单字)", [("试", "shi"), ("作", "zuo"), ("古", "gu"), ("华", "hua")])
run("试作|古|华(混合)", [("试作", "shi zuo"), ("古", "gu"), ("华", "hua")])
run("我|想(高频组合)", [("我", "wo"), ("想", "xiang")])
run("我|吃|了(含虚词)", [("我", "wo"), ("吃", "chi"), ("了", "le")])
run("东西|测试|一下(防误伤)", [("东西", "dong xi"), ("测试", "ce shi"), ("一下", "yi xia")])
run("给你|个|东西(高频混合)", [("给你", "gei ni"), ("个", "ge"), ("东西", "dong xi")])
run("拼音缺失", [("张", ""), ("布", "bu")])
run("单片段", [("张", "zhang")])
run("英文串", [("hello", "")])

# 重复模式：同一混合串两次连续出现
seen = {}
s1 = run("给你|东西(第1次)", [("给你", "gei ni"), ("东西", "dong xi")], seen)
print("   seen=%r" % seen)
s2 = run("给你|东西(第2次)", [("给你", "gei ni"), ("东西", "dong xi")], seen)
print("   seen=%r" % seen)

# 链尾窗口：造词后短组合不受污染
run("已造词条覆盖", [("张布斯", "zhang bu si"), ("的", "de")])
