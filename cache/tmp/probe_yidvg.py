# -*- coding: utf-8 -*-
"""yidvg 诊断：为什么打不出「一堆」。只加载码表/解析器/辅码，不加载底库。"""
import sys, os
BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)
os.chdir(BASE)

from mabiao import MaBiao
from fuma import FuMa
import parser as key_parser
from xiaohe import decode_syllable

mb = MaBiao()
mb.load_dir("mabiao")
fm = FuMa()
fm.load_dir("mabiao")

print("=== 辅码表 ===")
print("  word_match(堆,'g') =", fm.word_match("堆", "g"))
print("  word_match(一,'g') =", fm.word_match("一", "g"))
print("  word_match(堆,'a') =", fm.word_match("堆", "a"))

print("\n=== 音节解码 ===")
for s in ("yi", "dv", "yidv"):
    print("  %-5s -> %s" % (s, decode_syllable(s) if len(s) == 2 else "n/a(非双拼键)"))

print("\n=== parse('yidvg') 多假设 ===")
for syls, fuses in key_parser.parse("yidvg"):
    print("  syls=%s fuses=%s" % (syls, fuses))

print("\n=== mabiao 命中 ===")
print("  mb.exact('yidv') =", list(mb.exact("yidv"))[:8])
print("  mb.exact('yidvg') =", list(mb.exact("yidvg"))[:8])
print("  mb.prefix('yidv') 前8 =", list(mb.prefix("yidv", 8)))

print("\n=== 码表里「一堆」的编码 ===")
# 反查：码表里 value 含「一堆」的条目
if hasattr(mb, "index") or hasattr(mb, "exact_index"):
    for code in ("yidv", "yidg", "yid"):
        hits = [w for w in mb.exact(code)]
        print("  code=%-6s -> %s" % (code, hits[:10]))
