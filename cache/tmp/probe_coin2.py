# -*- coding: utf-8 -*-
"""自动造词时序模拟：逐字 push（不判定）+ 断链事件（判定），贴近真实打字流。"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE)

from dict_engine import DictEngine
from linghe import coin_pick

de = DictEngine()
de.load_dir(os.path.join(BASE, "dicts"))


class Sim:
    """模拟 Engine 的链行为：push 只积累，boundary() 才判定。"""

    def __init__(self):
        self.streak = []
        self.seen = {}
        self.coined = []

    def push(self, word, py):
        if not py:
            self.boundary()
            return
        if len(word) >= 2:
            self.boundary()
            self.streak = [(word, py)]
        else:
            self.streak.append((word, py))
            if len(self.streak) > 8:
                self.streak.pop(0)

    def boundary(self):
        if len(self.streak) < 2:
            self.streak = []
            return
        hit = coin_pick(self.streak, self.seen, de)
        if hit:
            s, p = hit
            self.coined.append(s)
            print("    [造词] %s (%s)" % (s, p))
        self.streak = []


def run(name, flow):
    print("  %s:" % name)
    sim = Sim()
    for item in flow:
        if item == "|":
            sim.boundary()
        else:
            sim.push(*item)
    print("    结果: %s" % (sim.coined or "无"))
    return sim.coined


print("== 真实时序场景 ==")
run("张布斯(逐字+句号)", [("张", "zhang"), ("布", "bu"), ("斯", "si"), "|"])
run("试作古华(全单字)", [("试", "shi"), ("作", "zuo"), ("古", "gu"), ("华", "hua"), "|"])
run("试作|古|华(词头+单字)", [("试作", "shi zuo"), ("古", "gu"), ("华", "hua"), "|"])
run("我想(在库)", [("我", "wo"), ("想", "xiang"), "|"])
run("我吃了(虚词)", [("我", "wo"), ("吃", "chi"), ("了", "le"), "|"])
run("东西测试一下(防误伤)", [("东西", "dong xi"), ("测试", "ce shi"), ("一下", "yi xia"), "|"])
run("我|吃|饭。继续打字", [("我", "wo"), ("吃", "chi"), ("饭", "fan"), "|", ("测试", "ce shi")])
run("英文断链", [("张", "zhang"), ("hello", ""), ("布", "bu"), "|"])
run("重复两次的混合串", [("给你", "gei ni"), ("东西", "dong xi"), "|",
                        ("给你", "gei ni"), ("东西", "dong xi"), "|"])
run("长链尾中间态(个一)", [("东西", "dong xi"), ("个", "ge"), ("一", "yi"), ("下", "xia"), "|"])
