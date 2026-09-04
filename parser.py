# -*- coding: utf-8 -*-
"""键入码解析器：把键入流解析成 (音节序列, 辅码约束) 的合法解释。

主人钦定的三种合法形态（辅码只用首键，1 键）：
1. 纯音节：每 2 键一个双拼音节，长度无上限（4 码 yiyi、6 码 vhmusi=詹姆斯）。
2. 词尾单辅：纯音节 + 最后 1 键辅码（yiyit / yiyic），辅码附着词中任意一个字。
3. 全辅 6 码型：每个音节后紧跟该字辅码（yityic = 异(t)议(c)），共 3 键/字。

另附加"过渡态"解释（奇数长度时去掉最后一键的纯音节）：打辅码/续码过程中
候选不闪空。解释并存，谁命中候选谁上。

产出：[(syls, fuses), ...]
  syls  = ("yi","yi",...)     双拼 2 键码
  fuses = ((idx, key), ...)   idx=音节序号（0 起），-1=词中任意字
"""


def _syls_of(code: str):
    return tuple(code[i:i + 2] for i in range(0, len(code) - 1, 2))


def parse(code: str):
    n = len(code)
    if n == 0:
        return []
    out = []

    # 模式 1：纯音节（偶数长度）
    if n % 2 == 0:
        out.append((_syls_of(code), ()))

    # 模式 3：全辅交替（3 的倍数长度，每 3 键=音节2+辅码1，辅码附该音节的字）
    if n % 3 == 0 and n >= 3:
        syls, fuses = [], []
        for k in range(0, n, 3):
            seg = code[k:k + 3]
            syls.append(seg[:2])
            fuses.append((len(syls) - 1, seg[2]))
        out.append((tuple(syls), tuple(fuses)))

    # 模式 2：词尾单辅（奇数长度，最后 1 键是辅码，附任意字）
    if n % 2 == 1 and n >= 3:
        out.append((_syls_of(code[:-1]), ((-1, code[-1]),)))
        # 过渡态：最后一键当作"还没打完"，显示此前键程的候选
        out.append((_syls_of(code[:-1]), ()))
    elif n % 2 == 1 and n == 1:
        out.append(((), ((0, code[0]),)))  # 单键：无词义，供上层忽略

    return out


if __name__ == "__main__":
    for c in ["yiyi", "yiyit", "yiyic", "yityic", "yityi", "vhmusi", "ulpb", "vhmus"]:
        print("%-7s ->" % c)
        for syls, fuses in parse(c):
            print("    音节=%-12s 辅码=%s" % ("-".join(syls) or "∅", fuses or "无"))
