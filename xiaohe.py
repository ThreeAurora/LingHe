# -*- coding: utf-8 -*-
"""小鹤双拼解码。

键位经多源核实（小鹤官方口诀：秋闱鹅软月云书痴哦撇 / 啊松呆坟更航安京亮 / 走下嘈追宾鸟眠）。
声母：zh=v, ch=i, sh=u，其余声母与键名同名。
韵母：Q=iu W=ei E=e R=uan T=ue Y=un U=u I=i O=uo P=ie
      A=a S=ong D=ai F=en G=eng H=ang J=an K=ing L=iang
      Z=ou X=ia C=ao V=ui B=in N=iao M=ian
（同键变体：K 也含 uai，L 也含 uang，S 也含 iong，X 也含 ua，V 也含 ü/üe——解码取主韵母即可）

零声母：单韵母双写（aa/oo/ee）；二字复韵母直接打字面（ai/ao/an/en/ei/ou）；
        三字韵母=韵母首字母+韵母键（ang→ah, eng→eg）。
"""

# 键 -> 声母
SHENG = {
    "v": "zh", "i": "ch", "u": "sh",
}

# 键 -> 韵母
YUN = {
    "q": "iu",  "w": "ei", "e": "e",  "r": "uan", "t": "ue",  "y": "un",
    "u": "u",   "i": "i",  "o": "uo", "p": "ie",
    "a": "a",   "s": "ong", "d": "ai", "f": "en",  "g": "eng",
    "h": "ang", "j": "an",  "k": "ing", "l": "iang",
    "z": "ou",  "x": "ia",  "c": "ao",  "v": "ui",
    "b": "in",  "n": "iao", "m": "ian",
}

# 零声母二字组合（前键=韵母首字母，后键=韵母次字母字面）
ZERO2 = {
    "aa": "a", "oo": "o", "ee": "e", "er": "er",
    "ai": "ai", "ei": "ei", "ou": "ou", "ao": "ao",
    "an": "an", "en": "en",
}

# 同键变体：键 -> (触发声母集合, 变体韵母, 默认韵母)
# 规则依据：变体的选择由声母唯一决定（如 shuang 合法而 shiang 不存在）。
VARIANT = {
    "l": ({"sh", "zh", "ch", "h", "g", "k"}, "uang", "iang"),  # shuang/zhuang/guang… vs liang/xiang…
    "k": ({"k", "g", "h"}, "uai", "ing"),          # kuai/guai/huai   vs bing/xing…
    "s": ({"j", "x", "q", "y"}, "iong", "ong"),    # jiong/xiong/yong vs song/dong…
    "x": ({"j", "q", "x"}, "ia", "ua"),            # jia/qia/xia      vs gua/hua/shua
    "o": ({"b", "p", "m", "f", "y", "w"}, "o", "uo"),  # bo/mo/wo     vs duo/tuo/guo
    "v": ({"l", "n"}, "v", "ui"),                  # lv/nv = lü/nü    vs dui/gui…
}


def decode_syllable(two: str) -> str:
    """解码单个双拼音节（2 键）为拼音；解不动就原样返回。"""
    c1, c2 = two[0], two[1]
    if c1 in "aoe":
        if two in ZERO2:
            return ZERO2[two]
        y = YUN.get(c2)
        if y:
            return y  # ang→ah：结果就是韵母本身
        return two
    sm = SHENG.get(c1, c1)
    y = YUN.get(c2)
    if y is None:
        return two
    var = VARIANT.get(c2)
    if var and sm in var[0]:
        y = var[1]
    return sm + y


def interpret(code: str):
    """给出一串键入码的拼音解读列表（供 AI 提示）。

    4 码存在天然歧义（小鹤音形方案）：
      解读A：单字音形 -> 前 2 键是音，后 2 键是形码
      解读B：二字词   -> 每 2 键各是一个音节
    2/3 码按单字音(+形)处理；任意长度也可能是码表简码，由码表侧负责。
    返回 [(拼音描述, 说明), ...]
    """
    out = []
    n = len(code)
    if n == 0:
        return out
    if n < 2:
        out.append((code, "简码或声母"))
        return out
    yin1 = decode_syllable(code[:2])
    rest = code[2:]
    if rest:
        out.append((yin1, "单字，形码=" + rest))
    else:
        out.append((yin1, "单字或简码"))
    if n == 4:
        yin2 = decode_syllable(code[2:])
        out.append((yin1 + " " + yin2, "二字词"))
        return out
    if n > 4:
        # 第 5 码起两种形态并行给 AI：纯双拼音节流 / 整句声母简拼
        out = []
        if n % 2 == 0:
            syls = [decode_syllable(code[i:i + 2]) for i in range(0, n, 2)]
            out.append((" ".join(syls), "多音节词"))
        out.append((" ".join(code), "整句声母简拼（每字取双拼声母键，zh/ch/sh 记作 v/i/u）"))
        return out
    return out


if __name__ == "__main__":
    tests = {
        "ul": "shuang", "pb": "pin", "aih": "ai(形h)", "aiz": "ai(形z)",
        "aa": "a", "ah": "ang", "eg": "eng", "er": "er", "ao": "ao",
        "ul": "shuang",
    }
    for k in ["ul", "pb", "ai", "ah", "eg", "er", "ao", "aa", "xn", "xk"]:
        print(k, "->", decode_syllable(k))
    for c in ["aiz", "ulpb", "an", "xnld"]:
        print(c, "=>", interpret(c))
