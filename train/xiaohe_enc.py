# -*- coding: utf-8 -*-
"""拼音 → 小鹤双拼 2 键编码器（训练数据转写用，与 xiaohe.decode_syllable 对拍）。

规则源：xiaohe.py 官方键位注释。
  声母：zh=v, ch=i, sh=u，其余声母=自身首字母。
  韵母→键：YUN 表；变体按声母唯一确定（与 decode_syllable.VARIANT 同源）。
  零声母：单韵母双写（aa/oo/ee）；二字复韵母字面（ai/ao/an/en/ei/ou/er）；
          三字韵母=首字母+韵母键（ang→ah, eng→eg）。

ü 约定（本模块 norm 后的规范形）：
  lü→lv, nü→nv, lüe/nüe→lue/nue（decode_syllable 的输出习惯）。
自验证：对 char_pinyin.txt 收集的全部真实音节做 encode→decode 往返一致性检查。
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from xiaohe import YUN, VARIANT, ZERO2, decode_syllable

# 韵母 -> 键（直接映射部分）
YUN_KEY = {}
for k, v in YUN.items():
    YUN_KEY.setdefault(v, k)
YUN_KEY["ve"] = "t"      # üe（lve/nve 形）与 ue 同键 T
YUN_KEY["o"] = "o"       # o/uo 全同键 O（lo咯≡luo罗 双拼同键）

# 变体韵母 -> (键, 触发声母集合)   源自 xiaohe.VARIANT
VAR_KEY = {
    "uang": ("l", VARIANT["l"][0]),
    "uai":  ("k", VARIANT["k"][0]),
    "iong": ("s", VARIANT["s"][0]),
    "ia":   ("x", VARIANT["x"][0]),
    "ua":   ("x", set("bpmfdtnlgkhrzcsyw") | {"zh", "ch", "sh"}),  # ua 默认韵（j/q/x/l/d 是 ia）
    "v":    ("v", VARIANT["v"][0]),   # lv/nv（ü）；其余声母 ui
}

SM_LIST = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g",
           "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w"]

# 零声母韵母 -> 键组合
ZERO_YUN = {"a": "aa", "o": "oo", "e": "ee", "er": "er",
            "ai": "ai", "ei": "ei", "ao": "ao", "ou": "ou",
            "an": "an", "en": "en",
            "ang": "ah", "eng": "eg"}

# 全局音节缓存
_SYL_TABLE = {}


def norm(py: str) -> str:
    """拼音归一化到本模块规范形（ü 语义 → v/ue 习惯）。"""
    return py.replace("üe", "ue").replace("ü", "v")


def _encode_yun(sm, yun):
    """给（声母, 韵母）找韵母键；返回键或 None。"""
    if sm == "" and yun in ZERO_YUN:
        return ZERO_YUN[yun]
    if yun in VAR_KEY:
        key, trig = VAR_KEY[yun]
        if sm in trig:
            return key
        return None
    return YUN_KEY.get(yun)


def encode_syllable(py: str) -> str:
    """拼音(无调,可带ü) -> 双拼 2 键。失败返回 ''。"""
    py = norm(py.lower().strip())
    if not py:
        return ""
    if py in _SYL_TABLE:
        return _SYL_TABLE[py]
    sm, rest = "", py
    for cand in SM_LIST:
        if py.startswith(cand):
            sm = cand
            rest = py[len(cand):]
            break
    sm_key = {"zh": "v", "ch": "i", "sh": "u"}.get(sm, sm)
    yk = _encode_yun(sm, rest)
    two = (sm_key + yk) if yk else ""
    _SYL_TABLE[py] = two
    return two


def collect_real_syllables(path):
    """从 char_pinyin.txt 收集真实音节（去调）。返回 {音节: 使用次数}。"""
    TONE = str.maketrans("āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ", "aaaaeeeeiiiioooouuuuvvvv")
    from collections import Counter
    cnt = Counter()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            pys = line.split(":", 1)[1].split("#")[0]
            for py in pys.split(","):
                py = py.strip().translate(TONE)
                if py:
                    cnt[py] += 1
    return cnt


# decode 输出习惯等价（音同写异）：归一后再比较
_EQUIV = {"yiong": "yong", "lo": "luo"}


def _norm_equiv(py: str) -> str:
    return _EQUIV.get(norm(py), norm(py))


def verify(char_pinyin_path):
    """对全部真实音节做往返验证，返回 (总数, 失败列表)。"""
    cnt = collect_real_syllables(char_pinyin_path)
    bad = []
    ok = 0
    for py in cnt:
        two = encode_syllable(py)
        if not two or len(two) != 2:
            bad.append((py, two, "<不可编码>"))
            continue
        back = decode_syllable(two)
        if _norm_equiv(back) != _norm_equiv(py):
            bad.append((py, two, back))
        else:
            ok += 1
    return ok, bad


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ok, bad = verify(os.path.join(base, "dicts", "char_pinyin.txt"))
    print("真实音节往返验证: %d ✓ / %d ✗" % (ok, len(bad)))
    for py, two, back in bad[:15]:
        print("   %-8s -> %-3s -> %s" % (py, two, back))
    print("== 抽查 ==")
    for py in ["xiang", "ming", "shuang", "liang", "wang", "lv", "lü",
               "lüe", "yue", "bo", "duo", "xia", "hua", "yong", "kuai",
               "en", "eng", "a", "er", "jiu", "xiong", "ni", "hao", "sheng",
               "zhuang", "chuang", "chuai", "shuai", "zhuai", "lia", "dia",
               "nv", "ji", "shi", "yu", "wu", "yi"]:
        two = encode_syllable(py)
        back = decode_syllable(two) if len(two) == 2 else "??"
        flag = "✓" if _norm_equiv(back) == _norm_equiv(py) else "✗"
        print("  %-7s -> %-3s -> %-8s %s" % (py, two, back, flag))
