# -*- coding: utf-8 -*-
"""把万象（rime-wanxiang）词库转成灵鹤底库格式。

万象词库格式（rime dict yaml）：
    ---
    name: jichu
    version: "..."
    sort: by_weight
    ...
    词语	nǐ hǎo	1234

灵鹤底库格式（dicts/*.dict.yaml）：
    词语	ni hao	1234

差别只在**拼音带声调**——输入法索引按无声调拼音建（小鹤双拼键位由音节
推导出），所以必须归一：
  - 去声调符号（nǐ → ni）
  - ü/ǖ/ǘ/ǚ/ǜ → v（拼音输入法惯例，小鹤里 ü 也走 v 键）

用法：
    python convert_wanxiang.py <输入.dict.yaml> <输出.dict.yaml> [--new-only]

--new-only：只输出灵鹤现有底库里没有的词（增量补充，避免词条数暴涨拖慢加载）。
"""
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))

# 声调归一表：带调元音 → 无声调；ü 系 → v
TONE = str.maketrans({
    "ā": "a", "á": "a", "ǎ": "a", "à": "a",
    "ō": "o", "ó": "o", "ǒ": "o", "ò": "o",
    "ē": "e", "é": "e", "ě": "e", "è": "e", "ê": "e",
    "ī": "i", "í": "i", "ǐ": "i", "ì": "i",
    "ū": "u", "ú": "u", "ǔ": "u", "ù": "u",
    "ǖ": "v", "ǘ": "v", "ǚ": "v", "ǜ": "v", "ü": "v",
    "ń": "n", "ň": "n", "ǹ": "n",
    " ": " ",
})


def strip_tone(py: str) -> str:
    """带调拼音串 → 无声调（空格分隔）。多音节按空格切。"""
    return " ".join(part.translate(TONE).strip() for part in py.split())


def iter_entries(path):
    """跳过 rime dict 的 YAML 头，逐行产出 (词, 带调拼音, 词频)。"""
    in_body = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not in_body:
                # YAML 头以 --- 开始，以 ... 结束
                if line.strip() == "...":
                    in_body = True
                continue
            if not line or line.startswith("#"):
                continue
            p = line.split("\t")
            if len(p) < 2:
                continue
            word = p[0].strip()
            py = p[1].strip()
            w = p[2].strip() if len(p) > 2 else "1"
            if not word or not py:
                continue
            try:
                w = int(float(w))
            except ValueError:
                w = 1
            yield word, py, w


def load_existing_words():
    """现有底库里已有的词（用于 --new-only 去增量）。"""
    words = set()
    for name in ("base.dict.yaml", "tencent.dict.yaml", "8105.dict.yaml"):
        path = os.path.join(BASE, "dicts", name)
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#") or line.startswith("---"):
                    continue
                p = line.split("\t")
                if p:
                    words.add(p[0].strip())
    return words


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 1
    src, dst = sys.argv[1], sys.argv[2]
    new_only = "--new-only" in sys.argv

    existing = load_existing_words() if new_only else set()
    valid_sylls = None  # 惰性：需要时才加载底库校验音节合法性

    out = []
    n_total = n_bad = n_dup = 0
    for word, py, w in iter_entries(src):
        n_total += 1
        # 音节数必须与字数一致（多音字/英文混排会不一致，直接丢弃）
        plain = strip_tone(py)
        syls = plain.split()
        if len(syls) != len(word):
            n_bad += 1
            continue
        if new_only and word in existing:
            n_dup += 1
            continue
        out.append((word, plain, w))

    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        f.write("# 由 convert_wanxiang.py 从万象词库转换（去声调，ü→v）\n")
        for word, plain, w in out:
            f.write("%s\t%s\t%d\n" % (word, plain, w))

    print("读入 %d 条，音节/字数不符丢弃 %d 条，已存在跳过 %d 条"
          % (n_total, n_bad, n_dup))
    print("输出 %d 条 → %s" % (len(out), dst))
    return 0


if __name__ == "__main__":
    sys.exit(main())
