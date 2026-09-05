# -*- coding: utf-8 -*-
"""热词管线：拉取公开开放词库 → dicts/hotwords_inc.dict.yaml（2 列格式）。

定位（主人 2026-09-05）：豆包的云端热词表盖住「全网的新词」（试作古华/
网络梗/新品牌），我们端侧的对应物是本脚本——定期跑一次，公共新词自动
进入底库加载链；用户个人世界的新词走连续片段自动造词（linghe.py coin_pick）。

词源：
  1. THUOCL 清华开放中文词库（GitHub thunlp/THUOCL，多领域、纯 txt 两列）
  2. 搜狗细胞词库 .scel（parse_scel 已实现，精选分类后手工放入 cache/tmp/scel/）

产出格式：词条<TAB>权重（2 列）——dict_engine._load_file 原生支持，加载时
自动逐字注音入全拼/简拼双索引，无需本脚本注音。

用法：
    python fetch_hotwords.py            # 拉取 THUOCL + 本地 scel，合并转制
"""
import glob
import os
import struct
import sys
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "dicts", "hotwords_inc.dict.yaml")

# THUOCL 类别（jsdelivr CDN 列出的真实文件名；404 的自动跳过）
THUOCL_FILES = [
    "THUOCL_IT.txt", "THUOCL_medical.txt", "THUOCL_food.txt",
    "THUOCL_law.txt", "THUOCL_animal.txt", "THUOCL_plant.txt",
    "THUOCL_car.txt", "THUOCL_caijing.txt", "THUOCL_diming.txt",
    "THUOCL_lishimingren.txt", "THUOCL_chengyu.txt", "THUOCL_poem.txt",
]
# 顺序即优先级：jsdelivr（GitHub 内容 CDN，国内可达实测 200）> gh 镜像 > raw 直连
MIRRORS = [
    "https://cdn.jsdelivr.net/gh/thunlp/THUOCL@master/data/",
    "https://mirror.ghproxy.com/https://raw.githubusercontent.com/thunlp/THUOCL/master/data/",
    "https://raw.githubusercontent.com/thunlp/THUOCL/master/data/",
]

MIN_FREQ = 200   # 词频阈值：THUOCL 频次是语料计数，200 以下噪声占比高
MIN_LEN, MAX_LEN = 2, 6  # 词条长度窗口


def is_good_word(w):
    if not (MIN_LEN <= len(w) <= MAX_LEN):
        return False
    return all("\u4e00" <= ch <= "\u9fff" for ch in w)


def fetch_thuocls():
    """THUOCL：词\t频次 两列文本，直转。返回 {词: 频次}。"""
    got = {}
    for name in THUOCL_FILES:
        text = None
        for mir in MIRRORS:
            url = mir + name
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "linghe/1.0"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    text = r.read().decode("utf-8", "ignore")
                break
            except Exception:
                continue
        if not text:
            print("  [跳过] %s（全部镜像不可达）" % name)
            continue
        n = 0
        for line in text.splitlines():
            parts = line.rstrip().split("\t")
            if len(parts) < 2:
                continue
            w, f = parts[0].strip(), parts[1].strip()
            if not is_good_word(w) or not f.replace(".", "").isdigit():
                continue
            f = int(float(f))
            if f < MIN_FREQ:
                continue
            if f > got.get(w, 0):
                got[w] = f
            n += 1
        print("  [OK] %s +%d 词" % (name, n))
    return got


def parse_scel(path):
    """搜狗细胞词库 .scel 解析（拼音表+词表二进制格式，社区逆向版）。

    返回 [(词, 拼音串), ...]。拼音缺失时调用方回退底库自动注音。
    """
    out = []
    with open(path, "rb") as f:
        data = f.read()
    # 拼音表：0x1540 起两字节音节数，之后每节 (索引, 音节长度, 音节GBK)
    pos, pys = 0x1540, {}
    try:
        (count,) = struct.unpack("<H", data[pos:pos + 2])
        pos += 2
        for _ in range(count):
            (idx, ln) = struct.unpack("<HH", data[pos:pos + 4])
            pos += 4
            py = data[pos:pos + ln].decode("gbk", "ignore")
            pos += ln
            pys[idx] = py
        # 词表：0x2628 起，每条 (同音词数, 音节 opini数, 音节索引..., 词长, 词GBK)
        pos = 0x2628
        while pos + 10 <= len(data):
            same, ln = struct.unpack("<HH", data[pos:pos + 4])
            pos += 4
            if same == 0 or ln == 0 or ln > 40:
                break
            idxs = struct.unpack("<%dH" % ln, data[pos:pos + ln * 2])
            pos += ln * 2
            (wln,) = struct.unpack("<H", data[pos:pos + 2])
            pos += 2
            word = data[pos:pos + wln].decode("gbk", "ignore")
            pos += wln + 2  # 词频两字节跳过
            py = " ".join(pys.get(i, "") for i in idxs).strip()
            if word and is_good_word(word):
                out.append((word, py))
    except (struct.error, IndexError):
        pass
    return out


def fetch_local_scel():
    got = {}
    for p in glob.glob(os.path.join(BASE, "cache", "tmp", "scel", "*.scel")):
        n = 0
        for word, py in parse_scel(p):
            got[word] = max(got.get(word, 0), 3000)  # scel 无词频，给中位可信值
            n += 1
        if n:
            print("  [OK] %s +%d 词" % (os.path.basename(p), n))
    return got


def main():
    print("== THUOCL 开放词库 ==")
    words = fetch_thuocls()
    print("== 本地搜狗 scel（cache/tmp/scel/*.scel）==")
    for w, f in fetch_local_scel().items():
        words[w] = max(words.get(w, 0), f)
    if not words:
        print("[热词] 无来源可用（网络不通且无本地 scel），不产出")
        return 1
    items = sorted(words.items(), key=lambda kv: -kv[1])
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("# 热词增量（fetch_hotwords.py 转制：THUOCL + 搜狗 scel）\n")
        f.write("# 2 列格式：词条<TAB>权重；加载时自动注音；权重=来源频次\n")
        for w, fq in items:
            f.write("%s\t%d\n" % (w, fq))
    print("[热词] 产出 %s：%d 词" % (OUT, len(items)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
