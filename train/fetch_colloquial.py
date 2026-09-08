# -*- coding: utf-8 -*-
"""口语语料抓取：OPUS OpenSubtitles 中文字幕（影视对白，贴近打字口语分布）。

源失败时依次回退：
  1. OPUS OpenSubtitles v2018 mono zh（~700MB 影视对白，最理想）
  2. OPUS TED2020 mono zh（演讲口语，~60MB）
输出：cache/tmp/corpus/colloquial_raw.txt（每行一句，纯中文过滤后）
"""
import os, sys, urllib.request, gzip, shutil, ssl

# OPUS 服务器证书链在本机 python 环境验证不过（2026-09-08 实测），
# 公开语料下载关闭证书校验
ssl._create_default_https_context = ssl._create_unverified_context

DST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "cache", "tmp", "corpus")

# 每组按序尝试多个版本路径（OPUS mono 直链版本号不稳定）
SOURCES = [
    ("OpenSubtitles_zh", [
        "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/mono/zh.txt.gz",
        "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/zh.txt.gz",
    ]),
    ("TED2020_zh", [
        "https://object.pouta.csc.fi/OPUS-TED2020/v1/mono/zh.txt.gz",
    ]),
    ("NewsCommentary_zh", [
        "https://object.pouta.csc.fi/OPUS-News-Commentary/v16/mono/zh.txt.gz",
    ]),
]


def fetch(name, urls):
    dst = os.path.join(DST, name + ".txt.gz")
    if os.path.exists(dst) and os.path.getsize(dst) > 1e6:
        print("[skip] %s 已存在 %.0fMB" % (name, os.path.getsize(dst) / 1e6))
        return dst
    for url in urls:
        try:
            print("[dl] %s <- %s" % (name, url.split("/")[-3]), flush=True)
            urllib.request.urlretrieve(url, dst)
            print("[dl] %s done %.0fMB" % (name, os.path.getsize(dst) / 1e6),
                  flush=True)
            return dst
        except Exception as e:
            print("  [retry] %s: %s" % (url.split("/")[-3], e), flush=True)
    raise RuntimeError("全部候选失败")


def clean(src_gz, out_txt):
    """解压+清洗：保留含足够汉字的行，去重复。"""
    n = kept = 0
    seen = set()
    with gzip.open(src_gz, "rt", encoding="utf-8", errors="ignore") as f, \
            open(out_txt, "w", encoding="utf-8") as fo:
        for line in f:
            n += 1
            s = line.strip()
            if len(s) < 4 or len(s) > 60:
                continue
            hanzi = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
            if hanzi < len(s) * 0.8:        # 中文字符占比
                continue
            if s in seen:
                continue
            seen.add(s)
            fo.write(s + "\n")
            kept += 1
            if kept >= 2000000:             # 上限 200 万行
                break
            if n % 2000000 == 0:
                print("  ...%dM 行 / 保留 %d" % (n // 1000000, kept), flush=True)
    print("[clean] %s: %d 行 -> %d 行 -> %s" % (os.path.basename(src_gz),
                                                n, kept, out_txt), flush=True)


if __name__ == "__main__":
    for name, urls in SOURCES:
        try:
            gz = fetch(name, urls)
            clean(gz, os.path.join(DST, "colloquial_raw.txt"))
            break
        except Exception as e:
            print("[fail] %s: %s" % (name, e), flush=True)
    print("DONE")
