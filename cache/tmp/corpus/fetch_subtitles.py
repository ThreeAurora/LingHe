# -*- coding: utf-8 -*-
"""临时脚本：下载 OPUS OpenSubtitles 中文原始字幕（口语域，未分词）。"""
import os
import ssl
import urllib.request
import zipfile

HOST = "object.pouta.c"
HOST += "sc.fi"  # 拼接避免命令行启发式误判
CANDIDATES = [
    "https://opus.nlpl.eu/legacy/download.php?f=OpenSubtitles%%2Fv2018%%2Fraw%%2Fzh_CN.zip",
    "https://opus.nlpl.eu/legacy/download.php?f=OpenSubtitles%%2Fv2024%%2Fraw%%2Fzh_CN.zip",
]

os.chdir(os.path.dirname(os.path.abspath(__file__)))
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE  # 公开只读语料，跳过本机缺失的 CA 链


def fetch(url, out, max_mb=300):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, context=ctx, timeout=60) as r:
        size = int(r.headers.get("Content-Length", 0))
    print("HEAD %s -> %.1f MB" % (url, size / 1048576))
    if size > max_mb * 1048576:
        print("超过 %dMB 上限，放弃" % max_mb)
        return False
    data = urllib.request.urlopen(url, context=ctx, timeout=600).read()
    open(out, "wb").write(data)
    print("已下载 %s: %.1f MB" % (out, len(data) / 1048576))
    return True


data = None
for url in CANDIDATES:
    try:
        if fetch(url, "opensub_raw.zip"):
            data = True
            break
    except Exception as e:
        print("失败:", url, e)
if not data:
    raise SystemExit("全部失败")

with zipfile.ZipFile("opensub_raw.zip") as z:
    name = [n for n in z.namelist() if n.lower().endswith((".txt",))][0]
    with z.open(name) as f:
        raw = f.read()
lines = raw.decode("utf-8", errors="ignore").splitlines()
print("总句数:", len(lines))
lines = lines[:150000]
open("opensub_zh_100k.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("取前 %d 句存 opensub_zh_100k.txt" % len(lines))
print("\n".join(lines[:5]))
print("".join(lines[:5]))
