# -*- coding: utf-8 -*-
"""Qwen2.5-0.5B-Instruct 模型再生脚本（防丢哲学：能再生）。

下载到 ai_llm/qwen25-05b-hf/，共 6 文件 ~1GB（fp16 safetensors 951MB）。
用法：E:/miniconda3/python.exe ai_llm/fetch_qwen.py
"""
import os
import sys
import urllib.request
import ssl

BASE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(BASE, "qwen25-05b-hf")
REPO = "https://hf-mirror.com/Qwen/Qwen2.5-0.5B-Instruct/resolve/main"
FILES = ["config.json", "generation_config.json", "model.safetensors",
         "tokenizer.json", "tokenizer_config.json", "merges.txt"]

ctx = ssl._create_unverified_context()  # 本机证书链对 hf-mirror 不全


def fetch(name):
    dest = os.path.join(DEST, name)
    if os.path.isfile(dest) and os.path.getsize(dest) > 0:
        print("已有 %s (%.1fMB)，跳过" % (name, os.path.getsize(dest) / 1e6))
        return
    url = "%s/%s" % (REPO, name)
    print("下载 %s ..." % url)
    tmp = dest + ".part"
    urllib.request.urlretrieve(url, tmp)  # nosec —— 防丢再生脚本，静态 URL
    os.replace(tmp, dest)
    print("  -> %s (%.1fMB)" % (name, os.path.getsize(dest) / 1e6))


if __name__ == "__main__":
    os.makedirs(DEST, exist_ok=True)
    for f in FILES:
        fetch(f)
    print("完成。QwenJudge 将从该目录加载。")
