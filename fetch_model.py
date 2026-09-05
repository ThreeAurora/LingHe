# -*- coding: utf-8 -*-
"""RBT3 模型权重再生脚本（防丢哲学：权重不入库，一条命令拉回）。

来源：hf-mirror.com 的 geofqiu0/rbt3-ONNX 镜像（HuggingFace hfl/rbt3 的
ONNX 导出版，fp16 77MB / int8 37MB）。网络可达性以实测为准（GitHub 系
在本机封锁，hf-mirror 白名单内可达）。

用法：
    python fetch_model.py
"""
import os
import sys
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(BASE, "ai_neural", "rbt3")
REPO = "https://hf-mirror.com/geofqiu0/rbt3-ONNX/resolve/main/"
FILES = [
    "model_fp16.onnx",          # DML GPU 走 fp16（引擎首选）
    "model_int8.onnx",          # CPU 兜底
    "vocab.txt", "config.json", "quantize_config.json",
]


def fetch(name):
    dst = os.path.join(OUT_DIR, name)
    if os.path.isfile(dst) and os.path.getsize(dst) > 10000:
        print("  [有] %s" % name)
        return True
    url = REPO + name
    print("  [拉] %s" % url)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "linghe/1.0"})
        with urllib.request.urlopen(req, timeout=60) as r, open(dst, "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        print("      %d 字节" % os.path.getsize(dst))
        return os.path.getsize(dst) > 10000
    except Exception as e:
        print("      失败: %r" % e)
        try:
            os.remove(dst)
        except OSError:
            pass
        return False


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    ok = all(fetch(n) for n in FILES)
    print("[模型] %s" % ("就绪" if ok else "不完整（重跑可续）"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
