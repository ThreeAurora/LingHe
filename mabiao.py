# -*- coding: utf-8 -*-
"""码表加载与查询。

格式约定（以"简单鹤V9.3.0.txt"实测为准）：
    词条<TAB>编码        例：艾\taiz 、爱情\taq 、阿\taa
兼容反向（编码<TAB>词条）与空格分隔；# 开头为注释；同码多行 = 多候选，
按文件行序保存即"固频"（码表作者排好的频度顺序）。

查询：exact(code) -> [词条, ...]  O(1)。
"""

import os
import sys
import time

_CODE_CHARS = set("abcdefghijklmnopqrstuvwxyz;")


def _looks_like_code(s: str) -> bool:
    return bool(s) and all(c in _CODE_CHARS for c in s)


def _looks_like_word(s: str) -> bool:
    return bool(s) and not _looks_like_code(s)


class MaBiao:
    def __init__(self):
        self._table = {}          # code -> [word, ...]（保序=固频）
        self._prefix = {}         # code前缀 -> [word, ...]（保序=固频，去重）
        self.size = 0
        self.files = []
        self.loaded = False
        self.load_ms = 0.0

    def load_dir(self, dir_path: str) -> int:
        """加载目录下全部 .txt/.yaml 码表文件，返回词条总数。"""
        t0 = time.perf_counter()
        table = {}
        prefix = {}
        files = []
        if not os.path.isdir(dir_path):
            self._table, self._prefix, self.files, self.loaded = {}, {}, [], False
            return 0
        for name in sorted(os.listdir(dir_path)):
            if not name.lower().endswith((".txt", ".yaml", ".yml")):
                continue
            if name.startswith("_") or "辅助" in name:  # 说明文件/辅码表跳过（辅码表由 fuma.py 加载）
                continue
            path = os.path.join(dir_path, name)
            n = self._load_file(path, table, prefix)
            if n:
                files.append((name, n))
        self._table = table
        self._prefix = prefix
        self.files = files
        self.size = sum(len(v) for v in table.values())
        self.load_ms = (time.perf_counter() - t0) * 1000
        self.loaded = True
        return self.size

    @staticmethod
    def _load_file(path: str, table: dict, prefix: dict) -> int:
        count = 0
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    line = raw.rstrip("\r\n").strip()
                    if not line or line.startswith("#") or line.startswith(";"):
                        continue
                    # 分隔：优先 TAB，其次多空格
                    if "\t" in line:
                        left, right = line.split("\t", 1)
                    else:
                        parts = line.split()
                        if len(parts) < 2:
                            continue
                        left, right = parts[0], " ".join(parts[1:])
                    left, right = left.strip(), right.strip()
                    if not left or not right:
                        continue
                    # 判向：简单鹤是"词条<TAB>编码"；兼容反向
                    if _looks_like_word(left) and _looks_like_code(right):
                        word, code = left, right
                    elif _looks_like_code(left) and _looks_like_word(right):
                        word, code = right, left
                    else:
                        continue
                    code = code.lower()
                    bucket = table.get(code)
                    if bucket is None:
                        table[code] = [word]
                    elif word not in bucket:
                        bucket.append(word)
                    else:
                        continue  # 同码同词已收录，前缀层也不必再加
                    count += 1
                    # 前缀索引：1..len(code) 每层都挂（打字途中即时出候选）
                    # 不在索引层查重（大桶 O(k) 拖慢加载），查询层用 seen 去重
                    for i in range(1, len(code) + 1):
                        pb = prefix.get(code[:i])
                        if pb is None:
                            prefix[code[:i]] = [word]
                        else:
                            pb.append(word)
        except OSError:
            pass
        return count

    def exact(self, code: str):
        """精确查询。返回码表命中的候选（固频序）；无命中返回 []。"""
        return self._table.get(code, ())

    def prefix(self, code: str, limit: int = 64):
        """前缀查询：所有以 code 开头的码的词条（固频序），供组码途中即时出候选。"""
        return self._prefix.get(code, ())[:limit]

    def has(self, code: str) -> bool:
        return code in self._table


if __name__ == "__main__":
    mb = MaBiao()
    base = os.path.dirname(os.path.abspath(__file__))
    n = mb.load_dir(os.path.join(base, "mabiao"))
    print("加载词条:", n, "码数:", len(mb._table), "耗时ms:", round(mb.load_ms))
    for c in ["aih", "aa", "an", "aq", "ulpb", "zzzz"]:
        print(c, "->", mb.exact(c)[:8])
