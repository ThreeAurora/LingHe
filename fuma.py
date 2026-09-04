# -*- coding: utf-8 -*-
"""手心辅助码表：字 <TAB> 2键辅助码（同字可多行=多音字多码）。

主人钦定用法：**只用首键**（第 1 位）做筛选，第 2 位弃用。
故对外接口只有 first_keys(字) -> {首键集合}。

如：异 tz / 异 tp -> 首键 {t, p}；议 ch / 议 cy -> 首键 {c, h}。
"""

import os


class FuMa:
    def __init__(self):
        self._by_word = {}    # 字 -> [完整辅码...]（保序，供未来扩展）
        self._first = {}      # 字 -> {首键...}（查询主接口）
        self.size = 0
        self.loaded = False

    def load_dir(self, dir_path: str) -> int:
        import glob
        count = 0
        for path in sorted(glob.glob(os.path.join(dir_path, "*辅助*.txt")) +
                           glob.glob(os.path.join(dir_path, "*fuma*.txt"))):
            count += self._load_file(path)
        self.size = count
        self.loaded = count > 0
        return count

    def _load_file(self, path: str) -> int:
        count = 0
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for raw in f:
                    line = raw.rstrip("\r\n").strip()
                    if not line or line.startswith("#"):
                        continue
                    if "\t" in line:
                        word, code = line.split("\t", 1)
                    else:
                        parts = line.split()
                        if len(parts) != 2:
                            continue
                        word, code = parts
                    word, code = word.strip(), code.strip().lower()
                    if len(word) != 1 or len(code) != 2 or not code.isalpha():
                        continue
                    self._by_word.setdefault(word, [])
                    if code not in self._by_word[word]:
                        self._by_word[word].append(code)
                    self._first.setdefault(word, set()).add(code[0])
                    count += 1
        except OSError:
            pass
        return count

    def first_keys(self, word: str):
        """单字的辅码首键集合。"""
        return self._first.get(word, ())

    def word_match(self, word: str, key: str) -> bool:
        """词中某字是否拥有首键=key 的辅码。"""
        return key in self._first.get(word, ())


if __name__ == "__main__":
    fm = FuMa()
    import os
    n = fm.load_dir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mabiao"))
    print("辅码行:", n, "字数:", len(fm._first))
    for w in ["异", "议", "詹", "姆", "斯", "的"]:
        print(w, sorted(fm.first_keys(w)))
