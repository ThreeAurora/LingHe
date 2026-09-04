# -*- coding: utf-8 -*-
"""兜底词库引擎（dicts/ 文件夹，优先级低于 mabiao/ 码表）。

数据源：雾凇拼音 cn_dicts（rime 格式：词条<TAB>全拼<TAB>权重，# 注释）。
能力：
1. 全拼串索引：用户双拼音节流解码成拼音串（"zhan mu si"）查词；
2. 声母键串索引（简拼）：wilygbz = 我吃了一个包子——每字取双拼声母键（zh/ch/sh→v/i/u），
   长句词库无命中时由 AI 整句兜底；
3. 自动记忆：commit 词库没有的新词 → user_dict.txt（词+拼音+次数），下次直接命中并调频。
"""

import os
import threading
from collections import defaultdict

# 拼音声母 → 双拼声母键
SM_KEY = {"zh": "v", "ch": "i", "sh": "u"}

_YUN_FIRST = {}  # 兜底：简拼需要韵母信息时不用，此处仅声母


_SM_CACHE = {}


def _sm_key(syllable: str) -> str:
    """全拼音节 → 双拼声母键（zh/ch/sh→v/i/u，其余首字母）。

    加载 152 万词时会调用约 450 万次（每词每音节一次），但音节种类只有一千
    多种；加缓存后命中率 99.9% 以上，是加载耗时的主要优化点之一。
    """
    k = _SM_CACHE.get(syllable)
    if k is None:
        if syllable.startswith("zh"):
            k = "v"
        elif syllable.startswith("ch"):
            k = "i"
        elif syllable.startswith("sh"):
            k = "u"
        else:
            k = syllable[0] if syllable else ""
        _SM_CACHE[syllable] = k
    return k


class DictEngine:
    def __init__(self, log=print):
        self.log = log
        self.by_pinyin = {}      # "zhan mu si" -> [(weight, word)...] 降序
        self.by_initial = {}     # "w i l y g b z" -> [(weight, word)...]
        self.word_py = {}        # word -> pinyin_str（供记忆/调频）
        self.word_weight = {}    # word -> 底库固有词频（统计重排的 unigram 来源）
        self.user_counts = {}    # word -> count（会话内调频）
        self.char_py = {}        # 字 -> 拼音（注音表，供无注音词库自动注音）
        self.size = 0
        self.loaded = False
        self.loading = False
        self.user_dict_path = None
        self._lock = threading.Lock()
        self._dirty = {}         # word -> count，待落盘的用户词（批量写，减少 IO）
        self._since_flush = 0

    # ---------- 加载 ----------

    def load_dir(self, dir_path: str) -> int:
        """同步加载（含后台模式由 load_dir_async 调用）。

        二进制缓存：首次文本解析约 20~23s（152.9 万词），之后走 marshal 缓存
        数秒内完成。缓存键=词库文件 (名字,大小,mtime) 签名，任何词库文件变化
        自动失效重建。缓存文件可再生，不入库（gitignore）。
        """
        import time
        t0 = time.perf_counter()
        if not os.path.isdir(dir_path):
            return 0
        cache_path = os.path.join(dir_path, ".cache.bin")
        sig = self._files_sig(dir_path)
        if sig and self._load_cache(cache_path, sig):
            self._after_load(dir_path, t0)
            return self.size
        self._load_char_pinyin(os.path.join(dir_path, "char_pinyin.txt"))
        raw = []  # (weight, word, py_str or None)
        flat_files = []
        for name in sorted(os.listdir(dir_path)):
            if not name.endswith(".yaml") and not name.endswith(".txt"):
                continue
            if name == "user_dict.txt":
                continue
            part, flat = self._load_file(os.path.join(dir_path, name))
            if part:
                raw += part
                if flat:
                    flat_files.append(name)
        # 用户词优先：权重放大
        raw += [(100000 + c, w, p) for w, (p, c) in self._load_user_dict(dir_path).items()]
        by_pinyin = self.by_pinyin
        by_initial = self.by_initial
        word_py = self.word_py
        word_weight = self.word_weight
        for weight, word, py in raw:
            if py is None:
                py = self._auto_annotate(word)
                if py is None:
                    continue  # 注不出音的词放弃
            old = word_py.get(word)
            if old is not None and old != py:
                continue  # 多音词取首个读音，避免索引分裂
            word_py[word] = py
            if weight > word_weight.get(word, 0):
                word_weight[word] = weight
            by_pinyin.setdefault(py, []).append((weight, word))
            ini = " ".join(_sm_key(s) for s in py.split())
            by_initial.setdefault(ini, []).append((weight, word))
        for d in (by_pinyin, by_initial):
            for k in d:
                d[k].sort(key=lambda t: -t[0])
        # 合法音节全集（注音表覆盖全部读音），用于记忆新词前的拼音校验
        self.valid_sylls = set()
        for py in self.char_py.values():
            self.valid_sylls.update(py.split())
        self.size = len(self.word_py)
        self.loaded = True
        self.user_dict_path = os.path.join(dir_path, "user_dict.txt")
        ms = round((time.perf_counter() - t0) * 1000)
        extra = ("，其中 %s 无词频列已按低可信权重处理" % "/".join(flat_files)) if flat_files else ""
        self.log("[底库] %d 个词条（全拼/简拼双索引），文本解析 %dms%s，已写缓存下次秒开"
                 % (self.size, ms, extra))
        self._dump_cache(cache_path, sig)
        return self.size

    # ---------- 二进制缓存（marshal：比文本解析快数倍，词库文件变化自动失效）----------

    def _files_sig(self, dir_path):
        out = []
        try:
            names = sorted(os.listdir(dir_path))
        except OSError:
            return None
        for name in names:
            if not (name.endswith(".yaml") or name.endswith(".txt")) or name == "user_dict.txt":
                continue
            p = os.path.join(dir_path, name)
            try:
                st = os.stat(p)
                out.append((name, st.st_size, int(st.st_mtime)))
            except OSError:
                return None
        return tuple(out)

    def _load_cache(self, path, sig):
        try:
            import marshal
            with open(path, "rb") as f:
                data = marshal.load(f)
            if data.get("sig") != sig:
                return False
            self.by_pinyin = data["by_pinyin"]
            self.by_initial = data["by_initial"]
            self.word_py = data["word_py"]
            self.word_weight = data["word_weight"]
            self.char_py = data["char_py"]
            self.valid_sylls = data["valid_sylls"]
            self.size = len(self.word_py)
            return True
        except Exception:
            return False

    def _dump_cache(self, path, sig):
        try:
            import marshal
            with open(path, "wb") as f:
                marshal.dump({
                    "sig": sig,
                    "by_pinyin": self.by_pinyin,
                    "by_initial": self.by_initial,
                    "word_py": self.word_py,
                    "word_weight": self.word_weight,
                    "char_py": self.char_py,
                    "valid_sylls": self.valid_sylls,
                }, f)
        except Exception:
            try:
                os.remove(path)  # 半成品宁可删掉，别留一个损坏缓存
            except OSError:
                pass

    def _after_load(self, dir_path, t0):
        """缓存命中路径：合入用户词（缓存不含 user_dict，它变化频繁）+ 日志。"""
        import time
        d = self._load_user_dict(dir_path)
        for w, (p, c) in d.items():
            if w not in self.word_py:
                self.word_py[w] = p
                self.word_weight[w] = 0
                self.by_pinyin.setdefault(p, []).append((100000 + c, w))
                self.by_initial.setdefault(" ".join(_sm_key(s) for s in p.split()), []).append((100000 + c, w))
                self.size += 1
        self.loaded = True
        self.user_dict_path = os.path.join(dir_path, "user_dict.txt")
        ms = round((time.perf_counter() - t0) * 1000)
        self.log("[底库] %d 个词条（全拼/简拼双索引），缓存加载 %dms" % (self.size, ms))

    def load_dir_async(self, dir_path: str, on_done=None):
        """后台加载：输入法立即可用，底库就绪后自动上线。"""
        self.loading = True

        def work():
            try:
                self.load_dir(dir_path)
            finally:
                self.loading = False
            if on_done:
                on_done(self.loaded)
        threading.Thread(target=work, daemon=True).start()

    # ---------- 注音 ----------

    def _load_char_pinyin(self, path: str):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or ":" not in line:
                        continue
                    head, _, tail = line.partition(":")
                    py_part = tail.split("#")[0].strip()
                    if len(head) != 6 or not head.startswith("U+"):
                        continue
                    try:
                        ch = chr(int(head[2:], 16))
                    except ValueError:
                        continue
                    first = py_part.split(",")[0]
                    # 去声调数字：pin1 -> pin
                    sylls = []
                    for s in first.split():
                        s = "".join(c for c in s if not c.isdigit())
                        if s:
                            sylls.append(s)
                    if sylls:
                        self.char_py[ch] = " ".join(sylls)
        except OSError:
            pass

    def _auto_annotate(self, word: str):
        """无注音词条逐字注音（多音字取注音表首读）。"""
        sylls = []
        for ch in word:
            py = self.char_py.get(ch)
            if py is None:
                return None
            sylls.append(py)
        return " ".join(sylls)

    @staticmethod
    def _load_file(path: str):
        """rime 词典，两种列格式都要认：

        - ``词条<TAB>拼音[<TAB>权重]``（base.dict.yaml，带真实词频 1~19M）
        - ``词条<TAB>权重``（tencent.dict.yaml，columns 只有 text+weight，无拼音列）

        旧版把第二种的第二列当拼音，导致 98 万词全部挂在假拼音 ``"100"`` 下，
        查询永不命中——底库名义 152 万、实际只有 base 的 54 万生效。此处按
        「第二列首字符是否为字母」区分拼音列与权重列。

        返回 (weight, word, py_or_None, flat_weight)：flat_weight 表示该文件
        所有权重相同（无区分度，如 tencent 的 100），需按低可信处理。
        """
        out = []
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for raw_line in f:
                    line = raw_line.rstrip("\r\n")
                    if not line or line[0] in "#.":
                        continue
                    parts = line.split("\t")
                    if len(parts) < 2:
                        continue
                    word = parts[0].strip()
                    if not word or not all(0x4E00 <= ord(ch) <= 0x9FFF for ch in word):
                        continue  # 只收纯汉字词
                    col1 = parts[1].strip()
                    if col1[:1].isalpha():
                        py = col1.lower()
                        wraw = parts[2].strip() if len(parts) > 2 else ""
                    else:
                        py, wraw = None, col1
                    try:
                        weight = int(wraw) if wraw else 0
                    except ValueError:
                        weight = 0
                    out.append((weight, word, py))
        except OSError:
            pass
        if not out:
            return out, False
        # 平权检测：权重全部相同 -> 该文件没有词频区分度
        w0 = out[0][0]
        flat = all(t[0] == w0 for t in out)
        if flat:
            out = [(50, w, p) for _, w, p in out]  # 压到低可信区间，靠长度奖励区分
        return out, flat

    # ---------- 查询 ----------

    def lookup_pinyin(self, py_str: str, n: int = 9):
        """全拼串查询（空格分隔），返回词条（权重+用户调频 降序）。"""
        with self._lock:
            hits = list(self.by_pinyin.get(py_str, ()))
        hits.sort(key=lambda t: -(t[0] + self.user_counts.get(t[1], 0) * 100000))
        out, seen = [], set()
        for weight, word in hits:
            if word in seen:
                continue
            seen.add(word)
            out.append(word)
            if len(out) >= n:
                break
        return out

    def lookup_initial(self, key_str: str, n: int = 9):
        """简拼声母键串查询（空格分隔双拼声母键）。"""
        with self._lock:
            hits = self.by_initial.get(key_str, ())
        out, seen = [], set()
        for weight, word in hits:
            if word in seen:
                continue
            seen.add(word)
            out.append(word)
            if len(out) >= n:
                break
        return out

    # ---------- 自动记忆/调频 ----------

    def _load_user_dict(self, dir_path: str):
        path = os.path.join(dir_path, "user_dict.txt")
        d = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        try:
                            d[parts[0]] = (parts[1], int(parts[2]) if len(parts) > 2 else 1)
                        except ValueError:
                            d[parts[0]] = (parts[1], 1)
        except OSError:
            pass
        self.user_dict_path = path
        # 原为 c-1（只统计本会话增量），导致历史词频重启即作废、调频失效；
        # 改为完整继承文件里的累计次数。
        self.user_counts = {w: c for w, (p, c) in d.items()}
        return d

    FLUSH_EVERY = 30  # 累计 30 次上屏合并写盘一次

    def remember(self, word: str, py_str: str):
        """上屏一次记一次：新词入库，旧词调频，两者都持久化。

        旧版只有「底库没有的新词」才写盘，而绝大多数上屏词底库里都有——等于
        打了半年的常用词重启后顺序全部回到解放前，记忆功能实质失效。改为全部
        落盘，但走脏表批量写（每 30 次上屏或 flush() 时合并一次），不碰每次
        上屏的关键路径。
        """
        with self._lock:
            c = self.user_counts.get(word, 0) + 1
            self.user_counts[word] = c
            if word not in self.word_py:
                self.word_py[word] = py_str
                self.word_weight[word] = 0
                self.by_pinyin.setdefault(py_str, []).append((100000 + c, word))
                ini = " ".join(_sm_key(s) for s in py_str.split())
                self.by_initial.setdefault(ini, []).append((100000 + c, word))
                self.size += 1
            self._dirty[word] = c
            self._since_flush += 1
        if self._since_flush >= self.FLUSH_EVERY:
            self.flush()

    def flush(self):
        """把脏表合并写回 user_dict.txt（全量重写，通常只有几千行）。"""
        with self._lock:
            if not self._dirty:
                return
            self._dirty.clear()
            self._since_flush = 0
            snapshot = {w: (self.word_py.get(w, ""), c)
                        for w, c in self.user_counts.items()}
        path = self.user_dict_path
        if not path:
            return
        old = self._read_user_file(path)
        for w, (py, c) in snapshot.items():
            if not py:
                continue
            prev = old.get(w)
            old[w] = (py, max(c, prev[1]) if prev else c)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("# 灵鹤用户词库：上屏自动记忆（词\\t拼音\\t次数）\n")
                for w, (py, c) in sorted(old.items(), key=lambda kv: -kv[1][1]):
                    f.write("%s\t%s\t%d\n" % (w, py, c))
        except OSError:
            pass

    @staticmethod
    def _read_user_file(path):
        d = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    p = line.split("\t")
                    if len(p) >= 2:
                        try:
                            d[p[0]] = (p[1], int(p[2]) if len(p) > 2 else 1)
                        except ValueError:
                            d[p[0]] = (p[1], 1)
        except OSError:
            pass
        return d

    def weight(self, word: str) -> int:
        """底库固有词频（0 = 无词频/未登录词），供统计重排做 unigram。"""
        return self.word_weight.get(word, 0)

    def user_count(self, word: str) -> int:
        """用户历史上屏次数，供统计重排做个性化加分。"""
        return self.user_counts.get(word, 0)
