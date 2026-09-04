# -*- coding: utf-8 -*-
"""端侧统计重排器 —— 豆包双层架构的第二层。

豆包那种「0.1 秒内出词、又明显有 AI 味」的架构是两层的：

    第一层  检索召回（<1ms）   Trie / 词表，本质是检索不是生成
    第二层  判别式重排（5~30ms） 对 (上下文, 候选) 打分，一次前向，不生成 token

灵鹤的第一层由 mabiao.py + dict_engine.py 承担（170 万级词条，<0.04ms）；
本模块是第二层的工程实现——用统计语言模型替代神经网络，纯 Python、零依赖，
单次调用 <5ms，同样具备「判别式」的全部关键性质：不生成 token、可批处理、
可端侧运行。

关键边界（主人明确约定，不可逾越）
---------------------------------
- 主码表 mabiao/ 是**固频**：背过的字按码表行序出现，永不参与重排。
- 底库 dicts/ 是**动态调频**：只有底库候选才会被本模块重排。

模型（代价越小越优）
-------------------
    cost(w | prev) = -[ A*log(1+freq(w))              unigram：底库真实词频
                      + L*(len(w)-1)                 长词奖励
                      - P*(len(w)==1)                单字惩罚
                      + U*log(1+user_count(w))       用户个性化
                      + B*log(1+bigram(prev,w)) ]    上下文（在线学习）

为什么需要「长词奖励 L」和「单字惩罚 P」：词频表是分词后的绝对计数，单字词
（一/个/包/子）的计数天然比多字词高出一到两个数量级。若直接用 log 词频相加，
Viterbi 必然把「一个包子」切成「一|个|包|子」——因为四个单字的 log 和更小。
L 与 P 的作用就是抵消单字词频的虚高，让长词切分胜出。两者是联动参数，取值
由 tune.py 在真实用例上扫参确定，不要凭直觉改。
"""

import math
import os
import threading

# ---- 模型参数（由 tune.py 在真实用例上扫参得到，改动前先跑 tune.py）----
# 当前组合 L=12 / P=4 在 10 个整句用例上 top1 命中 9/10（唯一失手的是简拼
# wilygbz，见文件末尾说明）。
A = 1.0      # unigram 词频权重
L = 12.0     # 长词奖励（每多一个字减多少代价）
P = 4.0      # 单字惩罚
U = 3.0      # 用户个性化权重
B = 4.0      # 上下文 bigram 权重（用户在线学习）
B_PRE = 1.5    # 预训练词级共现权重（弱于用户证据，只做平票裁决，与用户取 max 不叠加）
B_PRE_CHAR = 0.0  # 预训练字级共现：实测弊大于利（中性字对太多无区分度，10 例 9→8），默认禁用；
                  # 表仍在 cooc_pre.bin 里，如需启用调成 0.8~1.2
UNK_LOG = 1.4  # 未登录词（底库无词频）的默认 log 频

BEAM = 6        # Viterbi beam 宽度
MAX_SPAN = 4    # 一个词最多横跨几个音节
SPAN_CANDS = 8  # 每个跨度召回多少候选
MAX_KEYS = 16   # 整句最多处理多少音节（安全上限）


class StatReranker:
    def __init__(self, dict_engine, log=print):
        self.de = dict_engine
        self.log = log
        self.bigram = {}          # (prev_word, word) -> count，用户在线学习
        self.pre_word = {}        # 预训练词级共现（SIGHAN 语料统计，见 build_cooc.py）
        self.pre_char = {}        # 预训练字级共现（跨词边界尾字→首字）
        self._next_user = None    # 后继倒排（联想用），load/learn 时懒建
        self._next_pre = None
        self._path = None
        self._dirty = 0
        self._lock = threading.Lock()
        self._viterbi_cache = {}
        self._cache_ver = 0       # bigram 变化时自增，作废旧缓存

    # ---------- 持久化：用户 bigram + 预训练共现 ----------

    def load(self, dir_path):
        """加载用户二元表与预训练共现表（后者由 build_cooc.py 生成，可缺省）。"""
        self._path = os.path.join(dir_path, "user_bigram.txt")
        n = 0
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    p = line.split("\t")
                    if len(p) >= 2:
                        self.bigram[(p[0], p[1])] = int(p[2]) if len(p) > 2 and p[2].isdigit() else 1
                        n += 1
        except OSError:
            pass
        if n:
            self.log("[共现] 载入 %d 条用户二元关系" % n)
        pre_path = os.path.join(dir_path, "cooc_pre.bin")
        try:
            import marshal
            with open(pre_path, "rb") as f:
                pre = marshal.load(f)
            if "word" in pre:  # 新格式：{"word": {...}, "char": {...}}
                self.pre_word = pre["word"]
                self.pre_char = pre["char"]
            else:              # 旧格式：单层词级 dict
                self.pre_word = pre
                self.pre_char = {}
            self.log("[共现] 预训练表 词%d 字%d（SIGHAN 语料）"
                     % (len(self.pre_word), len(self.pre_char)))
        except (OSError, ValueError):
            self.pre_word = {}
            self.pre_char = {}
        self._build_next_index()
        return n

    def flush(self):
        """把 bigram 表写回磁盘（退出时调用）。只写用户表——预训练表是只读的。"""
        with self._lock:
            if not self._dirty or not self._path:
                return
            data = sorted(self.bigram.items(), key=lambda kv: -kv[1])
            self._dirty = 0
        try:
            with open(self._path, "w", encoding="utf-8") as f:
                f.write("# 灵鹤用户二元共现（上词\\t下词\\t次数）\n")
                for (a, b), c in data[:20000]:  # 只留最常用两万条，防止文件无限膨胀
                    f.write("%s\t%s\t%d\n" % (a, b, c))
        except OSError:
            pass

    def learn(self, prev_word, word):
        """用户上屏时调用：记住「上一个词 -> 这个词」的共现。"""
        if not prev_word or not word or prev_word == word:
            return
        with self._lock:
            k = (prev_word, word)
            self.bigram[k] = self.bigram.get(k, 0) + 1
            self._dirty += 1
            self._cache_ver += 1
            self._next_user.setdefault(prev_word, []).append(word)

    def _build_next_index(self):
        """后继倒排索引：联想查询要按上词取 top 后继，倒排后 O(后继数)。"""
        self._next_pre = {}
        for (a, b) in self.pre_word:
            self._next_pre.setdefault(a, []).append(b)
        self._next_user = {}
        for (a, b) in self.bigram:
            self._next_user.setdefault(a, []).append(b)

    def next_word(self, prev, n=1):
        """联想 prev 的下一个词（豆包式上屏联想）。

        用户表与预训练词级表取强者；只返回底库存在的词（不推生僻人名等）。
        返回 top n 词列表（可为空）。
        """
        if not prev:
            return []
        if self._next_pre is None:
            self._build_next_index()
        cand = {}
        for b in self._next_user.get(prev, ()):
            c = self.bigram.get((prev, b), 0)
            if b in self.de.word_py:
                cand[b] = max(cand.get(b, 0.0), B * math.log(1 + c))
        for b in self._next_pre.get(prev, ()):
            if b in cand:
                continue
            c = self.pre_word.get((prev, b), 0)
            if b in self.de.word_py and c:
                cand[b] = max(cand.get(b, 0.0), B_PRE * math.log(1 + c))
        if not cand:
            return []
        ranked = sorted(cand.items(), key=lambda kv: -kv[1])
        return [w for w, _ in ranked[:n]]

    # ---------- 打分 ----------

    def _cost(self, word):
        """单个词的 unigram 代价（越小越优）。"""
        w = self.de.weight(word)
        c = -A * (math.log(1 + w) if w > 0 else UNK_LOG)
        n = len(word)
        if n > 1:
            c -= L * (n - 1)
        else:
            c += P
        u = self.de.user_count(word)
        if u:
            c -= U * math.log(1 + u)
        return c

    def _bi_bonus(self, prev, word):
        """上文加分：用户在线学习（B）、预训练词级（B_PRE）、预训练字级（B_PRE_CHAR）
        三者取强者，不叠加。

        不叠加的原因：预训练语料规模与用户使用量差若干个数量级，线性相加会让
        预训练永久淹没用户证据；取 max 则用户打过一次的词立刻以更高权重生效。
        字级（prev 尾字 × word 首字）是词级未覆盖时的泛化兜底。
        """
        if not prev:
            return 0.0
        bonus = 0.0
        c = self.bigram.get((prev, word), 0)
        if c:
            bonus = B * math.log(1 + c)
        if self.pre_word:
            c2 = self.pre_word.get((prev, word), 0)
            if c2:
                bonus = max(bonus, B_PRE * math.log(1 + c2))
        if self.pre_char:
            c3 = self.pre_char.get((prev[-1], word[0]), 0)
            if c3:
                bonus = max(bonus, B_PRE_CHAR * math.log(1 + c3))
        return bonus

    def rerank(self, cands, prev_word=None):
        """对底库候选按 (unigram + 上文 + 用户习惯) 重排。

        注意：只应传入底库候选。码表（mabiao）候选是固频，不参与重排。
        """
        if not cands:
            return []
        scored = [(self._cost(w) - self._bi_bonus(prev_word, w), i, w)
                  for i, w in enumerate(cands)]
        scored.sort()
        return [w for _, _, w in scored]

    # ---------- 整句切分（Viterbi / beam search）----------

    def _span_cands(self, keys, mode, limit):
        """取 [i:j) 这段音节对应的词。"""
        key = " ".join(keys)
        d = self.de.by_pinyin if mode == "py" else self.de.by_initial
        lst = d.get(key)
        if not lst:
            return ()
        return tuple(w for _, w in lst[:limit])

    def viterbi(self, keys, mode="ini", n=5, ret_cost=False):
        """在音节序列上切分出最优整句，返回 n 个候选整句。

        keys: ['w','i','l','y','g','b','z']（mode="ini" 声母简拼）
              或 ['zhan','mu','si']（mode="py" 全拼音节）
        ret_cost=True 时返回 (句子, 代价, 音节数)，供调用方把全拼/简拼两路
        候选按「每字平均代价」合并排序——两路的代价不可直接比较（词数不同），
        但每字平均代价语义一致。
        """
        keys = tuple(keys[:MAX_KEYS])
        if not keys:
            return []
        ck = (keys, mode, n, self._cache_ver)
        hit = self._viterbi_cache.get(ck)
        if hit is not None:
            return hit
        out = self._viterbi_raw(keys, mode, n, ret_cost)
        if len(self._viterbi_cache) > 256:
            self._viterbi_cache.clear()
        self._viterbi_cache[ck] = out
        return out

    def _viterbi_raw(self, keys, mode, n, ret_cost=False):
        m = len(keys)
        # dp[j] = [(cost, words_tuple, last_word), ...] 保留 BEAM 条最优
        dp = [None] * (m + 1)
        dp[0] = [(0.0, (), "")]
        for j in range(1, m + 1):
            cand = []
            lo = max(0, j - MAX_SPAN)
            for i in range(lo, j):
                prev_states = dp[i]
                if not prev_states:
                    continue
                words_here = self._span_cands(keys[i:j], mode, SPAN_CANDS)
                if not words_here:
                    continue
                for w in words_here:
                    cw = self._cost(w)
                    for c0, seq, prev in prev_states:
                        cand.append((c0 + cw - self._bi_bonus(prev, w), seq + (w,), w))
            if not cand:
                # 该位置切不出任何词：退回单字，保证链路不断
                prev_states = dp[j - 1]
                if not prev_states:
                    dp[j] = None
                    continue
                one = self._span_cands(keys[j - 1:j], mode, 4)
                if one:
                    for w in one:
                        for c0, seq, prev in prev_states:
                            cand.append((c0 + self._cost(w) + 6.0, seq + (w,), w))
                if not cand:  # 连单字都没有：用拼音字面占位
                    for c0, seq, prev in prev_states:
                        cand.append((c0 + 30.0, seq + (keys[j - 1],), keys[j - 1]))
            cand.sort(key=lambda t: t[0])
            dp[j] = cand[:BEAM]
        final = dp[m]
        if not final:
            return []
        out, seen = [], set()
        for c, seq, _ in final:
            s = "".join(seq)
            if s not in seen:
                seen.add(s)
                out.append((s, c, m) if ret_cost else s)
            if len(out) >= n:
                break
        return out


if __name__ == "__main__":
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dict_engine import DictEngine
    base = os.path.dirname(os.path.abspath(__file__))
    de = DictEngine()
    de.load_dir(os.path.join(base, "dicts"))
    rr = StatReranker(de)
    rr.load(os.path.join(base, "dicts"))
    for keys, mode, want in [
        (list("wilygbz"), "ini", "我吃了一个包子"),
        (list("wilygbz"), "ini", None),
        ("zhan mu si".split(), "py", None),
        ("wo chi le yi ge bao zi".split(), "py", None),
        ("xiao he yin xing".split(), "py", None),
    ]:
        t0 = time.perf_counter()
        got = rr.viterbi(keys, mode, 3)
        ms = (time.perf_counter() - t0) * 1000
        print("%-28s %s  %.2fms -> %s" % ("".join(keys), mode, ms, got))
