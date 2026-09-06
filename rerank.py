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


def _is_cjk(word):
    """纯中文词判定（英文/混合串一律 False）。"""
    return bool(word) and all("\u4e00" <= ch <= "\u9fff" for ch in word)

# ---- 模型参数（由 tune.py 在真实用例上扫参得到，改动前先跑 tune.py）----
# 当前组合 L=12 / P=4 在 10 个整句用例上 top1 命中 9/10（唯一失手的是简拼
# wilygbz，见文件末尾说明）。
A = 1.0      # unigram 词频权重
L = 12.0     # 长词奖励（每多一个字减多少代价）
P = 4.0      # 单字惩罚
U = 3.0      # 用户个性化权重
B = 4.0      # 上下文 bigram 权重（用户在线学习）
B_PRE = 1.5    # 预训练词级共现权重（弱于用户证据，只做平票裁决，与用户取 max 不叠加）
B_PRE_CHAR = 0.4  # 预训练字级共现（微开）。2026-09-05 复扫：P 扫参下 0.4 与 0.0 同为
                  # 9/13，但 0.4 让 (吃,了)=74 这类词表缺失的字级连接进场
                  # （(吃,包子) 词级对已由短语挖矿补上，字级只管 词干+助词）。
                  # 0.8 以上仍净伤害；换现代口语语料后再重扫。
UNK_LOG = 1.4  # 未登录词（底库无词频）的默认 log 频

DECAY = 0.5    # 多上文衰减：第 i 个上文词的 bonus 乘 DECAY^i。
               # 0.5 = 紧邻全额、隔一词 1/2、隔两词 1/4——「我吃了一个」里
               # 动词「吃」在两个词之外，靠这个通道参与裁决（bz→包子 案）。

BEAM = 16       # Viterbi beam 宽度。6 会被 2 字词占满：1+1 单字对（吃|了 -21.4）
                # 系统性比 2 字词（成立 -25.9）贵 ~4 nat，整句需要的词根在 j=2
                # 就被挤出 beam（ilygbz 案）。16 让 前缀|一个 这类「先吃强共现
                # 红利」的路径活到中段；代价=长码 viterbi ~15ms（有缓存，可接受）。
MAX_SPAN = 4    # 一个词最多横跨几个音节
LONG_SPAN_BOOST = 8.0   # 长跨度（3~4音节）top3 温和奖励。注意只是微调：曾试
                        # 12/24/32 想把西湖醋鱼顶进 beam，全部失败且引发军备
                        # 竞赛（「后槽牙/会采用」都是 hcy 的 top3，竞争句叠
                        # 双 boost 反超）——统计 beam 装不下口语专名串是零和
                        # 死局，正解是 anchor_sentences 锚定通道 + 神经裁决。
SPAN_CANDS = 48  # 每个跨度召回多少候选（去重前）。8 太窄：高频词（如「现在」之于 xz）
                 # 会被同键位的低频词挤出 span，整句从第一个词就错（xzdwtu 案）。
                 # 24 也不够：by_initial 带重复词条且单字池极深（吃 在 ch-单字里
                 # 排 109），24 会把整句需要的动词第一个音节就漏掉（ilygbz 案），
                 # 48+去重后 覆盖到唯一排名 ~70。
MAX_KEYS = 16   # 整句最多处理多少音节（安全上限）

ANCHOR2_FREQ = 3000   # 2 音节锚点的最低词频。2 音节 rank≤2 的词是强锚
                      # （东西 dx rank2），但没频次门槛的话每个双声母键的
                      # rank2（年味/那次）全成锚，垃圾句淹池。


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
        self.sp2 = None           # 口语 char-2gram（char_chains 懒加载）
        self._char_inv_cache = None
        self.dir = ""             # dicts 目录（char_chains 找 spoken_2gram 用）

    # ---------- 持久化：用户 bigram + 预训练共现 ----------

    def load(self, dir_path):
        """加载用户二元表与预训练共现表（后者由 build_cooc.py 生成，可缺省）。"""
        self.dir = dir_path
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
        # 口语词频表（wordfreq zipf×100）：底库词频源自书面语料，口语词
        # （想吃 rank11）被书面词（消除/县城/薪酬 rank 前列）系统性压制——
        # 锚点选择按口语序重排（2026-09-05 wjtxixhcy 用例钉死）。表缺失
        # 时优雅降级为空表（锚点回退库序）。
        self.spoken = {}
        sp_path = os.path.join(dir_path, "spoken_freq.txt")
        try:
            with open(sp_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    p = line.split("\t")
                    if len(p) >= 2:
                        self.spoken[p[0]] = int(p[1])
            if self.spoken:
                self.log("[共现] 口语词频 %d 词" % len(self.spoken))
        except OSError:
            pass
        self._build_next_index()
        return n

    def _spoken(self, w):
        """口语频 zipf×100（缺失回退库 weight 的弱归一值）。"""
        z = self.spoken.get(w)
        if z:
            return z
        wt = self.de.weight(w)
        return min(500, int(math.log(max(1, wt)) * 40))  # 库频弱映射，量级对齐

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
        """用户上屏时调用：记住「上一个词 -> 这个词」的共现。

        只学中文词——英文/混合串进共现表会污染联想与上文衰减
        （主人明令：不记录英文）。
        """
        if not prev_word or not word or prev_word == word:
            return
        if not (_is_cjk(prev_word) and _is_cjk(word)):
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
        n = len(word)
        u = self.de.user_count(word)
        if w <= 0:
            # 无词频词条分两档。绝对不能让它们吃到 L 长词奖励：
            # 「不子」这类死词条（weight=0）按 UNK_LOG=1.4 计频再吃 L=12，
            # 代价 -13.4 反超真词 豹子（-21.8），霸占简拼候选前排。
            if u:
                return -UNK_LOG - U * math.log(1 + u)  # 用户自造词：使用次数当频
            return -UNK_LOG + 6.0                      # 纯垃圾占位：全场最差
        c = -A * math.log(1 + w)
        if n > 1:
            c -= L * (n - 1)
        else:
            c += P
        # 个性化只作用于多字词。单字的 user_count 来自码表固频上屏（吧=3、
        # 做=7 这类），是肌肉记忆的副产品而非词汇偏好——让它进 U 项会把
        # Viterbi 单字序列的代价压低 ~10 个 nat（吧做 -21.9→-32.1），直接
        # 碾压真词「包子」(-24.6)，这就是 2 键简拼出「吧做/被做」的根源。
        # 单字的排序由码表固频负责，统计层不再放大。
        if n > 1 and u:
            c -= U * math.log(1 + u)
        return c

    def _bi_bonus(self, prev, word):
        """上文加分：**分级回退**（用户 > 预训练词级 > 预训练字级），取第一级
        命中的证据，不叠加、也不取 max。

        为什么不取 max：字级是「字出现次数」，天然比词级高 1~2 个数量级
        （实测「了→一」字级 3403 vs 词级 382，「吃→了」字级 74 vs 词级 0）。
        取 max 会让字级永久淹没词级，中性字对（的/了/是）的噪声被放大到压过
        真正的词搭配——这就是字级一度被整体禁用的原因。

        为什么用回退：冷启动时大量真实搭配词级未收录（如「吃→了」），
        字级恰好能补上这类连接。分级回退既保住词级搭配的精确性
        （词级有证据就用词级，不让字级干扰），又让词级缺失的连接在
        冷启动时也能生效。

        优先级：用户在线学习（B，最强证据）→ 预训练词级（B_PRE）→
        预训练字级（B_PRE_CHAR，泛化兜底）。
        """
        if not prev:
            return 0.0
        c = self.bigram.get((prev, word), 0)
        if c:
            return B * math.log(1 + c)
        if self.pre_word:
            c2 = self.pre_word.get((prev, word), 0)
            if c2:
                return B_PRE * math.log(1 + c2)
        if self.pre_char and B_PRE_CHAR > 0:
            c3 = self.pre_char.get((prev[-1], word[0]), 0)
            if c3:
                return B_PRE_CHAR * math.log(1 + c3)
            # 动词词干泛化：上文是「吃了/喝了」这类 词干+助词 短语时，
            # 尾字（了）与名词无搭配，证据在首字——(吃,包)=2 这类字级对。
            c4 = self.pre_char.get((prev[0], word[0]), 0)
            if c4 and len(prev) > 1:
                return B_PRE_CHAR * math.log(1 + c4)
        return 0.0

    def _bi_bonus_multi(self, prevs, word):
        """多上文衰减打分（豆包「读光标附近整句」的统计近似）。

        prevs: [最近词, 次近词, ...]（caret_ctx.tail_words 切出）。
        第 i 个上文的 bonus 乘 DECAY^i，跨位置累加；每个位置内部仍走
        _bi_bonus 的分级回退（用户 > 预训练词级 > 预训练字级）。

        为什么需要多上文：光标前的动词往往不在紧邻位——「我吃了一个」打 bz，
        紧邻上文是「一个」（量词，对 包子/杯子/豹子 全中性），真正的裁决证据
        是两个词之外的「吃」。衰减求和让远处的动词以 1/4 强度参与，紧邻的
        强搭配仍占主导——两层证据各就各位。
        """
        if isinstance(prevs, str):
            prevs = [prevs] if prevs else []
        total = 0.0
        for i, prev in enumerate(prevs):
            if not prev:
                continue
            # 动词词干回退：上文是 吃了/追着/写过 这类 词干+助词 时，词级
            # 共现表里记的是 (吃,包子)（挖自短语 吃包子），不是 (吃了,包子)。
            # 剥掉助词再查一次，取两形中更强者——否则「我吃了一个」打 bz，
            # 紧邻的动词证据因为带了个「了」就永远查不到。
            forms = [prev]
            if len(prev) > 1 and prev[-1] in "了着过":
                forms.append(prev[:-1])
            b = max(self._bi_bonus(p, word) for p in forms)
            if b:
                total += b * (DECAY ** i)
        return total

    def _bi_bonus_span(self, prev, w):
        """Viterbi 跨度专用上文加分。

        预训练共现只助推**多字词跨度**；单字跨度只认用户在线证据。
        原因：语料虚词对（一个,不)=13 是真实文本证据，但让任意 z 声母
        单字（不|子、不|在）借位上位——首字吃了 (一个,不) 的 4.4 nat，
        后面的字靠着高频字频白嫖整条路径。词级跨度自身有 unigram 证据，
        虚词对助推整词是合理的；单字跨度则必须用户亲自教过才作数。
        """
        if len(w) == 1:
            c = self.bigram.get((prev, w), 0)
            return B * math.log(1 + c) if c else 0.0
        return self._bi_bonus(prev, w)

    def rerank(self, cands, prev_word=None):
        """对底库候选按 (unigram + 上文 + 用户习惯) 重排。

        注意：只应传入底库候选。码表（mabiao）候选是固频，不参与重排。
        prev_word 可以是单个词，也可以是 tail_words 切出的多词列表。
        """
        if not cands:
            return []
        scored = [(self._cost(w) - self._bi_bonus_multi(prev_word, w), i, w)
                  for i, w in enumerate(cands)]
        scored.sort()
        return [w for _, _, w in scored]

    def score_word(self, word, prev_word=None):
        """单个词的统计代价（越小越优），供主引擎做「词/句同池比价」。

        与 Viterbi 的整句代价同一度量体系（同参数同上下文 bonus），
        除以音节数后即可和整句的每音节均代价直接比较。
        prev_word: str 或多词 list（多上文衰减，见 _bi_bonus_multi）。
        """
        return self._cost(word) - self._bi_bonus_multi(prev_word, word)

    # ---------- 整句切分（Viterbi / beam search）----------

    def _span_cands(self, keys, mode, limit):
        """取 [i:j) 这段音节对应的词（按权重降序，去重）。

        去重：by_initial/by_pinyin 里同一词可能来自多个词库（包子 同时在
        base 和 wanxiang），重复词条白占候选名额，把有效召回深度砍半。
        ini 模式走双拼简容错口径（s/c/z 命中平/翘舌两组，见
        dict_engine.initial_span）。
        """
        if mode == "py":
            key = " ".join(keys)
            lst = self.de.by_pinyin.get(key) or ()
            out, seen = [], set()
            for _, w in lst:
                if w in seen:
                    continue
                seen.add(w)
                out.append(w)
                if len(out) >= limit:
                    break
            return tuple(out)
        return self.de.initial_span(keys, limit)

    def viterbi(self, keys, mode="ini", n=5, ret_cost=False, prev=""):
        """在音节序列上切分出最优整句，返回 n 个候选整句。

        keys: ['w','i','l','y','g','b','z']（mode="ini" 声母简拼）
              或 ['zhan','mu','si']（mode="py" 全拼音节）
        prev: **光标处上文词**（由 caret_ctx 读出）。它决定首词的 bigram 条件——
              这是「指哪打哪」的关键：同样的键串，在不同上文下应切出不同句子。
        ret_cost=True 时返回 (句子, 代价, 音节数)，供调用方把全拼/简拼两路
        候选按「每字平均代价」合并排序——两路的代价不可直接比较（词数不同），
        但每字平均代价语义一致。
        """
        keys = tuple(keys[:MAX_KEYS])
        if not keys:
            return []
        ck = (keys, mode, n, self._cache_ver, prev)
        hit = self._viterbi_cache.get(ck)
        if hit is not None:
            return hit
        out = self._viterbi_raw(keys, mode, n, ret_cost, prev)
        if len(self._viterbi_cache) > 256:
            self._viterbi_cache.clear()
        self._viterbi_cache[ck] = out
        return out

    def max_match_sentences(self, keys, mode="ini"):
        """贪心最大匹配通道：像分词器一样从前向后，每步取最长跨度键串的
        top1 词；再从后向前来一遍。产出两条「骨架句」。

        动机（2026-09-05 主人验收用例）：目标句的每一段都是词库真实词且
        排名靠前（给你 gn=7、东西 dx=3、测试一下 cyx=1、西湖醋鱼 xhcy=1），
        但统计 beam 输给「文件体现|出」类书面长词链（L 奖励+高频 unigram），
        beam256 也召不回。最大匹配不看代价只管「每个键段都用真实词填满」，
        正向产出「我今天想吃西湖醋鱼」、11 键产出「那我给你个东西测试一下」
        的完整骨架——垃圾风险（段 top1 可能是怪词）由伪似然终审兜底。

        返回 [(句子, 代价, 音节数)]，代价给固定中位值（本通道不走统计比价）。
        """
        m = len(keys)
        if m < 3:
            return []
        out, seen = [], set()

        def beam_seg(seq, bw=4):
            """分段 beam：每步 L=4..1 各取 top2 词扩展，收 bw 条完整骨架。

            2026-09-06 武媚娘传奇案：top1 贪心在「外贸年」类高权同键词上
            必岔（'w m n' 的 top1），weight 1200 的「武媚娘」永远出不了头
            ——骨架句的价值不在权重在「词库真实词填满键串」，beam 让低权
            真词路径同场竞技，排序交 LLM 终审。
            """
            finished, states = [], [([], 0)]
            while states:
                ws, i = states.pop(0)
                if i >= len(seq):
                    finished.append(ws)
                    if len(finished) >= bw:
                        break
                    continue
                ext = []
                for L in (4, 3, 2, 1):
                    if i + L > len(seq):
                        continue
                    if mode == "py":
                        lst = self.de.by_pinyin.get(" ".join(seq[i:i + L])) or ()
                        tops = [w for _, w in lst[:2]]
                    else:
                        tops = list(self.de.initial_span(seq[i:i + L], 2))
                    ext.extend((ws + [w], i + L) for w in tops)
                states.extend(ext[:bw])
                if len(states) > 16:
                    states = states[:16]
            return finished

        for rev in (False, True):
            seq = list(keys)[::-1] if rev else list(keys)
            for words in beam_seg(seq):
                if rev:
                    words = words[::-1]
                if not words:
                    continue
                s = "".join(words)
                if s in seen or len(s) < 3:
                    continue
                seen.add(s)
                # 代价=Σ词代价（与主 beam 同量级，保证池内比价不虚）；本通道的
                # 价值不在统计比价，在「产出目标骨架句」交给伪似然终审
                total = sum(self._cost(w) for w in words)
                out.append((s, total, len(s)))
        return out

    def _char_inv(self):
        """单字声母索引（小鹤键位口径），按口语频排序。懒加载缓存。

        双拼简容错：首字母键 s/c/z 的候选组并入对应翘舌组（u/i/v）——
        主人 2026-09-06 键码里 上/睡=sh 用了 s，字链通道同样要能命中。
        """
        if self._char_inv_cache is not None:
            return self._char_inv_cache
        SM2KEY = {"zh": "v", "ch": "i", "sh": "u"}
        raw = {}
        for w, py in self.de.word_py.items():
            if len(w) != 1 or not ("\u4e00" <= w <= "\u9fff"):
                continue
            parts = py.split()
            if not parts:
                continue
            k = parts[0][:2] if parts[0][:2] in ("zh", "ch", "sh") \
                else parts[0][0]
            raw.setdefault(SM2KEY.get(k, k), set()).add(w)
        inv = {k: sorted(v, key=lambda w: -self._spoken(w))[:10]
               for k, v in raw.items()}
        for plain, canon in (("s", "u"), ("c", "i"), ("z", "v")):
            pool = raw.get(plain, set()) | raw.get(canon, set())
            if pool:
                inv[plain] = sorted(pool, key=lambda w: -self._spoken(w))[:10]
        self._char_inv_cache = inv
        return self._char_inv_cache

    def char_chains(self, keys, mode="ini", n=6, beam=32, prev=""):
        """口语字链召回：每键位连一个汉字，口语 unigram+char-2gram 打分，
        束搜索吐 n 条整链。

        动机（2026-09-05 回归案 wilygbz 钉死）：词库与口语 2gram 里
        吃了(7747)/我吃(3782)/了个(8389) 全都在，但 viterbi 的书面代价、
        锚点的书面词频门槛、max_match 的「外出旅游」贪心，三路都召不回
        「我吃了一个包子」。真人打真实句子时**逐键取字的字链就是目标句
        本身**（nwgngdxcuyx 逐位=那我给你个东西测试一下 11 字全对）——
        本通道不与统计比价，专管把字链塞进池，排序交 LLM 终审。

        打分：-log10(口语频+1) 逐字累加 + 字对 bigram 未见罚（口语 2gram
        表 21 万对，吃接了/个接一 这类接对是强证据）。垃圾链（每个键都
        取到同音高频字但不成句）由终审降权，召回侧宁滥勿缺。
        """
        m = len(keys)
        if m < 4 or m > MAX_KEYS:
            return []
        if self.sp2 is None:
            self.sp2 = {}
            p2 = os.path.join(self.dir, "spoken_2gram.txt")
            if os.path.isfile(p2):
                with open(p2, "r", encoding="utf-8") as f:
                    for line in f:
                        p = line.split()
                        if len(p) >= 2:
                            try:
                                self.sp2[p[0]] = int(p[1])
                            except ValueError:
                                pass
        inv = self._char_inv()
        states = [(0.0, "", "")]
        for k in keys:
            cands = inv.get(k, [])
            if not cands:
                return []
            nxt = []
            for sc, s, last in states:
                for c in cands:
                    # unigram：-log10(口语频)，缺表字回退库频，双缺重罚
                    f = self.spoken.get(c) or 0
                    if f:
                        cu = -math.log10(f)
                    else:
                        wt = self.de.weight(c)
                        cu = -math.log10(wt) if wt else 3.0
                    # bigram：口语 2gram 全权重（-log10 域，与 unigram 同
                    # 量纲直接可比——我吃 3782→-3.6 vs 缺证 +0.5，差距
                    # 足以压过「出/就/没」类高频字的 unigram 先验）
                    if last:
                        t = self.sp2.get(last + c) or 0
                        cb = -math.log10(t + 1) if t else 0.5
                    else:
                        cb = 0.0
                    nxt.append((sc + cu + cb, s + c, c))
            nxt.sort(key=lambda t: t[0])
            seen, pruned = set(), []
            for sc, s, c in nxt:
                if s in seen:
                    continue
                seen.add(s)
                pruned.append((sc, s, c))
                if len(pruned) >= beam:
                    break
            states = pruned
        out, seen_s = [], set()
        for sc, s, _ in states:
            if s in seen_s or len(s) < m:
                continue
            seen_s.add(s)
            out.append((s, sc, m))
            if len(out) >= n:
                break
        return out

    def anchor_sentences(self, keys, mode="ini", n=32, prev=""):
        """锚点串接召回：强锚（长跨度精准命中 / 双声母高频 rank≤2）按键序
        强制串成整句，缝隙用 viterbi top1 填充。

        背景（2026-09-05 主人验收用例钉死）：目标句「那我给你个东西测试
        一下」每段都是真实词且靠前（东西 dx rank2、测试一下 cuyx rank1），
        但统计 beam 里「的|乡村|是一项」类高频短词链每字便宜 ~2 nat，
        beam256 也召不回；旧单锚版的前缀变体同样来自统计 viterbi top3，
        「东西」在 dx 段 rank2 永远进不了前缀——目标句整句缺席。串接
        思路：不做全句统计比价（这正是口语串的死因），把互不重叠的强锚
        按键序首尾相接，锚间缝隙由 viterbi 最优路径填充——
        「那我给你个」(前缀top1) +「东西」+「测试一下」= 目标句整句进池，
        排序交给神经终审（口语通顺度完爆书面链，-6.56 vs -7.78 实测）。

        锚点判定：
        - 4 音节键串 top1 必锚（键串特异性高，西湖醋鱼/测试一下）。
        - 2 音节键串**去重后** rank≤3 且词频 ≥ ANCHOR2_FREQ 才锚（东西
          50万 ✓、想吃 5.8万 ✓、年味 数百 ✗）。rank 必须按去重后序数：
          by_initial 含重复词条（xi 原始 top5 是 形成/宣传/新车/形成/宣传），
          按原始索引扫会漏真锚（实测钉死）。
        - 3 音节键串一律不锚：候选 ≤8 的门槛挡不住「小吃小喝/通讯程序/
          给你惯的」类低特异性怪词——它们全是垃圾句源（2026-09-05 追踪
          实测），span4/span2 已足够覆盖真锚。

        核心轮询顺序按「跨度长→词频高」而非键序：span4 真锚（西湖醋鱼/
        测试一下）必须先于垃圾 span2 锚（内外/那位/危机/文件）被轮到。

        右侧串接取**就近优先**（起点最小，同起点按质量序）：9 键用例里
        核心=今天 时 pos=3 处的「想吃」若被全局质量序更前的西湖醋鱼
        跳过，缝隙 viterbi(xi) 只能填出「形成」——就近串接才保住
        「今天+想吃+西湖醋鱼」的骨架（2026-09-05 追踪实测钉死）。

        返回 [(句子, 代价, 音节数)]，代价=粗拼接值（不参与最终排序，
        终审由整句伪似然负责）。
        """
        keys = tuple(keys[:MAX_KEYS])
        m = len(keys)
        if m < 4:
            return []
        # ---- 收集强锚 (i, j, word) ----
        cores = []
        for i in range(m):
            for L in (2, 4):
                if i + L > m:
                    continue
                if mode == "py":
                    lst = self.de.by_pinyin.get(" ".join(keys[i:i + L])) or ()
                    top_words, seen_w = [], set()
                    for _, w in lst:
                        if w not in seen_w:
                            seen_w.add(w)
                            top_words.append(w)
                        if len(top_words) >= 48:
                            break
                else:
                    # 双拼简容错口径（s/c/z 命中平/翘舌两组），已按权重
                    # 降序去重——首字母键码也能锚到 上/睡/传 类翘舌词
                    top_words = list(self.de.initial_span(keys[i:i + L], 48))
                if not top_words:
                    continue
                if L == 4:
                    # span4 也设词频门槛：4 音节键串候选稀少，top1 常是
                    # 「小吃小喝(1889)/通讯程序(1725)/误尽天下(1006)」类
                    # 低频怪词——它们占锚产出纯垃圾句；真锚西湖醋鱼(3607)/
                    # 测试一下(14305)/那个东西(9620) 全部过线（2026-09-05 实测）
                    if self.de.weight(top_words[0]) >= ANCHOR2_FREQ:
                        cores.append((i, i + L, top_words[0]))
                else:
                    # 口语序去重取前 6：底库 rank 是书面语料的序（xi 组
                    # 想吃 rank11 被消除/县城/薪酬压制），口语 zipf 序里
                    # 想吃升到第 5——按口语频排序后再取（2026-09-05 实测）
                    # 准入双通道（2026-09-06 wztwsmsh 案）：书面 weight≥3000
                    # 或口语频≥480——「没睡/丢下」类口语词书面权重不足 3000
                    # 被旧门槛挡在核外，锚点串接只剩「没说」类书面同键词
                    # 口语序→掺权序（weight 主导 + 口语加权）：2gram 碎片
                    # 词（到现/得像）靠字对频次在纯口语序里压真词（丢下），
                    # 掺权后碎片(weight<1000)沉底（2026-09-06 bydxwhm 案）
                    uniq = top_words[:32]
                    uniq.sort(key=lambda w: -self.de._spoken_rank(
                        w, self.de.weight(w)))
                    for w in uniq[:10]:
                        if self.de.weight(w) >= ANCHOR2_FREQ \
                                or self._spoken(w) >= 480:
                            cores.append((i, i + L, w))
        if not cores:
            return []
        # 长锚优先、同长口语频优先——真锚先被轮询（东西 zipf>大学/大型，
        # 想吃 zipf 升 xi 组第 5，见 docstring 与 _spoken 注释）
        cores.sort(key=lambda a: (-(a[1] - a[0]), -self._spoken(a[2]), a[0]))
        # ---- 每个锚为核心：前缀 viterbi top1 + 本锚 + 右侧就近串接 ----
        out, seen_sent, seen_core = [], set(), set()
        for (ai, aj, aw) in cores:
            if aw in seen_core:
                continue
            seen_core.add(aw)
            pres = self.viterbi(keys[:ai], mode, 1, ret_cost=True, prev=prev) \
                if ai else [("", 0.0, 0)]
            if not pres:
                continue
            # 同前缀去重已废除（2026-09-06 bydxwhm 案钉死）：旧规则同前缀
            # 只留口语频最高核——「不要」前缀下 东西(38127) 霸位，丢下
            # (2517) 永远进不了串接。裁判时代垃圾句按分数沉底不再挤排名，
            # 池名额由 n 封顶，每个过门槛的核都给串接机会。
            # 词链（不拼串）：同起点并列的选择统计信号全线失效——weight 压
            # （想吃 rank11）、衔接字级共现噪声压（(天,宣)=31「今天宣传」、
            # (天,选)「天选」压 (天,想)=6）。**只有神经终审能裁**：
            # 高口语频核心（zipf≥5.4，今天/东西类）的同起点锚**全部展开**
            # 进池（8 句里多句垃圾可接受）；普通核心维持双支控池规模
            # （2026-09-05 实测钉死）。
            head = ([pres[0][0]] if pres[0][0] else []) + [aw]
            total = pres[0][1] + self._cost(aw) - LONG_SPAN_BOOST * 0.5
            # hot 口径=口语频次绝对值（v3 表是真实频次）：5000 次以上才是
            # 今天(61308)/东西(38127) 级高频核心；阈值 540（旧 zipf 口径）
            # 会让「集团/我就/我叫」类中等词也全展开，垃圾句挤爆候选池
            hot_core = self._spoken(aw) >= 5000
            fork_cap = 8 if hot_core else 2
            made_cap = fork_cap
            # FIFO 队列：先分叉的链先出句——LIFO 会让 branch-of-branch
            # 抢在前头耗尽每核心配额，正主（想吃支）饿死（实测钉死）
            queue = [(aj, head, fork_cap)]
            made = 0
            while queue and made < made_cap and len(out) < n:
                pos, words, forks = queue.pop(0)
                while True:
                    ahead = [a for a in cores if a[0] >= pos]
                    if not ahead:
                        # 尾缝 top2：单键尾字歧义（h→和/好）交 LLM 终审，
                        # 召回侧两种都给（2026-09-06 wztwsmsh 案）
                        gap = self.viterbi(keys[pos:], mode, 2)
                        for g in (gap or [])[:2]:
                            if not g:
                                continue
                            sent = "".join(words + [g[0]])
                            if sent not in seen_sent:
                                seen_sent.add(sent)
                                out.append((sent, total, m))
                                made += 1
                        break
                    nmin = min(a[0] for a in ahead)  # 就近优先
                    same = [a for a in ahead if a[0] == nmin]
                    last = words[-1] if words else ""
                    s4 = [a for a in same if a[1] - a[0] >= 3]
                    s4.sort(key=lambda a: -self.de.weight(a[2]))
                    r2 = [a for a in same if a[1] - a[0] < 3]
                    r2.sort(key=lambda a: (-self._bi_bonus_span(last, a[2]),
                                           -self.de.weight(a[2])))
                    main = s4[0] if s4 else (r2[0] if r2 else None)
                    if main is None:
                        break

                    def _adv(nxt, wds, fk, cur_pos=pos):
                        gw = []
                        if nxt[0] > cur_pos:
                            gap = self.viterbi(keys[cur_pos:nxt[0]], mode, 1)
                            if gap and gap[0]:
                                gw = [gap[0]]
                        return (nxt[1], wds + gw + [nxt[2]], fk)

                    branch = None
                    if forks > 0:
                        if hot_core and s4:
                            # 高频核心：span4 主链之外，span2 组也全展开
                            branches = [a for a in r2 if a is not main][:forks]
                        else:
                            alt = r2 if s4 else r2[1:]
                            # 双支：口语组合词（想吃 3509 vs 想出 3523）频次
                            # 差距可小到 14 次，单支必被同衔接对手抢掉
                            branches = alt[:2]
                        if branches:
                            branch = branches
                    if branch:
                        forks -= 1
                        for b in branch:
                            queue.append(_adv(b, words, forks))
                    # 缝隙次优变体（2026-09-06 wztwsmsh 案）：中缝单键的
                    # viterbi top1 被书面搭配（没说/和）压住（没睡/好），
                    # top2 入队交 LLM 终审裁决，召回侧两种都给
                    if main[0] > pos:
                        gap2 = self.viterbi(keys[pos:main[0]], mode, 2)
                        if len(gap2) > 1 and gap2[1] and gap2[1][0] \
                                and (not gap2[0] or gap2[1][0] != gap2[0][0]):
                            queue.append((main[1],
                                          words + [gap2[1][0], main[2]], 0))
                    pos, words, forks = _adv(main, words, forks)
                    # 主链原地继续推进；副链在栈中稍后处理
            if len(out) >= n:
                break
        return out

    def _viterbi_raw(self, keys, mode, n, ret_cost=False, prev=""):
        m = len(keys)
        # dp[j] = [(cost, words_tuple, last_word), ...] 保留 BEAM 条最优
        # dp[0] 的 last_word 设为光标处上文词（caret_ctx 读出），
        # 这样首词能吃到 _bi_bonus(prev, w)——上文条件由此进入整句预测。
        # 跨度代价预计算：_cost(w) 只依赖 w，不依赖 dp 状态。SPAN_CANDS=48
        # 后内层循环约 64 跨度 x 48 词 x 6 状态，不预缓存的话每次击键要
        # 重复算 ~1.8 万次 _cost，纯浪费。
        span_cost = {}

        def cost_of(w):
            c = span_cost.get(w)
            if c is None:
                c = span_cost[w] = self._cost(w)
            return c

        dp = [None] * (m + 1)
        dp[0] = [(0.0, (), prev)]
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
                # 长跨度精准命中奖励：3~4 音节键串的 top3 命中是强信号——
                # 键串特异性高（x h c y 全库仅 5 个候选，西湖醋鱼 rank1），
                # 用户打出这种串就是冲着这个词来的。低频长词的 unigram 打
                # 不过高频单字链（西湖+醋+鱼 -52 vs 西湖醋鱼 -44.5），不
                # boost 的话口语专名串永远在 beam 外（2026-09-05 主人
                # wjtxixhcy 用例）。2 音节键串候选数百计，rank3 证据弱，
                # 不 boost。
                boost = 0.0
                span_len = j - i
                if span_len >= 3:
                    key_str = " ".join(keys[i:j])
                    lst = self.de.by_pinyin.get(key_str) if mode == "py" \
                        else self.de.by_initial.get(key_str)
                    if lst:
                        top3 = {w for _, w in lst[:3]}
                else:
                    top3 = ()
                for w in words_here:
                    cw = cost_of(w) - (LONG_SPAN_BOOST if w in top3 else 0.0)
                    for c0, seq, prev in prev_states:
                        cand.append((c0 + cw - self._bi_bonus_span(prev, w), seq + (w,), w))
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
            # 首词单字保留通道：beam 上半留全局最优，下半优先补「首词是单字」
            # 的路径。整句的第一个词经常是 我/你/还/吃 这类单字，而 2 字词
            # （成立 -25.9）在偶数位上系统性比 1+1 单字对（吃|了 -21.4）便宜
            # ~2-4 nat——纯全局 top-K 会把单字开头的路径在 j=2 全灭，可它们
            # 的共现红利（(了,一个)=395）要到 j=4 才兑现（ilygbz 案）。
            half = max(1, BEAM // 2)
            kept = cand[:half]
            if len(kept) < BEAM:
                extra = [t for t in cand[half:] if t[1] and len(t[1][0]) == 1]
                kept += extra[: BEAM - len(kept)]
            dp[j] = kept
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
