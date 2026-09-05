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
        """
        key = " ".join(keys)
        d = self.de.by_pinyin if mode == "py" else self.de.by_initial
        lst = d.get(key)
        if not lst:
            return ()
        out, seen = [], set()
        for _, w in lst:
            if w in seen:
                continue
            seen.add(w)
            out.append(w)
            if len(out) >= limit:
                break
        return tuple(out)

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

    def anchor_sentences(self, keys, mode="ini", n=6, prev=""):
        """长词锚定召回：3~4 音节键串的 top1 命中词，强制构造整句进池。

        背景（2026-09-05 主人 wjtxixhcy 用例）：主 beam 的统计代价系统性
        偏向「新车型|后槽牙」类词典可命中长词链——每字 -15.6，把「我今天
        想吃西湖醋鱼」这类口语串全程剪枝；给长词加 boost 又引发军备竞赛
        （3 音节键串候选少，top3 门槛形同虚设）。锚定通道绕开零和竞争：
        对每个 3~4 音节跨度，取该键串的 top1 词做锚，前缀/后缀用常规
        viterbi 最优路径填充，整句直进候选池——检索便宜、裁决全面，排序
        交给神经层（「我今天想吃西湖醋鱼」的 MLM 通顺度完爆荒谬长词链）。

        返回 [(句子, 代价, 音节数)]，代价是粗粒度拼接值（接缝 bigram 不算），
        只用于与整句候选按每字均价比价，精度要求不高。
        """
        m = len(keys)
        if m < 4:
            return []
        out, seen_anchor, seen_sent = [], set(), set()
        for i in range(0, m - 2):
            for j in (i + 3, i + 4):
                if j > m:
                    continue
                key_str = " ".join(keys[i:j])
                lst = self.de.by_pinyin.get(key_str) if mode == "py" \
                    else self.de.by_initial.get(key_str)
                if not lst:
                    continue
                w = lst[0][1]
                if w in seen_anchor:
                    continue
                # 锚点门槛：4 字词必锚（4 音节键串候选天然稀少，rank1 就是
                # 用户想要的）；3 字词只在该键串候选极少数时锚——否则「和
                # 参与/小吃下」这类 3 字词占满锚点，真锚（西湖醋鱼）轮不到。
                if len(w) < 4 and len(lst) > 8:
                    continue
                seen_anchor.add(w)
                pres = self.viterbi(keys[:i], mode, 3, ret_cost=True,
                                    prev=prev) if i else [("", 0.0, 0)]
                sufs = self.viterbi(keys[j:], mode, 1, ret_cost=True) if j < m \
                    else [("", 0.0, 0)]
                if not pres or not sufs:
                    continue
                ss, sc, _ = sufs[0]
                # 前缀出 top3 变体：前缀的统计代价被「文件体现」类 4 字词条
                # （L×3 奖励）垄断，「我今天想吃」这类口语前缀排不进 top1，
                # 但它的 MLM 通顺度完爆——多路进池，排序交给神经裁决。
                for ps, pc, _ in pres:
                    sent = ps + w + ss
                    if sent in seen_sent:
                        continue
                    seen_sent.add(sent)
                    total = pc + self._cost(w) - LONG_SPAN_BOOST * 0.5 + sc
                    out.append((sent, total, m))
                if len(out) >= n:
                    return out
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
