# -*- coding: utf-8 -*-
"""神经判别重排器 —— RBT3(3层RoBERTa-wwm中文) ONNX 推理，豆包第二层的正式形态。

原理
----
RBT3 是中文掩码语言模型（MLM）。把「上文 + 候选词」一起喂进去，读 MLM 头在
候选各字位置上的 log-softmax，得到 **P(候选词 | 上文)** 的对数似然——
一次前向、不生成 token、天然判别式。与统计共现表的本质区别：
    共现表只能记「语料里出现过 打碎+杯子 没有」——没出现过就无证据；
    MLM 学过整个语料的语义——「打碎」「追」的宾语语义约束（易碎物/动物）
    是参数里的知识，**能泛化到没见过的搭配**。

工程
----
- 批量：top-N 候选拼成一个 batch，一次前向全出（GPU ~10-30ms / CPU int8 ~40-80ms）。
- 异步精排：击键仍由统计层即时响应（<13ms），神经分 ~30-100ms 后到达，
  经既有队列刷新候选顺序（豆包「候选自我修正」的体验形态）。
- 运行时：onnxruntime-directml（RTX 2060 走 DML，无需 cuDNN），CPU EP 兜底。
- tokenizer：内置最小 WordPiece（只依赖 vocab.txt），运行时零 transformers 依赖。

模型获取（防丢哲学：能再生）
---------------------------
ai_neural/fetch_model.py 从 hf-mirror.com 重新下载，模型文件不入库（.gitignore）。
"""

import os
import queue
import threading
import time

import numpy as np


class WordPiece:
    """最小中文 BERT 分词器：CJK 逐字、ASCII 词片贪心最长匹配。"""

    def __init__(self, vocab_path):
        self.vocab = {}
        with open(vocab_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                self.vocab[line.rstrip("\n")] = i
        self.cls = self.vocab.get("[CLS]", 101)
        self.sep = self.vocab.get("[SEP]", 102)
        self.unk = self.vocab.get("[UNK]", 100)
        self.pad = self.vocab.get("[PAD]", 0)

    def _pieces(self, word):
        """对一个连续片段做 WordPiece 贪心（##前缀续片）。"""
        out = []
        start = 0
        while start < len(word):
            end = len(word)
            piece = None
            while start < end:
                sub = word[start:end]
                if start > 0:
                    sub = "##" + sub
                if sub in self.vocab:
                    piece = sub
                    break
                end -= 1
            if piece is None:
                return out + [self.unk]
            out.append(piece)
            start = end
        return out

    def encode(self, text):
        """文本 -> token id 列表（不含 [CLS]/[SEP]，由调用方拼接）。"""
        ids = []
        buf = ""  # 连续 ASCII 片段缓存
        for ch in text:
            if ch == " " or ch == "\t":
                if buf:
                    ids.extend(self.vocab.get(p, self.unk) for p in self._pieces(buf))
                    buf = ""
                continue
            if ord(ch) < 128 and ch.isalnum():
                buf += ch.lower()
                continue
            if buf:
                ids.extend(self.vocab.get(p, self.unk) for p in self._pieces(buf))
                buf = ""
            # CJK/标点：全角与常见中文标点在词表里逐字收录；没有就 [UNK]
            ids.append(self.vocab.get(ch, self.unk))
        if buf:
            ids.extend(self.vocab.get(p, self.unk) for p in self._pieces(buf))
        return ids


class NeuralReranker:
    """RBT3 ONNX 批量打分器 + 异步请求队列（与 AIEngine 同构的线程模型）。"""

    def __init__(self, model_dir, log=print):
        self.dir = model_dir
        self.log = log
        self.ready = False
        self.tok = None
        self.sess = None
        self.provider = ""
        self.on_result = None          # 结果回调（由引擎注入）
        self.on_ready = None           # 就绪回调（引擎用来升级在屏旧码）
        self._q = queue.Queue()
        self._load_ms = 0

    # ---------- 加载 ----------

    def load_async(self):
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        t0 = time.perf_counter()
        try:
            import onnxruntime as ort
            path = os.path.join(self.dir, "model_int8.onnx")
            fp16 = os.path.join(self.dir, "model_fp16.onnx")
            if os.path.isfile(fp16):
                path = fp16  # fp16 优先（DML 走 GPU 更快；CPU 跑 fp16 会自动转 fp32）
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            # 禁用图优化：onnxruntime 1.24 的 SimplifiedLayerNormFusion 在
            # RBT3 fp16 模型上初始化即崩（InsertedPrecisionFreeCast 找不到
            # 节点参数，2026-09-06 实测钉死）。代价是 CPU 推理慢一点，
            # RBT3 三层小模型可接受。
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            try:
                self.sess = ort.InferenceSession(
                    path, opts,
                    providers=["DmlExecutionProvider", "CPUExecutionProvider"])
            except Exception:
                self.sess = ort.InferenceSession(path, opts,
                                                 providers=["CPUExecutionProvider"])
            self.provider = self.sess.get_providers()[0]
            self.tok = WordPiece(os.path.join(self.dir, "vocab.txt"))
            self.score("预热", ["词汇"])  # DML 首次调用要编译着色器(~1s)，在加载期付掉
            self.ready = True
            self._load_ms = round((time.perf_counter() - t0) * 1000)
            self.log("[神经] RBT3 就绪 provider=%s 加载%dms" % (self.provider, self._load_ms))
            threading.Thread(target=self._worker, daemon=True).start()
            if self.on_ready:
                try:
                    self.on_ready()
                except Exception:
                    pass
        except Exception as e:
            self.log("[神经] 加载失败（统计层不受影响）: %r" % e)

    # ---------- 打分 ----------

    def score(self, ctx_text, cands, max_cand=16, max_ctx=16, chunk=32):
        """批量打分：返回 {词: 每字平均logP}（越大=上文条件下越自然）。

        伪似然（Wang & Cho 式）：候选的每个字轮流 [MASK]，条件=上文+候选其余
        字，读被掩字的 log-softmax。为什么不能全喂直接读：双向 MLM 会看见
        候选的全部字，常见字对（班↔子）互相预测导致分数虚高且与语境无关
        （实测 班子 在 吃/打碎/追上 三种上文下都是 -1.6~3.2，判别力归零）。

        ctx_text: 上文字符串（直接截尾）
        cands: 候选列表。max_cand=16：不只词——**整句候选（5~12 字）也在
        池里**（viterbi 产出），必须整句进模型。实测整句级裁决区分度极强：
        「那我给你个东西测试一下」(-6.56) 甩开「年我国能够的乡村是一些」
        (-7.78) 一个身位——口语通顺度是 MLM 参数里的知识，统计共现表
        （SIGHAN 新闻语料）里根本没有这些口语接龙对。分块前向
        （chunk=32，每块行数=Σ候选字数）：logits 张量 [B,L,21128]，
        分块把峰值显存压住。
        """
        if not self.ready or not cands:
            return {}
        ctx = ctx_text[-max_ctx:]
        mask_id = self.tok.vocab.get("[MASK]", 103)
        rows, read_ids, owners = [], [], []
        for w in cands:
            a = self.tok.encode(ctx)[-max_ctx:]
            b = self.tok.encode(w)[:max_cand]
            if not b:
                continue
            base_row = [self.tok.cls] + a + [self.tok.sep] + b + [self.tok.sep]
            off = 1 + len(a) + 1
            for i in range(len(b)):
                row = list(base_row)
                row[off + i] = mask_id
                rows.append(row)
                read_ids.append(b[i])   # 读被掩的原字，不是 [MASK] 自己
                owners.append(w)
        out = {}
        # 定长填充：导出模型对 attention_mask 的执行不一定严格（尤其 DML），
        # 变长批会让 [PAD] 泄漏进注意力、同一词在不同批组成下得分漂移。
        # 全部 pad 到固定 SEQ_LEN，批内批间行为一致（实测 drift=0.000；
        # 代价是每行多算几步，可接受）。SEQ_LEN=40：16 字候选整句
        # （CLS+16ctx+SEP+16+SEP=34）也装得下。
        SEQ_LEN = 40
        for lo in range(0, len(rows), chunk):
            part = rows[lo:lo + chunk]
            ml = SEQ_LEN
            B = len(part)
            input_ids = np.zeros((B, ml), dtype=np.int64)
            attn = np.zeros((B, ml), dtype=np.int64)
            ttype = np.zeros((B, ml), dtype=np.int64)
            for i, r in enumerate(part):
                input_ids[i, :len(r)] = r
                attn[i, :len(r)] = 1
            feed = {"input_ids": input_ids, "attention_mask": attn,
                    "token_type_ids": ttype}
            logits = self.sess.run(None, feed)[0]  # [B, L, V]
            for i in range(B):
                w = owners[lo + i]
                pos = part[i].index(mask_id)  # 每行唯一的被掩位
                row = logits[i, pos].astype(np.float64)
                row -= row.max()
                lse = np.log(np.exp(row).sum())
                out[w] = out.get(w, 0.0) + (row[read_ids[lo + i]] - lse)
        for w in out:
            out[w] /= max(1, len(self.tok.encode(w)[:max_cand]))
        return out

    # ---------- 异步请求（与 AIEngine 同构） ----------

    def request(self, code, ctx_text, cands):
        if self.ready and cands:
            self._q.put((code, ctx_text, cands))

    def _worker(self):
        while True:
            code, ctx_text, cands = self._q.get()
            # 只处理最新请求：连击时中间码的精排已经没意义，排队反而积压
            try:
                while True:
                    code, ctx_text, cands = self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                t0 = time.perf_counter()
                scores = self.score(ctx_text, cands)
                ms = round((time.perf_counter() - t0) * 1000)
                if self.on_result:
                    self.on_result(code, scores, ms)
            except Exception as e:
                self.log("[神经] 打分异常已兜底: %r" % e)


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    nr = NeuralReranker(os.path.join(base, "ai_neural", "rbt3"))
    loaded = threading.Event()
    orig = nr._load

    def _load2():
        orig()
        if nr.ready:
            loaded.set()
    nr._load = _load2
    nr.load_async()
    loaded.wait(60)
    if not nr.ready:
        print("加载失败")
        raise SystemExit(1)

    # 三场景验证：MLM 语义迁移能力
    cases = [
        ("我吃了一个", ["包子", "杯子", "豹子", "不足", "不再"]),
        ("我打碎了一个", ["杯子", "包子", "豹子", "盘子", "碗"]),
        ("我追上了一个", ["豹子", "杯子", "包子", "小偷", "球"]),
    ]
    for ctx, cands in cases:
        t0 = time.perf_counter()
        s = nr.score(ctx, cands)
        ms = (time.perf_counter() - t0) * 1000
        rank = sorted(cands, key=lambda w: -s.get(w, -99))
        print("%-10s %6.1fms  %s" % (ctx, ms, "  ".join("%s:%.2f" % (w, s[w]) for w in rank)))
