# -*- coding: utf-8 -*-
"""LLM 终审裁判 —— Qwen2.5-0.5B 整句判别打分（豆包第二层的换代形态）。

为什么换代（2026-09-05）
------------------------
RBT3 的新闻语料偏差是实测天花板：「文件体现出西湖醋鱼」(-7.86) 压过
「我今天想吃西湖醋鱼」(-9.21)——MLM 在书面语料上训练，口语串被判死。
Qwen2.5-0.5B 在口语/对话语料上预训练过，整句判别区分度实测极强：
    那我给你个东西测试一下 -5.61  >>  那我给你过渡性测试一下 -6.46
    我今天想吃西湖醋鱼   -3.52  >>  我今天想出西湖醋鱼   -5.34

「生成」与「判别」的分界（2026-09-05 实测钉死）
------------------------------------------------
- 生成（约束束搜索）：0.5B 五种 prompt 全试（few-shot/纯续写/口语引子/
  chat 模板），束搜索只能产出「局部通顺」的串（文末有小/那我给那句…），
  距离首选要求差一个模型代际——**0.5B 生成不行**。
- 判别（教师强制整句 logP）：一次前向出全句分数，无自回归漂移，
  口语句完胜书面腔——**0.5B 判别行**。主人的「直接问 LLM 哪个最有可能」
  在 0.5B 上的可行形态就是判别式打分。

工程
----
- 打分：P(候选句 | 口语引子) 逐字 log-softmax 累加，按字长归一。引子把
  续写分布校准到口语域（裸 <|endoftext|> 后 0.5B 会续出代码注释体，
  实测 -3.52 vs -33 的病根之一）。
- 批量：右填充 batch 前向（bs=8），~0.8s/8句 CPU fp32。异步精排，
  击键不被阻塞（与 NeuralReranker 同构的线程模型）。
- 注意：tok.eos_token_id 是 <|im_end|>(151645)，引子首 token 必须用
  <|endoftext|>(151643)——im_end 后 0.5B 续出 /API/QueryString 类代码体。

模型获取（防丢哲学：能再生）
---------------------------
ai_llm/fetch_qwen.py 从 hf-mirror.com 重新下载，模型文件不入库（.gitignore）。
"""

import os
import queue
import threading
import time

# 口语引子：不含任何验收用例句（防泄漏），只负责把分布校准到口语域
PRIMER = ("<|endoftext|>我们明天去公园吧。你吃饭了吗？这个东西多少钱？"
          "我不知道他在哪里。一起去看电影吧。今天天气真好。")


class QwenJudge:
    """Qwen2.5-0.5B 整句判别打分器 + 异步请求队列（NeuralReranker 孪生接口）。"""

    def __init__(self, model_dir, log=print, max_cand=72, bs=8):
        self.dir = model_dir
        self.log = log
        self.ready = False
        self.on_result = None          # 结果回调（由引擎注入）
        self.on_ready = None           # 就绪回调（引擎用来升级在屏旧码）
        self.max_cand = max_cand       # 单次打分候选上限（CPU 耗时护栏）
        self.bs = bs
        self._q = queue.Queue()
        self._tok = None
        self._model = None
        self._p_ids = None
        self._load_ms = 0
        self._cache = {}               # (ctx,句)→logP：同码重打零耗时

    # ---------- 加载 ----------

    def load_async(self):
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        t0 = time.perf_counter()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            torch.set_grad_enabled(False)
            # GPU 优先（2026-09-07 主人判死 CPU 速度：「不可能等一个候选六
            # 秒」——CPU int8 仍 75ms/句，fp16 GPU ~20ms/批，全池 66 句一遍
            # ~0.3s，豆包级）。无 CUDA 再落 CPU int8。
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self._tok = AutoTokenizer.from_pretrained(self.dir,
                                                      trust_remote_code=True)
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.dir, trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device == "cuda"
                    else torch.float32)
                # ⚠️ CPU 严禁 int8 动态量化（2026-09-07 实测钉死）：量化把
                # logP 动态范围压塌（包子 -5.45→-6.50，被子 -11.24→-7.04，
                # 判别分差 5.8nat→0.5nat），B 案包子从第 1 掉到第 2——
                # 判别裁判的价值全在跨数量级的区分度上。CPU 回退保持
                # fp32，靠收紧送裁数量（max_cand 72→20）补速度。
                self._model.eval().to(self.device)
            except Exception as e:
                if self.device != "cuda":
                    raise
                self.log("[LLM裁判] CUDA 加载失败（%r），回退 CPU fp32" % e)
                self.device = "cpu"
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.dir, trust_remote_code=True, torch_dtype=torch.float32)
                self._model.eval()
            if self.device == "cpu":
                self.max_cand = 20  # fp32 0.35s/句 × 20 句 ≈ 7s，保区分度
            self.bs = 16 if self.device == "cuda" else 8
            self._p_ids = self._tok.encode(PRIMER, add_special_tokens=False)
            self.score(["预热一句"])  # 首次前向的初始化开销在加载期付掉
            self.ready = True
            self._load_ms = round((time.perf_counter() - t0) * 1000)
            self.log("[LLM裁判] Qwen2.5-0.5B 就绪 加载%dms" % self._load_ms)
            threading.Thread(target=self._worker, daemon=True).start()
            if self.on_ready:
                try:
                    self.on_ready()
                except Exception:
                    pass
        except Exception as e:
            self.log("[LLM裁判] 加载失败（回退 RBT3/统计层）: %r" % e)

    # ---------- 打分 ----------

    def score(self, cands, ctx_text=""):
        """批量整句打分：返回 {句: 每字平均logP}（越大=口语域越自然）。

        教师强制：一次前向读候选逐字 log-softmax，无自回归漂移。
        ctx_text：上文尾串（如「诗仙」「我今天吃了一个」）——2026-09-06
        主人样本钉死：无条件下 2 字词判别失效（诗仙+lb 里 来吧 -4.94
        压过 李白，接龙专名全灭）；条件化后 诗圣+df 里 杜甫 -7.56
        完胜 对方 -9.51。logP 只数候选自身 token，上文只是条件。
        """
        if not self.ready or not cands:
            return {}
        ctx_key = (ctx_text or "")[-16:]
        cache = self._cache
        out = {c: cache[(ctx_key, c)] for c in cands if (ctx_key, c) in cache}
        miss = [c for c in cands if (ctx_key, c) not in cache]
        if not miss:
            return out
        tok, model, p_ids = self._tok, self._model, self._p_ids
        import torch
        dev = self.device
        ctx_ids = tok.encode(ctx_key, add_special_tokens=False) if ctx_key else []
        p0 = len(p_ids) + len(ctx_ids)
        seqs = [p_ids + ctx_ids + tok.encode(s, add_special_tokens=False)
                for s in miss]
        pad = tok.pad_token_id or tok.eos_token_id
        for lo in range(0, len(seqs), self.bs):
            chunk = seqs[lo:lo + self.bs]
            L = max(len(s) for s in chunk)
            ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for r, s in enumerate(chunk):
                ids[r, :len(s)] = torch.tensor(s)
                att[r, :len(s)] = 1
            logits = model(ids.to(dev), attention_mask=att.to(dev)).logits
            for r, s in enumerate(chunk):
                nc = len(s) - p0
                if nc <= 0:
                    continue
                # 只在候选 token 的预测位取 log_softmax（fp16 logits 先升
                # fp32 保精度）；旧写法把整条序列在 15 万词表上全展开
                pos = torch.arange(p0 - 1, p0 - 1 + nc, device=dev)
                tgt = torch.tensor(s[p0:], dtype=torch.long, device=dev)
                lp = torch.log_softmax(logits[r, pos].float(), -1)
                v = lp[torch.arange(nc, device=dev), tgt].sum().item() / nc
                out[miss[lo + r]] = v
                cache[(ctx_key, miss[lo + r])] = v
        if len(cache) > 8192:
            cache.clear()  # 会话级缓存不做淘汰，满了整体重来
        return out

    # ---------- 异步请求（与 NeuralReranker 同构） ----------

    def request(self, code, ctx_text, cands):
        if self.ready and cands:
            # 整句优先占名额（2026-09-06 主人案二次钉死）：整句组是裁判的
            # 核心价值所在，截断把它切掉＝白裁（wjtxixhcy 静态位次 57 >
            # 旧 max_cand=48，目标句根本没进评分名单，刷新后纹丝不动）。
            # 词侧有统计+融合分兜底，让位。
            # ⚠️ 不做前缀去重：最小语义对（想吃/想出）恰在前 4 字内分歧，
            # 任何前缀粗筛都会把目标句切出评分名单（v2 探针 C 组实测 FAIL）。
            # 容量压力交给 RBT3 粗筛两级裁决（引擎 _on_nr_result）。
            sents = [c for c in cands if len(c) >= 5]
            words = [c for c in cands if len(c) < 5]
            sents = sents[:self.max_cand]
            words = words[: max(0, self.max_cand - len(sents))]
            self._q.put((code, (ctx_text or "")[-16:], sents + words))

    def _worker(self):
        while True:
            code, ctx_text, cands = self._q.get()
            # 只处理最新请求：连击时中间码的精排没有意义
            try:
                while True:
                    code, ctx_text, cands = self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                t0 = time.perf_counter()
                scores = self.score(cands, ctx_text)
                ms = round((time.perf_counter() - t0) * 1000)
                if self.on_result:
                    self.on_result(code, scores, ms)
            except Exception as e:
                self.log("[LLM裁判] 打分异常已兜底: %r" % e)


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    qj = QwenJudge(os.path.join(base, "ai_llm", "qwen25-05b-hf"))
    loaded = threading.Event()
    orig = qj._load

    def _load2():
        orig()
        if qj.ready:
            loaded.set()
    qj._load = _load2
    qj.load_async()
    loaded.wait(120)
    if not qj.ready:
        print("加载失败")
        raise SystemExit(1)

    t0 = time.perf_counter()
    s = qj.score(["那我给你个东西测试一下", "那我给你过渡性测试一下",
                  "我今天想吃西湖醋鱼", "我今天想出西湖醋鱼",
                  "我吃了一个包子", "完成了一个不足"])
    print("%.0fms" % ((time.perf_counter() - t0) * 1000))
    for w, v in sorted(s.items(), key=lambda kv: -kv[1]):
        print("  %.2f %s" % (v, w))
