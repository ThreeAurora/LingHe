# -*- coding: utf-8 -*-
"""端侧快速重排层 FastRerank：ONNX 模型 + margin 门控。

定位（2026-09-07 定案）：
  - 本地 <30ms，吃云端裁判吃不到的场景（0.2s 内本地闭环）；
  - **保守门控**：模型首选与引擎原首选分差 >= tau 才换位——
    eval 实测 tau=1.5 时 P@1 净赚 26+ 点、rank0 误伤 0.4%；
  - 模型不Reload会学 rank 特征（输入即引擎序），推理不打乱候选。

用法：
    fr = FastRerank(BASE, cfg, log)
    new_cands = fr.rerank(ctx_text, cands, n_fixed)   # 未就绪/异常返回原序
"""
import os
import json

import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None


class FastRerank:
    CTX_LEN = 12
    CAND_LEN = 4
    MAX_CAND = 96   # 与真机 candidate_pool=90 对齐：深位金标必须进视野
    NFEAT = 3

    def __init__(self, base_dir, cfg, log=lambda s: None):
        self.ready = False
        self.log = log
        fcfg = (cfg.get("fast_rank") or {})
        self.enabled = bool(fcfg.get("enabled", True))
        self.tau = float(fcfg.get("tau", 1.5))
        if not self.enabled or ort is None:
            return
        mpath = os.path.join(base_dir, "models", "reranker.onnx")
        vpath = os.path.join(base_dir, "models", "reranker_vocab.json")
        if not (os.path.exists(mpath) and os.path.exists(vpath)):
            return
        try:
            so = ort.SessionOptions()
            so.intra_op_num_threads = 1          # 输入法主线程内跑，禁并发
            self.sess = ort.InferenceSession(
                mpath, so, providers=["CPUExecutionProvider"])
            self.vocab = json.load(open(vpath, encoding="utf-8"))
            self.ready = True
            self.log("[快排] ONNX 就绪 tau=%.1f" % self.tau)
        except Exception as e:                    # 模型坏不拖垮输入法
            self.log("[快排] 加载失败: %s" % e)

    def _ids(self, word):
        return [self.vocab.get(ch, 1) for ch in word[: self.CAND_LEN]]

    def rerank(self, ctx_text, cands, n_fixed):
        """返回重排后候选列表；未就绪/异常/少候选一律原序返回。"""
        if not self.ready or not self.enabled or len(cands) < 2:
            return cands
        try:
            top = cands[: self.MAX_CAND]
            rest = cands[self.MAX_CAND:]
            ctx_ids = [self.vocab.get(ch, 1)
                       for ch in (ctx_text or "")[-self.CTX_LEN:]]
            ctx_ids = ([0] * (self.CTX_LEN - len(ctx_ids)) + ctx_ids)[: self.CTX_LEN]
            cand_ids = [self._ids(w) + [0] * (self.CAND_LEN - len(self._ids(w)))
                        for w in top]
            while len(cand_ids) < self.MAX_CAND:
                cand_ids.append([1] * self.CAND_LEN)
            feats, mask = [], []
            for i, w in enumerate(top):
                feats.append([np.log1p(i) / 6.0,
                              1.0 if i < n_fixed else 0.0,
                              len(w) / 4.0])
                mask.append(1.0)
            while len(feats) < self.MAX_CAND:
                feats.append([0.0, 0.0, 0.0])
                mask.append(0.0)
            sc = self.sess.run(["scores"], {
                "ctx_ids": np.array([ctx_ids], dtype=np.int32),
                "cand_ids": np.array([cand_ids], dtype=np.int32),
                "feats": np.array([feats], dtype=np.float32),
                "cand_mask": np.array([mask], dtype=np.float32),
            })[0][0]
            pred = int(np.argmax(sc))
            # margin 门控：分差不够大不动原首选
            if pred != 0 and (sc[pred] - sc[0]) < self.tau:
                return cands
            if pred == 0:
                return cands
            new_top = [top[pred]] + [w for i, w in enumerate(top) if i != pred]
            self.log("[快排] %s 置顶 (margin=%.2f)" % (top[pred], sc[pred] - sc[0]))
            return new_top + rest
        except Exception as e:
            self.log("[快排] 推理异常: %s" % e)
            return cands
