# -*- coding: utf-8 -*-
"""端侧判别重排器 FastReranker（20M 级内，ONNX 可导出）。

结构：共享 char-transformer 双塔 + listwise 打分头
  ctx 塔  : 上文尾部 12 字 → 编码 → 向量
  cand 塔 : 候选词 1~4 字 → 同一编码器 → 向量
  打分头  : [ctx; cand; ctx⊙cand; 标量特征] → MLP → 标量分
  训练    : 池内 softmax 交叉熵（候选顺序随机打乱，原始 rank 降级为特征，
            防位置作弊；rank 信号由 rank 特征显式提供）

推理形态（ONNX）：
  输入 ctx_ids[12] int32, cand_ids[C,4] int32, feats[C,3] float32, cand_mask[C] float32
  输出 scores[C] float32   （C=池大小，动态轴）
"""
import math
import torch
import torch.nn as nn


class FastReranker(nn.Module):
    PAD, UNK = 0, 1

    def __init__(self, vocab_size, d=256, nlayer=2, nhead=4,
                 ctx_len=12, cand_len=4, nfeat=3, ffn=768):
        super().__init__()
        self.d = d
        self.ctx_len = ctx_len
        self.cand_len = cand_len
        self.PAD, self.UNK = 0, 1
        self.emb = nn.Embedding(vocab_size, d, padding_idx=self.PAD)
        self.pos_ctx = nn.Embedding(ctx_len, d)
        self.pos_cand = nn.Embedding(cand_len, d)
        layer = nn.TransformerEncoderLayer(
            d, nhead, ffn, dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, nlayer)
        self.enc_norm = nn.LayerNorm(d)
        # 标量特征: log1p(rank)/6, is_fixed, len/4
        self.head = nn.Sequential(
            nn.Linear(d * 3 + nfeat, 256), nn.GELU(),
            nn.Linear(256, 64), nn.GELU(),
            nn.Linear(64, 1))
        self.nfeat = nfeat

    def _encode(self, ids, pos_emb):
        # ids [B,L] / [B,C,L] → 向量池化
        shp = ids.shape
        flat = ids.reshape(-1, shp[-1])
        mask = flat.eq(self.PAD)
        # 首位置永远可见：句首无上文时 ctx 全 pad，若整行 mask 全 True
        # attention softmax 直接 NaN（2026-09-08 主人整句用例实测炸出）
        mask[:, 0] = False
        h = self.emb(flat) + pos_emb.weight[: shp[-1]].unsqueeze(0)
        h = self.encoder(h, src_key_padding_mask=mask)
        m = (~mask).float().unsqueeze(-1)
        v = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.enc_norm(v).reshape(*shp[:-1], self.d)

    def forward(self, ctx_ids, cand_ids, feats, cand_mask):
        """ctx_ids [B,12]  cand_ids [B,C,4]  feats [B,C,3]  cand_mask [B,C]"""
        cv = self._encode(ctx_ids, self.pos_ctx)                    # [B,d]
        wv = self._encode(cand_ids, self.pos_cand)                  # [B,C,d]
        c = cv.unsqueeze(1).expand(-1, wv.size(1), -1)              # [B,C,d]
        z = torch.cat([c, wv, c * wv, feats], dim=-1)               # [B,C,3d+3]
        scores = self.head(z).squeeze(-1)                           # [B,C]
        scores = scores.masked_fill(cand_mask.eq(0), -1e4)
        return scores


def build_vocab(sessions_path=None, min_freq=2, files=()):
    """从样本 JSONL 收集汉字词表。返回 {char: id}，PAD=0 UNK=1。"""
    from collections import Counter
    cnt = Counter()
    import json
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                for ch in r["ctx"] + r["gold"]:
                    cnt[ch] += 1
                for w in r["cands"]:
                    for ch in w:
                        cnt[ch] += 1
    vocab = {"<pad>": 0, "<unk>": 1}
    for ch, n in sorted(cnt.items(), key=lambda x: -x[1]):
        if n >= min_freq and "\u4e00" <= ch <= "\u9fff":
            vocab[ch] = len(vocab)
    return vocab


if __name__ == "__main__":
    m = FastReranker(vocab_size=8000)
    n_param = sum(p.numel() for p in m.parameters())
    print("参数量: %.2fM" % (n_param / 1e6))
    B, C = 2, 16
    ctx = torch.randint(2, 8000, (B, 12))
    cand = torch.randint(2, 8000, (B, C, 4))
    feats = torch.randn(B, C, 3)
    mask = torch.ones(B, C)
    s = m(ctx, cand, feats, mask)
    print("输出:", s.shape)
