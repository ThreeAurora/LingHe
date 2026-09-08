# -*- coding: utf-8 -*-
"""FastReranker -> ONNX 导出（端侧接入 onnxruntime）。

推理图（C=池大小，动态轴）:
  输入 ctx_ids[1,12] i32 / cand_ids[1,C,4] i32 / feats[1,C,3] f32 / mask[1,C] f32
  输出 scores[1,C] f32
量化（可选）: onnxruntime.quantization 动态 INT8 → 体积 ~1/4。
"""
import os, sys, json, argparse
import torch
from model import FastReranker

ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt/best.pt")
    ap.add_argument("--out", default="ckpt/reranker.onnx")
    ap.add_argument("--quant", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    ck = torch.load(a.ckpt, map_location="cpu")
    model = FastReranker(ck["vocab_size"], d=ck["d"], nlayer=ck["nlayer"])
    model.load_state_dict(ck["model"])
    model.eval()

    C = 16
    # 端侧喂 int32（fast_rerank 口径），图内 cast 回 long 喂 embedding
    ctx = torch.randint(2, ck["vocab_size"], (1, 12), dtype=torch.int32)
    cand = torch.randint(2, ck["vocab_size"], (1, C, 4), dtype=torch.int32)
    feats = torch.zeros(1, C, 3)
    mask = torch.ones(1, C)

    class _CastIn(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, ctx, cand, feats, mask):
            return self.m(ctx.long(), cand.long(), feats, mask)

    torch.onnx.export(
        _CastIn(model).eval(), (ctx, cand, feats, mask), a.out,
        input_names=["ctx_ids", "cand_ids", "feats", "cand_mask"],
        output_names=["scores"],
        dynamic_axes={"cand_ids": {1: "C"}, "feats": {1: "C"},
                      "cand_mask": {1: "C"}, "scores": {1: "C"}},
        opset_version=14)
    print("exported ->", a.out)

    # 数值一致性校验
    import onnxruntime as ort
    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    o = sess.run(["scores"], {
        "ctx_ids": ctx.numpy().astype("int32"),
        "cand_ids": cand.numpy().astype("int32"),
        "feats": feats.numpy().astype("float32"),
        "cand_mask": mask.numpy().astype("float32")})[0]
    with torch.no_grad():
        ref = model(ctx.long(), cand.long(), feats, mask).numpy()
    err = abs(o - ref).max()
    print("onnx vs torch max|Δ| = %.2e" % err)

    if a.quant:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        q = a.out.replace(".onnx", ".int8.onnx")
        quantize_dynamic(a.out, q, weight_type=QuantType.QInt8)
        print("quantized ->", q)
        sz = os.path.getsize(a.out) / 1e6
        szq = os.path.getsize(q) / 1e6
        print("size: fp32=%.1fMB int8=%.1fMB" % (sz, szq))
        # int8 数值校验（端侧实际跑的就是这张图）
        sessq = ort.InferenceSession(q, providers=["CPUExecutionProvider"])
        oq = sessq.run(["scores"], {
            "ctx_ids": ctx.numpy().astype("int32"),
            "cand_ids": cand.numpy().astype("int32"),
            "feats": feats.numpy().astype("float32"),
            "cand_mask": mask.numpy().astype("float32")})[0]
        print("int8 vs torch max|Δ| = %.2e" % abs(oq - ref).max())


if __name__ == "__main__":
    main()
