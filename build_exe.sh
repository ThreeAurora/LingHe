#!/bin/bash
# LingHe exe 一键组装：PyInstaller → 补 onnxruntime dll → 拷数据目录 → 就绪。
# 用法：bash build_exe.sh（在项目根执行；产物 dist/LingHe/，双击 LingHe.exe）
# v2：不再排除 torch/transformers——QwenJudge 裁判随 exe 走（wilygbz/接龙全量验收），
#     模型权重 ai_llm/qwen25-05b-hf 约 1GB 一并拷入，产物约 2.5GB。
set -e
cd "$(dirname "$0")"

PY=${PY:-E:/miniconda3/python.exe}

echo "[1/4] PyInstaller 打包（含 torch/transformers，QwenJudge 裁判完整内嵌）"
"$PY" -m PyInstaller --noconfirm --windowed --name LingHe \
    linghe.py > cache/tmp/pyinstaller_log.txt 2>&1

echo "[2/4] 补 onnxruntime providers dll（PyInstaller hook 遗漏）"
SP=$("$PY" -c "import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'))")
cp "$SP/onnxruntime_providers_shared.dll" dist/LingHe/_internal/onnxruntime/capi/

echo "[3/4] 拷数据目录（重打包会清空 dist，此步必须每次执行）"
D=dist/LingHe
cp config.json $D/
mkdir -p $D/dicts $D/mabiao $D/ai_neural $D/ai_llm
for f in 8105.dict.yaml base.dict.yaml char_pinyin.txt cooc_pre.bin cooc_pre.txt \
         hotwords_inc.dict.yaml spoken_2gram.txt spoken_3gram.txt spoken_freq.txt \
         tencent.dict.yaml user_dict.txt wanxiang_inc.dict.yaml; do
    cp dicts/$f $D/dicts/
done
cp -r mabiao/. $D/mabiao/
cp -r ai_neural/rbt3 $D/ai_neural/
# QwenJudge 模型权重（frozen 模式 base=exe 目录，ai_llm_judge.py 会到 <exe>/ai_llm 找）
cp -r ai_llm/qwen25-05b-hf $D/ai_llm/
# 不拷：dicts/.cache.bin（运行时重建）、dicts/lexicon.db（孤儿产物）

echo "[4/4] 完成：$(du -sh $D | cut -f1) → $D/LingHe.exe"
