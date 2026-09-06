#!/bin/bash
# LingHe exe 一键组装：PyInstaller → 补 onnxruntime dll → 拷数据目录 → 就绪。
# 用法：bash build_exe.sh（在项目根执行；产物 dist/LingHe/，双击 LingHe.exe）
set -e
cd "$(dirname "$0")"

PY=${PY:-E:/miniconda3/python.exe}

echo "[1/4] PyInstaller 打包（排除 torch/transformers，QwenJudge 自动回退 RBT3）"
"$PY" -m PyInstaller --noconfirm --windowed --name LingHe \
    --exclude-module torch --exclude-module transformers \
    --exclude-module tokenizers --exclude-module safetensors \
    --exclude-module huggingface_hub linghe.py > cache/tmp/pyinstaller_log.txt 2>&1

echo "[2/4] 补 onnxruntime providers dll（PyInstaller hook 遗漏）"
SP=$("$PY" -c "import onnxruntime, os; print(os.path.join(os.path.dirname(onnxruntime.__file__), 'capi'))")
cp "$SP/onnxruntime_providers_shared.dll" dist/LingHe/_internal/onnxruntime/capi/

echo "[3/4] 拷数据目录（重打包会清空 dist，此步必须每次执行）"
D=dist/LingHe
cp config.json $D/
mkdir -p $D/dicts $D/mabiao $D/ai_neural
for f in 8105.dict.yaml base.dict.yaml char_pinyin.txt cooc_pre.bin cooc_pre.txt \
         hotwords_inc.dict.yaml spoken_2gram.txt spoken_3gram.txt spoken_freq.txt \
         tencent.dict.yaml user_dict.txt wanxiang_inc.dict.yaml; do
    cp dicts/$f $D/dicts/
done
cp -r mabiao/. $D/mabiao/
cp -r ai_neural/rbt3 $D/ai_neural/
# 不拷：dicts/.cache.bin（运行时重建）、dicts/lexicon.db（孤儿产物）、ai_llm/（torch 依赖，exe 版回退 RBT3）

echo "[4/4] 完成：$(du -sh $D | cut -f1) → $D/LingHe.exe"
