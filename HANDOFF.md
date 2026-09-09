# LingHe 开发交接文档（2026-09-09 更新）

> 写给接手的开发者：对外介绍看 `README.md`，本文只讲开发环境、当前状态、
> 验收用例和踩过的坑。项目路径 = 本仓库根目录（远端 ThreeAurora/LingHe）。

## 1. 环境

- Python 需要带 tkinter（本项目用 miniconda；嵌入式发行版没有 tkinter）。
- 启动：`run.bat`（源码模式）；打包：`bash build_exe.sh`（PyInstaller → `dist/LingHe/`）。
- 底库 273 万词：全量缓存加载约 44s；轻量索引 20 万词 2~9s 即可打字，全量后台热切换。
- 云端裁判：DeepSeek（`config.json` 的 `judge` 段，key 不入库）；本地 Qwen 裁判模型
  （`ai_llm/qwen25-05b-hf`）需自备，加载约 40s。
- 探针脚本请后台跑（初始化约 100s+）。

## 2. 架构一句话

击键 → 池装配（码表固频 + 底库简拼 + Viterbi 整句 + 锚点串接 + 贪心 max_match +
口语字链 char_chains）→ 统计排序 → 万象语法（mmap 直读、微秒级）→ 云端 LLM 异步终审
→ 刷新候选；端侧神经重排 FastReranker 已训练导出，待接线。

核心文件：`linghe.py`（Engine/钩子/UI/候选装配）、`rerank.py`（StatReranker：
viterbi/anchor/max_match/char_chains/score_word/_eval_sent/_clean_chain）、
`dict_engine.py`（底库/索引/两段式加载）、`gram_model.py`（万象语法）、
`cloud_judge.py`（DeepSeek 云端裁判）、`fast_rerank.py`（端侧神经重排件）。

## 3. 当前 git 状态（2026-09-09）

- README 已全面重写至最新进度；`config.json` 已移出版本库（`config.example.json` 为模板）；
- `cache/tmp/` 等一次性中间产物已从索引清理（`.gitignore` 覆盖，本地文件保留）；
- LICENSE：GPL-3.0（词库按 GPL 分发，整体同许可）；
- 历史中曾出现的明文 API key 已从索引移除；彻底清除需重写历史（`git filter-repo`）。

## 4. 关键机制（2026-09-07 ~ 09-09）

### 4.1 万象语法模型（gram_model.py）

- mmap 直读 `.gram`（Rime::Grammar/1.0 + Darts-clone），查询微秒级；
- `gram.query(context, word, is_rear=False)` 返回搭配对数概率；`NC_PEN = -12.0`
  为无证据基准，`ev = 分 - NC_PEN`；
- 调用约定 `is_rear=(len(w)>=5)`（长词才按句尾查）；
- 用法（`linghe.py` compute()）：动态池候选按上下文减价
  `GRAM_ALPHA(0.25) × min(GRAM_MAX(12), ev)`，有证据的词浮顶；压顶闸
  `GRAM_TOP_EV(6.0)`。

### 4.2 多窗口上下文

- 病根：语法模型只认尾缀搭配，「喷出了孢子」无词条 → 整段证据归零；「蘑菇→孢子」
  藏在更早位置被中间词截断；
- 解法：`gwin = [尾词] + 上文最近 3 词`，对每个候选逐窗查取 max ev；
- 验证：`bczi` + 上文「蘑菇喷出了」→ 孢子第 1。

### 4.3 零孤字词链

- 病根：整句闸只认口语 3gram 证据，5 字短语中间跨词 3gram（史级对）天然是残渣，
  真短语与拼字伪句同分被挡；
- 解法：`_clean_chain(s)` = 整句可贪心切分成纯多字词典词（史诗级|对决），无单字残渣，
  只对整句闸放行；
- 验证：`uiuijidvjt` → 史诗级对决 进候选第 4 位。

### 4.4 整句音节守卫

- 病根：dyn 压顶把 Viterbi 拼字伪句（不存在出）与异长词（菠菜甾醇）提到固频前；
- 解法：`real_gev` 只保留「词库真词 + 音节数匹配」的候选；压顶门槛 `GRAM_TOP_EV=6.0`；
- 验证：`bczi` + 蘑菇喷出了 → 孢子 #1，伪句不进前排。

### 4.5 整句一次性识别

- `mogupfiulebczi` → 蘑菇喷出了孢子 第 1（dyn 开/关均 #1）；
- 句内尾词语法证据：对 `len>=5 且 _clean_chain` 的整句候选，用句内前词当窗口查尾词，
  封顶放宽 2×（整句内部排序是统计地界，证据差须完整反映）。

### 4.6 端侧神经重排 FastReranker（待接线）

- 训练管线 `train/`（`gen_data.py` 造语料、`eval.py` 评估、`export_onnx.py` 导出 + ckpt）；
- v4 评估（74398 验证样本、候选池 48）：基线 P@1 46.38% → 纯重排 86.58%，
  P@3 90.21%，MRR 0.8898；
- margin 门控：`tau=1.5` 时 P@1 85.81%、rank0 误伤 0.477%；
- 接线待办：`models/reranker.onnx` + `config.fast_rank` + `linghe.py` 调用点
  （线上池 90，需按 90 池复测）；
- 旧 RBT3（`neural_rerank.py`）已停用（`neural.enabled=false`），代码保留。

## 5. 验收用例全表（2026-09-07 更新，之后未重测）

| # | 码 | 期望 | 状态 |
|---|------|------|---------|
| 1 | wjtxixhcy | 我今天想吃西湖醋鱼 | ✅ 排名 1 |
| 2 | nwgngdxcuyx | 那我给你个东西测试一下 | 🔶 排名 2 |
| 3 | wilygbz | 我吃了一个包子 | ❌ 缺席（旧遗留，见 §6） |
| 4 | uiuijidvjt | 史诗级对决 | ✅ 第 4 位（进池） |
| 5 | bczi（上文 蘑菇喷出了） | 孢子 | ✅ 第 1 |
| 6 | mogupfiulebczi | 蘑菇喷出了孢子 | ✅ 第 1 |
| 7 | uiui（上文 这是一部） | 史诗 | ✅ 第 1 |
| 8 | 我今天吃了一个+bz | 包子 | ✅ 裁判链 rank 1 |
| 9 | 诗仙+lb / 诗圣+df | 李白 / 杜甫 | ✅ rank 1 |
| 10 | wztwsmsh / wztwsmuh | 我昨天晚上没睡好 | ❓ 未重测 |
| 11 | bydxwhm | 不要丢下我好吗 | ❓ 未重测 |
| 12 | wmncq / wmniq | 武媚娘传奇 | ❌ 底库无此词（数据问题） |
| 13 | tjz | 统计中 | ❓ 未测（权重偏低） |

## 6. 遗留问题

1. `wilygbz` 的「包子」尾巴：尾巴证据不足（包子 vs 不在，sp2 差距大）。建议方向：
   `char_chains` 选尾词时把头链末词（「一个」）作为句内上文传给 `score_word`，
   或对 `sp2(头末字+尾首字)` 加权；
2. 重测用例 10/11/13（历史探针脚本已不入库，需按 §4 诊断思路重建）；
3. FastReranker 接线（见 §4.6）；
4. 「史诗级」以词入库可让 `uiuijidvjt` 升第 1（可选优化）。

## 7. 避坑清单

1. **ctypes GetLastError 污染**——互斥误判导致「双击没反应」
   （须 `use_last_error=True` + `ctypes.get_last_error()`）；
2. **探针 ≠ 真实链路**：必须用 Engine 本体，否则修了探针坏了真实；
3. **语法模型只认尾缀搭配**：中间截断词串证据归零，必须逐上文词查窗取 max（§4.2）；
4. **dyn 压顶前必须过滤**：词库真词 + 音节数匹配，否则拼字伪句/异长词蹭证据霸榜（§4.4）；
5. **整句排序封顶不能死用 GRAM_MAX**：同音词句会全顶到上限分不出先后（§4.5）；
6. **judge 就绪要时间**：启动前半分钟打字走回退，别在这窗口测智能句；
7. **rerank 的 cost 越小越好**（负数比较），排查排序矛盾先查符号方向；
8. 验收口径：部分用例只要求「出现即可」（如 武媚娘传奇）；
9. 一功能一 git 提交，中文前缀提交信息。

## 8. 探针工具箱

`cache/tmp/` 已整体不入库（`.gitignore`），历史探针（`probe_bczi9.py`、
`probe_real_chain.py` 等）可能已清理；需要时按 §4 的诊断思路重建。

跑法：`python cache/tmp/xxx.py`（后台跑，初始化约 100s）。
