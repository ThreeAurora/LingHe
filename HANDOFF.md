# LingHe 交接文档（2026-09-06）

> 写给接手的 AI：主人是小鹤音形用户，要求**整句/接龙智能候选**。请先读完本文再动手。
> 项目路径：`E:\CCSpace\projects\2026\09\linghe`（git 仓库，远端 ThreeAurora/LingHe 私密）

---

## 1. 环境（必读，踩过坑）

- **Python 必须用 `E:/miniconda3/python.exe`**（托管 3.13 嵌入式版无 tkinter）。
- 路径一律 `E:/...` 风格；git 用 `git -C "E:/CCSpace/projects/2026/09/linghe"`（/e/ 路径 Windows git 不认）。
- 启动：`run.bat`（源码模式）。打包：`bash build_exe.sh`（一键 PyInstaller → dist/LingHe/，含 torch 版脚本已写好但**未重新打包验证**）。
- 裁判模型：`ai_llm/qwen25-05b-hf`（Qwen2.5-0.5B 判别式整句 logP，transformers 4.46.3 + torch 2.2.2 CPU，加载约 35s）。
- 底库加载约 30-45s（273 万词），探针请后台跑。

## 2. 架构一句话

击键 → 池装配（码表固频 + 底库简拼 + Viterbi 整句 + 锚点串接 anchor_sentences + 贪心 max_match + 口语字链 char_chains）→ 统计排序 → **LLM 裁判异步终审**（整句按纯 logP 置顶，词按 pool_score − λ·logP 融合）→ 刷新候选。

核心文件：`linghe.py`（Engine/钩子/UI）、`rerank.py`（StatReranker：viterbi/anchor/max_match/char_chains/score_word）、`dict_engine.py`（底库/索引/_ini_variants 平翘舌容错/_spoken_rank 掺权）、`ai_llm_judge.py`（QwenJudge）。

## 3. ⚠️ 当前 git 状态（接手第一件事）

**今天（2026-09-06）的修复已全部在工作区，先 `git -C <项目> diff` 通读再动代码**。要点：

1. `linghe.py`：单实例互斥 `acquire_single_instance()`（**ctypes 必须 `WinDLL("kernel32", use_last_error=True)` + `ctypes.get_last_error()`**，`windll.kernel32.GetLastError()` 读到的是残留错误码——曾致无实例也误判"已运行"直接退出，主人双击没反应的元凶之一）。windowed 日志写 `linghe.log`（追加模式）。
2. `linghe.py compute()`：删掉 `not mb_exact` 前置——它曾把 2 键简拼底库召回整个挡死（**bz 词池为空的根因**）；码表固频区封顶 24（`mb_part`），否则 prefix 字上百个吃光 90 池且 `n_mb > len(base)` 让 `_on_neural` 直接 return 丢弃裁判结果。
3. `linghe.py _commit()`：上屏后滚动 `ctx_tail_text`（裁判上文；旧代码只更新 ctx_prevs，读屏失败的应用里裁判永远看不到刚上屏的话）。
4. `rerank.py anchor_sentences()` 连环修复（**这是今天最大的战场，5 个 bug 叠加**）：
   - cores 排序改**口语频优先**（旧"跨度优先"让垃圾 span4 锚吃光名额）；
   - span4 锚从"只取 top1"改"掺权序取前 3"（initial_span top1 是书面权重序，测试一下 14305 被产生影响 43145 压住）；
   - 尾缝 `pos >= m` 分支：核链恰好填满全部键位时旧代码 `viterbi(空列表)` 返回空 → **目标句在最后一步被丢弃**（wjtxixhcy 缺席的根因）；
   - hot 核的 span4 变体分支**优先入队**（排 span2 分支后面会被吃光 made_cap=8 配额）；
   - span4 核 fork_cap=1 限流。
5. `rerank.py char_chains()`：尾词/字对变体（头链 top3 × 尾 2 键候选），prevs 可传入做上文条件化排序。

**验证状态（probe_e2e_v5 最新终验）**：wjtxixhcy 排名 1 ✅；nwgngdxcuyx 排名 2（进池了，裁判排第二——差最后一步）；wilygbz 缺席（见 §5 遗留）。

## 4. 主人验收用例全表

| # | 码 | 期望 | 键制 | 当前状态 | 病根/备注 |
|---|------|------|------|---------|----------|
| 1 | wjtxixhcy | 我今天想吃西湖醋鱼 | 双拼简 9 键 | **✅ 排名 1** | 已修复 |
| 2 | nwgngdxcuyx | 那我给你个东西测试一下 | 双拼简 11 键 | 🔶 **排名 2** | 进池了；裁判排 2，打印 top1 看是谁、分差多少，调融合或准入 |
| 3 | wilygbz | 我吃了一个包子 | 双拼简 7 键 | ❌ 缺席 | 见 §5.1 |
| 4 | wztwsmsh | 我昨天晚上没睡好 | 双拼简 8 键（修正码 wztwsmuh 睡=sh→u 同测） | ❓ 未在修复后重测 | 没睡 spoken 2517 过锚点门槛，名额修复后大概率好转，**先重测** |
| 5 | bydxwhm | 不要丢下我好吗 | 双拼简 7 键 | ❓ 同上 | 丢下 spoken 2517，同 §5.2 名额链 |
| 6 | wmncq / wmniq | 武媚娘传奇 | 双拼简 5 键 + py 全码 wumwnlirqi | ❌ 缺席 | **底库无此词**（weight 0）——数据问题，走 fetch_hotwords.py 热词入库，或接受做不到 |
| 7 | wbzdlszmsd | 我不知道老师怎么说的 | 双拼简 10 键 | ❓ 未重测 | 老师 weight 332029 在库；历史/老师同键竞争，靠锚点+裁判 |
| 8 | tjz | 统计中 | 双拼简 3 键 | ❓ 未测 | 统计中 weight 50 偏低，靠裁判 |
| 9 | ahhjtll | 阿哈哈鸡汤来咯 | 双拼简 7 键 | ❌ | 阿哈哈不在库（主人已说不需要首选，极难） |
| 10 | 诗仙+lb | 李白 | 接龙 2 键 | ✅ 裁判复核 rank 1 | |
| 11 | 诗圣+df | 杜甫 | 接龙 2 键 | ✅ 同上 | |
| 12 | 我今天吃了一个+bz | 包子 | 接龙 2 键 | ✅ **裁判链 rank 1**（probe_real_chain B 案） | 主诉案已修 |
| 13 | 草原上奔跑了一只+lyz | 老鹰？ | 接龙 3 键 | ❌ | **码本身存疑**：老鹰=ly 两键，lyz 匹配的是"了一只"；先和主人确认期望 |

## 5. 遗留问题与修复方向（按优先级）

### 5.1 wilygbz 的"包子"尾巴（最大遗留，已挖到证据链底层）
- 头链已修好：字链 beam top3 里有"我吃了一个"。
- 尾巴证据全部不足：`包`字不在 b 键口语字库前 20；sp2(包子)=465 vs sp2(不在)=26794；lookup_initial('b z') 静态序包子排 27；**无屏幕上文时 score_word 也压不过部长/不足**。
- **建议方向（未实施）**：char_chains 变体选尾词时，把**头链末词（"一个"）作为句内上文**传给 score_word（现在的 prevs 只来自屏幕）；或对 sp2(头末字+尾首字)（如"个包"）加权。若仍不通，此案需神经引导 beam（RBT3 逐前缀剪枝），是下一个大工程。
- 注：主人说"武媚娘传奇/不要丢下我好吗"这类**不需要首选**，只要出现在候选项里即可——放宽验收口径能救回一批。

### 5.2 接手后先做（半天内能出结果）
1. 重测 wztwsmsh/wztwsmuh/bydxwhm/tjz（名额修复后未验证，探针 `cache/tmp/probe_real_chain.py` 加 case 即可）。
2. nwgngdxcuyx 排名 2：打印裁判 top3 分数，若第一名是垃圾句，调 `_on_neural` 整句组或 λ。
3. 全部通过后：`bash build_exe.sh`（**含 torch 版**，QwenJudge 完整内嵌；旧版 exclude torch 是错的——裁判是智能句的命根子）+ 双击冒烟（互斥已修，双击多次只跑一个）+ `git commit/push`。

### 5.3 裁判速度（体验项）
41 句全量打分 15-25s（CPU fp32）。异步不阻塞但刷新慢。方向：request 只送统计序前 20 句；或换 ONNX int8；或 batch 调大。

## 6. 避坑清单（前人踩过的，别再踩）

1. **ctypes GetLastError 污染**（见 §3.1）——互斥误判导致"双击没反应"。
2. **多实例叠加**：旧版 exe 无互斥，主人双击 4 次跑 4 个实例互相抢键，表现为"只能打一个字母"。现在有互斥了。
3. **探针≠真实链路**：probe 手拼通道与 Engine.compute() 是两条代码路径，必须用 `cache/tmp/probe_real_chain.py`（Engine 本体）验证，否则修了探针坏了真实。
4. **judge 就绪要 35s**：启动后半分钟内打字走 RBT3 回退，判别力弱。别在这窗口里测智能句。
5. **验收口径**：部分用例主人明说"不需要首选，出现即可"（wmncq/ahhjtll）；别按全必 top1 收敛。
6. rerank 的 cost 是**越小越好**（负数比较），排查"排序矛盾"先查符号方向。
7. 一功能一 git 提交（主人长期纪律），中文前缀提交信息。

## 7. 探针工具箱（cache/tmp/，均可直接后台跑）

- `probe_real_chain.py` — **首选**：Engine 本体 + 裁判，三案（wilygbz/bz有上文/bz无上文）
- `probe_e2e_v5.py` — 端到端终验三用例（nwgngdxcuyx/wjtxixhcy/wilygbz），输出"验收: ✗(2)/✓/✗(None)"格式
- `probe_owner_samples.py` — 主人样本批测（整句组/接龙组/py全码）
- `probe_anchor_diag2.py` — 锚点 cores 收集与串接产物诊断（本次连环 bug 全靠它定位）
- `probe_diag_bz.py` — 字库/字对频诊断模板

跑法：`E:/miniconda3/python.exe cache/tmp/xxx.py`，裁判加载 35s，全套 2 分钟，**后台跑**。
