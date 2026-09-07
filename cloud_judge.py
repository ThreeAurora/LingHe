# -*- coding: utf-8 -*-
"""云端大模型裁判 —— 火山引擎 Ark（豆包）判别式候选排序。

为什么是云端（2026-09-07 主人拍板，与 0.5B 的分水岭）
------------------------------------------------
本地 Qwen2.5-0.5B 判别式打分是「候选组内比通顺」，没有世界知识：
- 简拼整句（wjtxixhcy→我今天想吃西湖醋鱼）信息严重不足，0.5B 只能
  靠字面通顺猜，歧义天花板上限很低；
- 豆包输入法「简拼首选」的真正来源是云端大模型：语义+常识+用户意图
  理解。0.5B 与云端 LLM 的差距不是调参能补的，是模型规模的分水岭。

本裁判**不生成**，只做判别式重排（豆包第二层路线的云端形态）：
- 输入：上文 + 键入码 + 本地召回候选列表；
- 输出：模型按「用户此刻最可能想输入的」从高到低排序的候选数组；
- 排序转分数 {词: score}（大=更可能），与 QwenJudge 的每字 logP
  方向一致，可直接走 _on_neural 的融合回流（pool − λ·score）。

工程
----
- OpenAI 兼容（Ark /chat/completions），urllib 零依赖（复用 ai_engine
  的请求写法）；异步单 worker + 只保留最新请求（同 QwenJudge/ai_engine
  的线程模型），击键永不阻塞。
- 失败/超时/未就绪 → 静默降级（ready=False，引擎回落本地 0.5B/统计），
  日志只记一行。
- 结果按 (上文, 键入码) 缓存：同码重打零网络开销。
"""

import json
import math
import queue
import re
import threading
import time
import urllib.error
import urllib.request

# 排序提示词：只准输出 top10 数组，禁止解释/增删改。
# 【2026-09-07 截断案】请求名单必须含全量动态真词池（包子静态 33 位，
# 只传前 24 等于没发——云端不是神仙）；模型只回 top10：输出越短越快，
# 未上榜词保持统计原序（_on_cloud_rank 落位规则）。
SYSTEM_PROMPT = (
    "你是中文输入法的候选排序助手。用户正在输入法里打字，给出了一串输入编码和"
    "一组候选（词或句子）。请结合上文，把候选中「用户此刻最可能想输入的」前 10 个"
    "按可能性从高到低挑出来。只输出一个 JSON 字符串数组（最多 10 个元素，元素必须"
    "是候选原文），禁止任何解释、拼音或注释。"
)

USER_PROMPT = (
    "上文：{ctx}\n"
    "输入编码：{code}\n"
    "候选：{cands}\n"
    "排序结果："
)

# 生成首选提示词（2026-09-07 主人「真正的智能」案）：与排序通道的分水岭——
# 排序通道从候选池里挑（池里没有的永远出不来），生成通道直接写答案：
# 豆包输入法简拼整句首选（wjtxixhcy→我今天想吃西湖醋鱼）靠的就是生成，
# 不需要这句话存在于任何词表。只给输入码+上文，不喂候选池。
# 编码两种形态都要认：简拼=每键一声母（wjtx=我今天想），双拼=每 2 键一音节。
# 2026-09-07 探针钉死：把 wjtxixhcy 当双拼解码会出「我今天下午去」这类
# 音不对的答案；明确声明简拼后模型才能对号入座。
# 2026-09-07 二修（主人「没联想词语和纯双拼」案）：只给声母映射时，双拼
# 词码 wm=我们 模型解不出韵母返回空——补齐完整小鹤双拼键位（声母+韵母+
# 零声母），并给出「奇数键=简拼 / 偶数键=双拼（兼简拼词）」的判定规则。
GEN_SYSTEM = (
    "你是中文输入法，负责把输入编码转成用户想打的词或句子。\n"
    "编码有两种形态：\n"
    "1. 简拼：每键一声母。zh/ch/sh 分别用 v/i/u 键，其余键即声母本身。\n"
    "   例1：wjtxixhcy = w我 j今 t天 x想 i吃 x西 h湖 c醋 y鱼 = 我今天想吃西湖醋鱼。\n"
    "   例2：ilyg = i出(ch) l来 y一 g个 = 出来一个（注意：i 键表示声母 ch，不是 y）。\n"
    "2. 双拼（小鹤）：每 2 键一个完整音节。声母键：v=zh i=ch u=sh，其余键即声母。\n"
    "   韵母键：q=iu w=ei e=e r=uan t=ue y=un u=u i=i o=uo p=ie a=a s=ong d=ai\n"
    "   f=en g=eng h=ang j=an k=ing l=iang z=ou x=ia c=ao v=ui b=in n=iao m=ian。\n"
    "   零声母两键：aa=a oo=o ee=e er=er ai=ai ei=ei ou=ou ao=ao an=an en=en。\n"
    "判定：奇数键=简拼（按声母逐字读）；偶数键=双拼整音节，同时可作简拼词解读"
    "（如 wm=我们/外贸 是简拼；纯双拼例：wj=w+an=晚/万、pb=p+in=拼、ul=sh+uang=双）。\n"
    "硬性要求：输出的每个字，其读音必须与编码严格对应——简拼时每键是该字声母"
    "（ch→i 键、zh→v 键、sh→u 键），双拼时每 2 键解码后正好是该字拼音。"
    "任何一个字对不上编码就整条作废。宁缺毋滥：解不出就输出空数组 []。\n"
    "请结合上文（若有）和常识，输出用户此刻最可能想打的 1 到 3 个词或句子，"
    "按可能性从高到低。只输出一个 JSON 字符串数组，禁止任何解释、拼音或注释。"
)

GEN_PROMPT = (
    "上文：{ctx}\n"
    "输入编码：{code}\n"
    "候选："
)


_OPENER = None


def _no_proxy_opener():
    """绕过系统代理直连。

    2026-09-07 实测钉死：Windows 上 urllib 自动读注册表系统代理
    （Clash Verge 设的 127.0.0.1:7890），流量进 mihomo mixed 端口后被
    拖到 9.3s/请求（连 GET /models 都 9.4s）；socket 直连走 TUN 的
    DIRECT 规则只要 65~120ms。云端裁判是打字节奏里的东西，9s 等于
    不可用，必须绕过代理直连。
    """
    global _OPENER
    if _OPENER is None:
        _OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return _OPENER


def _post_json(url, payload, headers=None, timeout=8.0):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with _no_proxy_opener().open(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _parse_array(text: str):
    """从模型输出中提取候选数组（尽量宽松）。"""
    if not text:
        return []
    m = re.search(r"\[.*\]", text, re.S)
    if m:
        try:
            arr = json.loads(m.group(0))
            if isinstance(arr, list):
                return [str(x).strip() for x in arr if str(x).strip()]
        except (json.JSONDecodeError, ValueError):
            pass
    return []


class CloudJudge:
    """云端大模型判别裁判（Ark/OpenAI 兼容端点）。

    与 QwenJudge 同构接口：load_async() / ready / on_result(code, scores, ms)
    / on_ready / request(code, ctx_text, cands)。scores 为 {词: score}，
    score 大 = 更可能是用户想打的（排序靠前的候选 score 更大）。
    """

    def __init__(self, cfg, log=print, max_cand=30):
        self.cfg = cfg or {}
        self.log = log
        self.ready = False
        self.on_result = None          # 结果回调（引擎注入，走 _on_neural）
        self.on_ready = None           # 就绪回调（引擎升级在屏旧码）
        self.max_cand = int(self.cfg.get("max_cand", max_cand))
        self._base = (self.cfg.get("api_base", "") or
                      "https://ark.cn-beijing.volces.com/api/v3").rstrip("/")
        self._key = self.cfg.get("api_key", "")
        self._model = self.cfg.get("model", "")
        self._timeout = float(self.cfg.get("timeout", 8.0))
        self._q = queue.Queue()
        self._cache = {}               # (ctx_key, code) -> {词: score}
        self._cache_order = []
        self._lock = threading.Lock()
        self._probe_ms = 0
        self._fail = 0                 # 连续失败计数（达阈值自降级）

    # ---------- 探测（异步）----------

    def load_async(self):
        threading.Thread(target=self._load, daemon=True).start()

    def _load(self):
        """小请求验证 key/model/base 连通。成功才置 ready。

        2026-09-07 探针案：启动瞬间网络未就绪/偶发超时会让探测一次失败就
        永久停用（回落本地），云端能力整个消失；且启动期底库全量加载（99s）
        霸占 GIL，网络线程 8s 超时必饿死。改为 10s 间隔宽松重试直到成功
        （最多 5 分钟，远超 99s 加载窗），连续探测失败只记日志。
        """
        if not (self._base and self._key and self._model):
            self.log("[云端裁判] 未配置 api_base/api_key/model，停用（回落本地）")
            return
        attempt = 0
        while attempt < 30:
            attempt += 1
            try:
                t0 = time.perf_counter()
                ordered = self._call_gen("wjtx", "")  # 生成通道探测：一个双拼整词码
                self._probe_ms = round((time.perf_counter() - t0) * 1000)
                if ordered:
                    self.ready = True
                    self.log("[云端裁判] 就绪 %s 模型=%s 探测%dms" %
                             (self._base, self._model, self._probe_ms))
                    threading.Thread(target=self._worker, daemon=True).start()
                    if self.on_ready:
                        try:
                            self.on_ready()
                        except Exception:
                            pass
                    return
                self.log("[云端裁判] 探测第 %d 次无输出，10s 后重试" % attempt)
            except Exception as e:
                self.log("[云端裁判] 探测第 %d 次失败（10s 后重试）: %r" % (attempt, e))
            time.sleep(10)
        self.log("[云端裁判] 5 分钟仍未就绪，停用（回落本地）")

    # ---------- 排序 ----------

    def score(self, code, ctx_text, cands):
        """云端重排候选，返回 {词: score}。score = -ln(rank+1)，大=更可能；
        未入模型排序名单的词给最低分（排在全部已排序候选之后）。"""
        if not self.ready or not cands:
            return {}
        ctx_key = (ctx_text or "")[-32:]
        ck = (ctx_key, code)
        with self._lock:
            if ck in self._cache:
                return self._cache[ck]
        ordered = self._call(cands, ctx_text, code)
        scores = {}
        if ordered:
            for i, w in enumerate(ordered):
                if w in cands:
                    scores[w] = -math.log(i + 1)
        base_min = -math.log(len(cands) + 1) if cands else 0.0
        for w in cands:
            scores.setdefault(w, base_min)
        with self._lock:
            if ck not in self._cache:
                self._cache_order.append(ck)
            self._cache[ck] = scores
            if len(self._cache_order) > 256:
                old = self._cache_order.pop(0)
                self._cache.pop(old, None)
        return scores

    def _call(self, cands, ctx_text, code):
        """发一次排序请求，返回模型排序后的候选列表（失败返回空）。"""
        try:
            payload = {
                "model": self._model,
                # seed-2.0 思考模型默认带 reasoning：思考过程把 max_tokens 耗尽、
                # chunk 间隔 250ms/行，非流式 25s 都等不完。关掉后 1382ms 秒回
                # （2026-09-07 探针钉死：no-thinking 非流式 1382ms 完整出答案）。
                "thinking": {"type": "disabled"},
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT.format(
                        ctx=(ctx_text or "")[-120:] or "（无）",
                        code=code, cands=json.dumps(cands, ensure_ascii=False))},
                ],
                "temperature": 0.0,
                # 24 个候选的 JSON 数组 ~100 token：max_tokens 越大模型越可能
                # 拖尾拉长响应（生成通道 1567ms 案同因）。120 足够且更快。
                "max_tokens": 120,
                "stream": False,
            }
            resp = _post_json(
                self._base + "/chat/completions", payload,
                headers={"Authorization": "Bearer " + self._key},
                timeout=self._timeout)
            text = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content", "")
            out = _parse_array(text)
            if out:
                self._fail = 0
            return out
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            self._fail += 1
            if self._fail >= 3:  # 连续失败 → 自降级，引擎回落本地裁判
                self.ready = False
                self.log("[云端裁判] 连续失败自降级（回落本地）: %r" % e)
            return []

    # ---------- 异步请求（与 QwenJudge 同构） ----------

    def request(self, code, ctx_text, cands):
        """排序通道：从候选池里挑最可能（判别式，豆包第二层）。

        名单须含全量动态真词池（包子静态 33 位案：截断=云端没机会）；
        模型只回 top10，输出短、回得快。"""
        if self.ready and cands:
            # 整句优先占名额：整句是裁判核心价值
            sents = [c for c in cands if len(c) >= 5]
            words = [c for c in cands if len(c) < 5]
            sents = sents[: self.max_cand // 2]
            words = words[: max(0, self.max_cand - len(sents))]
            self._q.put(("rank", code, (ctx_text or "")[-32:], sents + words))

    def request_generate(self, code, ctx_text):
        """生成通道：直接写用户最可能想打的候选（豆包式首选）。"""
        if self.ready and code:
            self._q.put(("gen", code, (ctx_text or "")[-32:], None))

    def _worker(self):
        while True:
            kind, code, ctx_text, cands = self._q.get()
            # 只处理最新请求：连击时中间码的生成/精排没有意义
            try:
                while True:
                    kind, code, ctx_text, cands = self._q.get_nowait()
            except queue.Empty:
                pass
            try:
                t0 = time.perf_counter()
                if kind == "gen":
                    items = self.generate(code, ctx_text)
                    ms = round((time.perf_counter() - t0) * 1000)
                    if items and self.on_generate:
                        self.on_generate(code, items, ms)
                else:
                    scores = self.score(code, ctx_text, cands)
                    ms = round((time.perf_counter() - t0) * 1000)
                    if self.on_result:
                        self.on_result(code, scores, ms)
            except Exception as e:
                self.log("[云端裁判] 处理异常已兜底: %r" % e)

    # ---------- 生成首选（豆包式：只给码+上文，直接写答案） ----------

    def generate(self, code, ctx_text):
        """生成用户最可能想打的 1~3 个候选。不喂候选池——池里没有的句子
        （简拼整句）排序通道永远出不来，生成通道可以。"""
        if not self.ready or not code:
            return []
        ctx_key = (ctx_text or "")[-32:]
        ck = ("gen", ctx_key, code)
        with self._lock:
            if ck in self._cache:
                return self._cache[ck]
        items = self._call_gen(code, ctx_text)
        items = [w for w in items if w]  # 防空串
        with self._lock:
            if ck not in self._cache:
                self._cache_order.append(ck)
            self._cache[ck] = items
            if len(self._cache_order) > 256:
                old = self._cache_order.pop(0)
                self._cache.pop(old, None)
        return items

    def _call_gen(self, code, ctx_text):
        try:
            payload = {
                "model": self._model,
                # 同 _call：关思考才秒回（探针 2026-09-07 钉死）
                "thinking": {"type": "disabled"},
                "messages": [
                    {"role": "system", "content": GEN_SYSTEM},
                    {"role": "user", "content": GEN_PROMPT.format(
                        ctx=(ctx_text or "")[-120:] or "（无）", code=code)},
                ],
                "temperature": 0.0,
                # 1~3 个短候选用不了 200 token：max_tokens 越大，模型越可能
                # 拖尾生成多余内容拉长响应（1567ms 案）。80 足够且更快。
                "max_tokens": 80,
                "stream": False,
            }
            resp = _post_json(
                self._base + "/chat/completions", payload,
                headers={"Authorization": "Bearer " + self._key},
                timeout=self._timeout)
            text = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content", "")
            out = _parse_array(text)
            if out:
                self._fail = 0
            return out
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            self._fail += 1
            if self._fail >= 3:  # 连续失败 → 自降级，引擎回落本地裁判
                self.ready = False
                self.log("[云端裁判] 连续失败自降级（回落本地）: %r" % e)
            return []
