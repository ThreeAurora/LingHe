# -*- coding: utf-8 -*-
"""AI 预测引擎。

职责：
1. 端点自动探测（"自动调配"）：
   Ollama(:11434) -> LM Studio(:1234) -> 其他 OpenAI 兼容本地端点 -> config 手填 API。
   探测到 Ollama 时自动取模型列表；config.ai.model 非空则强制指定。
2. 异步预测：单 worker 线程 + "只保留最新请求"队列 —— 用户继续打字时旧请求
   自然作废（天然防抖与竞态取消）；结果按 (上下文键, 键入码) 缓存。
3. 输出：候选字词/短句列表，按可能性降序。

零第三方依赖：HTTP 用 urllib。
"""

import json
import queue
import re
import threading
import urllib.error
import urllib.request

SYSTEM_PROMPT = (
    "你是中文输入法引擎。根据上下文和键入拼音，输出用户此刻最想输入的中文候选"
    "（单字/词语/短句），按可能性降序 3~6 项。只输出一个 JSON 数组，形如"
    " [\"候选\",\"候选\"]，禁止拼音、字母和解释。\n"
    "若拼音解读是「整句声母简拼」（每字取声母键），把它还原成完整句子作为第一候选。\n"
    "示例1——上文：他跑得；键入码 hk（拼音 huai）：输出 [\"很快\",\"好快\",\"怀\"]\n"
    "示例2——上文：（无）；键入码 ulpb（拼音 shuang pin）：输出 [\"双拼\",\"双频\"]\n"
    "示例3——上文：（无）；键入码 wilygbz（声母简拼 w i l y g b z）：输出 [\"我吃了一个包子\"]"
)


def _post_json(url, payload, headers=None, timeout=2.0):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def _get_json(url, timeout=0.8, headers=None):
    req = urllib.request.Request(url, method="GET")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", "replace"))


def parse_candidates(text: str):
    """从模型输出中尽力提取候选数组。"""
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
    # 降级：按行/顿号切
    parts = re.split(r"[\n、，,;；]+", text.strip().strip("```").strip())
    return [p.strip().lstrip("0123456789.、- )：:") for p in parts if len(p.strip()) >= 1][:8]


class AIEngine:
    def __init__(self, cfg, log=print):
        self.cfg = cfg
        self.log = log
        self.ready = False
        self.endpoint_desc = "未就绪"
        self._base = ""       # 形如 http://localhost:11434 或 http://x/v1
        self._kind = ""       # "ollama" | "openai"
        self._model = ""
        self._headers = {}
        self._timeout = float(cfg.get("timeout", 1.8))
        self._maxcand = int(cfg.get("max_candidates", 8))
        self._temp = float(cfg.get("temperature", 0.2))
        self._cache = {}                       # (ctx_key, code) -> [cands]
        self._cache_order = []
        self._job = None                       # (seq, code, ctx_key, context)
        self._seq = 0
        self._cv = threading.Condition()
        self._lock = threading.Lock()
        self.on_result = None                  # 回调：预测返回后通知 UI 刷新
        self._worker = threading.Thread(target=self._work_loop, daemon=True)
        self._worker.start()

    # ---------- 端点探测 ----------

    def probe(self):
        threading.Thread(target=self._probe_impl, daemon=True).start()

    def _probe_impl(self):
        ai = self.cfg
        mode = ai.get("mode", "auto")
        candidates = []
        if mode in ("auto", "local"):
            for url in ai.get("local_urls", ["http://localhost:11434", "http://localhost:1234"]):
                url = url.rstrip("/")
                try:
                    # Ollama?
                    tags = _get_json(url + "/api/tags", timeout=0.6)
                    models = [m.get("name") or m.get("model") for m in tags.get("models", [])]
                    models = [m for m in models if m]
                    if models:
                        candidates.append((url, "ollama", models))
                        break
                except (urllib.error.URLError, OSError, ValueError):
                    pass
                try:
                    # OpenAI 兼容（LM Studio / vLLM / llama.cpp server…）
                    models = _get_json(url + "/v1/models", timeout=0.6).get("data", [])
                    ids = [m.get("id") for m in models if m.get("id")]
                    if ids:
                        candidates.append((url + "/v1", "openai", ids))
                        break
                except (urllib.error.URLError, OSError, ValueError):
                    pass
        if mode in ("auto", "api") and not candidates:
            base = ai.get("api_base", "").strip()
            if base:
                base = base.rstrip("/")
                if not base.endswith("/v1"):
                    base += "/v1"
                headers = {}
                if ai.get("api_key"):
                    headers["Authorization"] = "Bearer " + ai["api_key"]
                candidates.append((base, "openai", ai.get("model", "")))

        if not candidates:
            self.endpoint_desc = "未发现 AI 端点（纯码表模式）"
            self.log("[AI] " + self.endpoint_desc)
            return

        base, kind, models = candidates[0]
        want = ai.get("model", "").strip()
        self._base, self._kind = base, kind
        self._model = want if want else (models[0] if isinstance(models, list) and models else "")
        # 指定模型不存在时回退到端点第一个可用模型
        if isinstance(models, list) and models and self._model not in models:
            self._model = models[0]
        if kind == "openai" and ai.get("api_key"):
            self._headers["Authorization"] = "Bearer " + ai["api_key"]
        self.ready = True
        tag = "本地" if "localhost" in base or "127.0.0.1" in base else "远程"
        self.endpoint_desc = "%s %s 模型=%s" % (tag, base, self._model)
        self.log("[AI] 就绪：" + self.endpoint_desc)

    # ---------- 请求调度 ----------

    def request(self, code: str, context: str):
        """发起/更新预测请求。只保留最新一个，旧的直接作废。"""
        if not code:
            return
        with self._cv:
            self._seq += 1
            ctx_key = context[-32:] if context else ""
            self._job = (self._seq, code, ctx_key, context)
            self._cv.notify()

    def _work_loop(self):
        while True:
            with self._cv:
                while self._job is None:
                    self._cv.wait()
                job, self._job = self._job, None
                seq, code, ctx_key, context = job
            # 缓存命中就不用打模型
            got = self._cache_get(ctx_key, code)
            if got is None and self.ready:
                got = self._call_model(code, context)
                if got:
                    self._cache_put(ctx_key, code, got)
            if got and self.on_result:
                self.on_result(seq)

    def _call_model(self, code: str, context: str):
        from xiaohe import interpret
        interps = interpret(code)
        inter_txt = "、".join(p for p, d in interps) or code
        user = "上文：%s\n键入码 %s（拼音：%s）：JSON数组=" % (
            context[-120:] if context else "（无）", code, inter_txt)
        payload_msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user}]
        try:
            if self._kind == "ollama":
                resp = _post_json(
                    self._base + "/api/chat",
                    {"model": self._model, "messages": payload_msgs, "stream": False,
                     "options": {"temperature": self._temp, "num_predict": 100}},
                    timeout=self._timeout)
                text = (resp.get("message") or {}).get("content", "")
            else:
                resp = _post_json(
                    self._base + "/chat/completions",
                    {"model": self._model, "messages": payload_msgs,
                     "temperature": self._temp, "max_tokens": 120, "stream": False},
                    headers=self._headers, timeout=self._timeout)
                text = ((resp.get("choices") or [{}])[0].get("message") or {}).get("content", "")
            cands = parse_candidates(text)[: self._maxcand]
            if cands:
                self.log("[AI] %s -> %s" % (code, " ".join(cands[:4])))
            return cands
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            self.log("[AI] 请求失败：%s" % e)
            return []

    def peek(self, ctx_key: str, code: str):
        """主引擎同步查询 AI 候选（仅读缓存，绝不阻塞按键线程）。"""
        return self._cache_get(ctx_key, code) or []

    # ---------- 缓存 ----------

    def _cache_get(self, ctx_key, code):
        with self._lock:
            return self._cache.get((ctx_key, code))

    def _cache_put(self, ctx_key, code, cands):
        with self._lock:
            key = (ctx_key, code)
            if key not in self._cache:
                self._cache_order.append(key)
            self._cache[key] = cands
            if len(self._cache_order) > 512:
                old = self._cache_order.pop(0)
                self._cache.pop(old, None)
