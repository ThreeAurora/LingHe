# -*- coding: utf-8 -*-
"""灵鹤 LingHe —— 外挂式 AI 中文输入法（小鹤音形向）。

设计要点：
- 外挂模式：WH_KEYBOARD_LL 低级钩子 + 自绘候选窗 + SendInput 上屏，
  免安装免签名，单目录绿色运行。
- 候选排序公式（核心约定）：
      候选 = [码表命中（固频序，共 k 个）] + [AI 预测（可能性降序）]
  码表命中 0 个 -> AI 从第 1 位起；命中 1 个 -> AI 从第 2 位起；以此类推。
  已背熟的音形码永远压住 AI，AI 只做兜底与扩展。
- AI：本地模型自动探测（Ollama/LM Studio 等 OpenAI 兼容端点），也可手填 API；
  智能读取上下文（含本外挂自己上屏的历史），异步预测不打断击键。
"""

import argparse
import ctypes
import ctypes.wintypes as wt
import json
import os
import queue
import sys
import time
from collections import deque

import tkinter as tk

from ai_engine import AIEngine
from ai_llm_judge import QwenJudge
from cloud_judge import CloudJudge
from candidate_ui import CandidateWindow
from dict_engine import DictEngine
from fuma import FuMa
from mabiao import MaBiao
from neural_rerank import NeuralReranker
from ai_llm_judge import QwenJudge
from rerank import StatReranker
from xiaohe import decode_syllable
import caret_ctx
import parser as key_parser

# ---------------- WinAPI ----------------

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.GetTickCount64.restype = ctypes.c_uint64

WH_KEYBOARD_LL = 13
WM_KEYDOWN = 0x0100
WM_KEYUP = 0x0101
WM_SYSKEYDOWN = 0x0104
LLKHF_INJECTED = 0x10

# 测试开关：端到端自动化验收需要注入按键驱动真实钩子链路。默认关闭——
# 生产环境注入键照旧放行不处理，行为与旧版完全一致。
_ACCEPT_INJECTED = bool(os.environ.get("LINGHE_ACCEPT_INJECTED"))

VK_BACK, VK_ESCAPE, VK_SPACE, VK_RETURN = 0x08, 0x1B, 0x20, 0x0D
VK_SHIFT, VK_CONTROL, VK_MENU = 0x10, 0x11, 0x12
VK_LSHIFT, VK_RSHIFT = 0xA0, 0xA1  # 低级钩子对 Shift 上报左右原始码，不折叠成 0x10
VK_LWIN, VK_RWIN = 0x5B, 0x5C
VK_TAB = 0x09
VK_L = 0x4C
VK_OEM_1 = 0xBA    # ; :
VK_OEM_7 = 0xDE    # ' "
VK_OEM_MINUS = 0xBD  # - _（上一页）
VK_OEM_PLUS = 0xBB   # = +（下一页）
VK_CAPITAL = 0x14    # CapsLock（大写锁定）

PUNCT_VKS = {
    0xBC,  # , <
    0xBE,  # . >
    0xBF,  # / ?
    0xDB, 0xDC, 0xDD,  # [ \
}

# 自动造词护栏：语料里几乎不连用的字（结构助词/语气词）。含任一字的组合
# 不是词，是句子碎片（「我吃了」「我的书」）。注意只拦这些——「我想」这类
# 名词+动词组合是正经词（主人拍板：用户会重复打的组合就该一键上屏，
# 语言学上是不是词不重要）。
COIN_BLOCK = set("的地得了吗呢吧啊呀哦呗嘛啦咯喽嗯哪哇噢喔哟哦呦")

COIN_MAX = 5      # 造词最长字数（专名/短语上限）
COIN_MULTI_MAX = 50000  # 混合模式里多字词片段的词频上限：超过算「句子里的
                        # 常用词」（东西/测试/一下），不算专名词头（试作）

# 高频功能字（2026-09-07 主人「穗心」案）：尾 2 字组合里只要含任何一个，
# 就是句子流顺手逐字打的（我|打、个|一、就|是、不|会），不是专名——不造。
# 专名（穗心/张阔/武媚娘）的两个字都不在功能字表里，照常造。
COIN_COMMON = set("的了是我你一不就都在和也他她它咱们俩有出着过被把让向从到对就这那还又要于与及各等给可没上正来去说想看看听问叫找打知道能会该要和得到我吗您请谢谢。，！？、、；：—…·")

# 候选装配参数（2026-09-07 主人 jylwviwu「菌类植物」案定案）：
WORD_BOOST = 9.0         # 全码整词提权：用户打满整词音节（菌类植物 4 音对
                         # 4 音），这个词就是目标本身——低频真词（-12.6/音节）
                         # 要压过高频虚词拼装的伪句（-14.4 更省，见 probe_junlei），
                         # 「先出词、后拼句」是输入法的本分。
LONGWORD_MIN_A = 40      # 3~4 音节真词进第一屏的最低权重闸。万象词库把
                         # 「菌类职务/菌类织物/菌类之屋」这类自然二连字垃圾
                         # 组合收成词，简拼全码召回归来一大把——低权长词
                         # 不给前排坑位，深水翻页可见。
EV_SENT_GOOD = 0.55      # 句子证据分阈值（StatReranker._eval_sent）：
                         # >= 视为真句（进前排/裁判名单），< 沉深水（不送裁判）。
                         # 0.4 太松：3gram 碎片句（就有了我这车晚上 0.40 /
                         # 就有了我这成为时 0.48）同分混进真句（我吃了一个
                         # 包子 0.55 / 今天天气不错 0.65）——碎片句 rate≈0.5
                         # +best3 恰好卡 0.4，升闸即清零。


def coin_pick(streak, seen, de):
    """连续上屏片段链 → 造词判定（纯函数，便于探针单测）。

    streak: [(片段, 拼音), ...] 连续上屏的单字/多字词。
    seen: {拼接串: 已连续出现次数}（重复模式用，函数内递增）。
    de: DictEngine（word_py 查存在性、weight 查词频）。

    返回 (词, 拼音串) 或 None。从最长组合往下试，命中即止：
      1. 纯单字 2~5 字 → 造（我想/张布斯/试作古华——词库没有就造，
         用户用辅码逐字打出来的生僻串几乎必是专名或高频自用组合）；
         其中 2 字组合只在整链恰为 2 片段时参与——长链的尾 2 字是
         长组合的中间态（东西|个|一 的「个一」），不是用户的意图词；
      2. 低频多字词+单字混合 → 造（试作|古|华：词头生僻=专名特征；
         东西|测试|一下 这种全高频词组合是句子，不造）；
      3. 同拼接串第二次连续出现 → 造（重复是用脚投票的最强信号，
         兜底覆盖含常用词的混合串）。

    判定时机由调用方控制：**只在断链事件**（标点/多字词边界/Esc/切窗）
    调用——逐字 push 时判会造出「试作古」这种中间垃圾词并清链，
    「试作古华」就永远造不出来了。

    链头检查：链首是高频多字词（weight≥5万）时整链视为正常句子流
    （东西|个|一|下 是「东西…一下」的打字过程，不是组专名）——「个一下」
    这种中间态不造。低频词头（试作 3270）才是组专名的信号。
    """
    if len(streak) < 2:
        return None
    head = streak[0][0]
    if len(head) >= 2 and de.weight(head) >= COIN_MULTI_MAX:
        return None
    for k in range(len(streak), 1, -1):
        frags = streak[-k:]
        s = "".join(w for w, _ in frags)
        n = len(s)
        if n > COIN_MAX:
            continue
        if de.word_py.get(s) or de.weight(s):
            return None  # 更长的组合已是词条，短组合被它覆盖，不再拆造
        if any(ch in COIN_BLOCK for ch in s):
            continue  # 含结构助词/语气词：句子碎片，试更短组合
        if any(not p for _, p in frags):
            return None  # 拼音不全（码表查无音的字）：防脏数据，放弃
        multi = [w for w, _ in frags if len(w) >= 2]
        singles = [w for w, _ in frags if len(w) == 1]
        seen_n = seen.get(s, 0)
        seen[s] = seen_n + 1
        ok = False
        if not multi:
            # 纯单字链：只造短组合（<=2 字）。聊天时逐字上屏的碎片
            # （原谅|我|我|打、浓茶|弄|成|有）是句子流不是专名——
            # 3 字以上纯单字链一律不自动造，只在重复出现时兜底造。
            # 尾 2 字组合单字含高频功能字（我|打、个|一）也不造。
            ok = n <= 2 or seen_n >= 1
            if ok and n == 2 and any(ch in COIN_COMMON for ch in s):
                ok = False
        elif singles and all(de.weight(m) < COIN_MULTI_MAX for m in multi):
            ok = True  # 生僻词头+单字：专名模式（试作|古|华）
        elif seen_n >= 1:
            ok = True  # 重复出现
        if ok:
            return s, " ".join(p for _, p in frags)
    return None


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", wt.DWORD), ("scanCode", wt.DWORD),
                ("flags", wt.DWORD), ("time", wt.DWORD),
                ("dwExtraInfo", ctypes.c_void_p)]


HOOKPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int, wt.WPARAM, wt.LPARAM)


class RECT(ctypes.Structure):
    _fields_ = [("left", wt.LONG), ("top", wt.LONG), ("right", wt.LONG), ("bottom", wt.LONG)]


class PT(ctypes.Structure):
    _fields_ = [("x", wt.LONG), ("y", wt.LONG)]


class GUITHREADINFO(ctypes.Structure):
    _fields_ = [("cbSize", wt.DWORD), ("flags", wt.DWORD),
                ("hwndActive", wt.HWND), ("hwndFocus", wt.HWND),
                ("hwndCapture", wt.HWND), ("hwndMenuOwner", wt.HWND),
                ("hwndMoveSize", wt.HWND), ("hwndCaret", wt.HWND),
                ("rcCaret", RECT)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", ctypes.c_void_p)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD),
                ("dwFlags", wt.DWORD), ("time", wt.DWORD), ("dwExtraInfo", ctypes.c_void_p)]


class HARDWAREINPUT(ctypes.Structure):
    _fields_ = [("uMsg", wt.DWORD), ("wParamL", wt.WORD), ("wParamH", wt.WORD)]


class _INPUTU(ctypes.Union):
    _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT), ("hi", HARDWAREINPUT)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("u", _INPUTU)]


INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_UNICODE = 0x0004

# 64 位下必须显式声明参数类型，否则指针参数溢出（int too long to convert）
user32.CallNextHookEx.argtypes = [ctypes.c_void_p, ctypes.c_int, wt.WPARAM, wt.LPARAM]
user32.CallNextHookEx.restype = ctypes.c_ssize_t
user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wt.HINSTANCE, wt.DWORD]
user32.SetWindowsHookExW.restype = wt.HHOOK
user32.UnhookWindowsHookEx.argtypes = [wt.HHOOK]
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetGUIThreadInfo.argtypes = [wt.DWORD, ctypes.POINTER(GUITHREADINFO)]
user32.ClientToScreen.argtypes = [wt.HWND, ctypes.POINTER(PT)]
user32.GetCursorPos.argtypes = [ctypes.POINTER(PT)]
user32.GetForegroundWindow.restype = wt.HWND
user32.SendInput.argtypes = [wt.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wt.UINT


def send_unicode(text: str):
    """以 Unicode 事件注入文本（对常规 Win32/浏览器/编辑器窗口有效）。"""
    units = []
    for ch in text:
        cp = ord(ch)
        if cp > 0xFFFF:
            cp -= 0x10000
            units += [0xD800 + (cp >> 10), 0xDC00 + (cp & 0x3FF)]
        else:
            units.append(cp)
    if not units:
        return
    arr = (INPUT * (len(units) * 2))()
    for i, u in enumerate(units):
        arr[i * 2].type = INPUT_KEYBOARD
        arr[i * 2].u.ki = KEYBDINPUT(0, u, KEYEVENTF_UNICODE, 0, None)
        arr[i * 2 + 1].type = INPUT_KEYBOARD
        arr[i * 2 + 1].u.ki = KEYBDINPUT(0, u, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP, 0, None)
    user32.SendInput(len(arr), arr, ctypes.sizeof(INPUT))


def key_down(vk):
    return bool(user32.GetAsyncKeyState(vk) & 0x8000)


def caret_pos():
    """取前台应用光标屏幕坐标；取不到则回退鼠标位置。"""
    info = GUITHREADINFO()
    info.cbSize = ctypes.sizeof(GUITHREADINFO)
    if user32.GetGUIThreadInfo(0, ctypes.byref(info)) and info.hwndCaret:
        p = PT(info.rcCaret.left, info.rcCaret.bottom)
        if user32.ClientToScreen(info.hwndCaret, ctypes.byref(p)):
            return p.x, p.y
    p = PT(0, 0)
    user32.GetCursorPos(ctypes.byref(p))
    return p.x, p.y


# ---------------- 引擎 ----------------

class Engine:
    def __init__(self, base_dir: str, cfg: dict, log=print):
        self.log = log
        self.cfg = cfg
        # ██ 状态字段全部最先创建：底库两段式加载的 worker 线程可能在
        # __init__ 完成前就回执 _on_dict_ready（mini 4~9s vs init 约 20s），
        # 缺 q/buffer/_cache_* 任何一个都会让回调崩溃、连带跳过全量热切换。
        self.q = queue.Queue()
        self.last_word = ""          # 上一个上屏词（bigram 学习与重排的上文）
        self.page = 0                # 候选翻页（-/=/Tab）
        self.n_pool = int(cfg.get("candidate_pool", 45))
        self._cache_code = None      # compute 结果缓存（同码串复用）
        self._cache_cands = None
        self._cache_nmb = 0
        self._cache_ctx = ""         # 缓存对应的上文（同码不同上文须重算）
        self.liaison = None          # 上屏联想词列表（豆包式：上屏后提示后续词）
        self.ctx_prevs = []          # 光标处上文词列表（tail_words 切出，最近在前；
                                     # 豆包「指哪打哪」：多上文衰减打分的输入）
        self.ctx_tail_text = ""      # 光标前原始文本尾部（神经重排的 MLM 输入）
        self._last_pos = (300, 300)  # 最近候选窗位置（异步刷新时复用）
        self._cache_pool_scores = {}  # 同池统计分缓存（神经精排的融合基底）
        self._cache_dynall = None     # 全量动态池（送裁判评分名单用，不截断）
        self._judge_ctx = ""          # 送裁时刻的上文（决定裁判 λ 是否放大）
        self._streak = []            # 连续上屏片段链（自动造词原料）：
                                     # 张|布|斯 →「张布斯」；试作|古|华 →「试作古华」
        self._streak_seen = {}       # 拼接串→连续出现次数（重复造词模式）
        self.buffer = ""
        self.cn_mode = True          # Shift 单击切换中/英
        self.shift_t0 = 0
        self.shift_alone = False
        self.last_fg = None
        self.context = deque(maxlen=int(cfg.get("context_max", 120)))
        self.stats = {"keys": 0, "eaten": 0, "commits": 0, "ai_hits": 0}
        self._hook = None
        self.max_len = int(cfg.get("max_buffer_len", 12))
        self.n_show = int(cfg.get("candidate_count", 9))

        self.mb = MaBiao()
        mdir = os.path.join(base_dir, cfg.get("mabiao_dir", "mabiao"))
        n = self.mb.load_dir(mdir)
        if n:
            log("[码表] %d 个词条，%d 个码位，加载 %dms" % (n, len(self.mb._table), round(self.mb.load_ms)))
        else:
            log("[码表] 目录为空（%s），纯 AI 模式" % mdir)

        self.fm = FuMa()
        nf = self.fm.load_dir(mdir)
        log("[辅码] %d 行（首键筛选），字数 %d" % (nf, len(self.fm._first)) if nf else "[辅码] 未加载")

        self.de = DictEngine(log=log)
        ddir = os.path.join(base_dir, "dicts")
        self.de.load_dir_async(ddir, on_done=self._on_dict_ready)

        # 端侧统计重排器（豆包第二层）：只重排底库候选，码表固频不参与
        self.rr = StatReranker(self.de, log=log)
        self.rr.load(ddir)

        self.ai = AIEngine(cfg.get("ai", {}), log=log)
        self.ai.on_result = lambda seq: self.q.put(("ai", seq))
        if cfg.get("ai", {}).get("enabled", True):
            self.ai.probe()
        else:
            self.ai.ready = False
            log("[AI] 已按配置停用（ai.enabled=false），纯静态模式")

        # 神经判别重排（RBT3 ONNX，2026-09-05 接棒 LLM）：击键仍由统计层即时
        # 响应，神经分 ~30-100ms 后异步到达并刷新候选顺序（豆包「候选自我
        # 修正」）。MLM 语义迁移能覆盖统计共现的天花板（打碎→易碎物）。
        ncfg = cfg.get("neural", {})
        self.nr = NeuralReranker(os.path.join(base_dir, ncfg.get("model_dir", "ai_neural/rbt3")),
                                 log=log)
        self.nr.on_result = self._on_nr_result
        self.neural_top_k = int(ncfg.get("top_k", 12))
        self.neural_lambda = float(ncfg.get("lambda", 1.0))
        self.prune_keep = int(ncfg.get("prune_keep", 20))
        if ncfg.get("enabled", True):
            self.nr.load_async()
        else:
            log("[神经] 已按配置停用（neural.enabled=false）")

        # 终审裁判两级：云端大模型（豆包式生成首选）→ 本地 Qwen2.5-0.5B。
        # 2026-09-07 主人「真正的智能」案定音：排序是判别式（候选池里挑，
        # 池里没有的永远出不来），生成是写答案（简拼整句首选只能靠生成）。
        # 配置了 api_key → 云端生成通道接管首选；否则回落本地判别裁判。
        # 回调与分数方向量纲完全兼容（每字 logP）。
        jcfg = cfg.get("judge", {})
        self.judge = QwenJudge(os.path.join(base_dir, jcfg.get("model_dir", "ai_llm/qwen25-05b-hf")),
                               log=log)
        self.judge.on_result = lambda code, scores, ms: self._on_neural(code, scores, ms, judge=True)
        self.judge.on_ready = self._on_backend_ready
        self.nr.on_ready = self._on_backend_ready
        self.cloud_judge = CloudJudge(jcfg, log=log)
        self.cloud_judge.on_generate = self._on_cloud_generate
        self.cloud_judge.on_ready = self._on_backend_ready
        if jcfg.get("enabled", True) and jcfg.get("api_key"):
            self.cloud_judge.load_async()
        elif jcfg.get("enabled", True):
            log("[云端裁判] 未配置 api_key，使用本地 QwenJudge")
            self.judge.load_async()
        else:
            log("[LLM裁判] 已按配置停用（judge.enabled=false）")

        self.buffer = ""
        self.enabled = True
        self._hook = None
        self._proc = HOOKPROC(self._hook_cb)

    def _on_dict_ready(self, ok):
        if ok:
            self._cache_code = None  # 底库上线，作废旧候选
            self.q.put(("flash", "[底库就绪 %d词]" % self.de.size))
            self.log("[底库] 后台加载完成：%d 词条" % self.de.size)
            if self.buffer:  # 加载窗口里打的码立即升级重算（主人 22:34 案）
                self._after_edit()

    def _on_backend_ready(self):
        """神经/裁判异步就绪（各自 worker 线程）：在屏旧码立即升级重判。

        启动加载窗口（底库 30-45s/裁判 35s）里打字的主人案：垃圾候选会
        永远停在屏上，除非再敲一键——后端一就绪就主动重算+重判。"""
        if not getattr(self, "buffer", ""):
            return
        self.q.put(("flash", "[智能终审就绪]"))
        self._after_edit()

    # ---- 候选组装：排序公式的唯一实现 ----
    def _fused_candidates(self, code):
        """辅码筛选词（最高优先）：多假设解析出 音节+辅码 解释，
        查码表+底库词后校验辅码（idx=第几字，-1=任意字），通过者按固频序。
        辅码作用于**所有词**（主人明确要求）：不止码表，152 万底库词同样可筛。"""
        if not self.fm.loaded or len(code) < 3:
            return []
        out, seen = [], set()
        for syls, fuses in key_parser.parse(code):
            if not fuses or len(syls) < 2:
                continue
            ss = "".join(syls)
            cands = list(self.mb.exact(ss)) if len(ss) <= 4 else []
            if self.de.loaded and len(syls) <= 4:
                # 底库同拼音词一并进入辅码筛选池。
                # 坑：syls 是**双拼键**（yi/dv），而底库 by_pinyin 索引是**全拼**
                # （yi dui），必须先 decode_syllable 转换。直接 join 会得到
                # "yi dv"，永远查不到任何词——「辅码作用于 152 万底库词」会静默
                # 失效，退化成只筛码表内的词（yidvg→一堆 案：码表只有 已对/乙队）。
                py = " ".join(decode_syllable(s) for s in syls)
                cands += self.de.lookup_pinyin(py, 30)
            for w in cands:
                if len(w) != len(syls) or w in seen:
                    continue
                ok = True
                for idx, key in fuses:
                    if idx >= 0:
                        if idx >= len(w) or not self.fm.word_match(w[idx], key):
                            ok = False
                            break
                    else:
                        if not any(self.fm.word_match(ch, key) for ch in w):
                            ok = False
                            break
                if ok:
                    out.append(w)
                    seen.add(w)
            if len(out) >= self.n_pool:
                break
        return out

    def compute(self, code):
        """候选装配（返回候选池，供翻页）。主人定约（2026-09-05）：

        1. 码表固频（辅码筛选 + 全码 exact + 前缀简码）永远在最前——肌肉记忆；
        2. 其后**任意码长**（1/2/3/4/5...）都是「底库词 + Viterbi 整句」同池，
           按「每音节平均代价」统一比价智能排序——词库里没有的组合，只要
           统计上最该出现就排前面（ibz→吃包子 案：整句「吃|包子」与词条
           「吃包子」同池竞争，谁代价小谁在前）；
        3. 同码串连续调用直接命中缓存（翻页/选字/AI 回包都不再重算）。
        """
        if code == self._cache_code and tuple(self.ctx_prevs) == self._cache_ctx \
                and self._cache_cands is not None:
            return self._cache_cands, self._cache_nmb
        n = len(code)
        fused = self._fused_candidates(code)
        mb_exact = list(self.mb.exact(code))
        seen = set(fused) | set(mb_exact)
        mb_hits = fused + mb_exact
        mb_prefix = []
        for w in self.mb.prefix(code, self.n_pool * 2):
            if w not in seen:
                mb_prefix.append(w)
                seen.add(w)
        # ---- 同池：底库词 + 整句，统一「每音节平均代价」度量 ----
        # 词的代价由 rerank.score_word 给出（unigram+用户+多上文），除以音节数；
        # 整句代价由 Viterbi 返回（已含词间 bigram），除以音节数。两者同参数
        # 同上下文，可直接比较——这是「智能排序」与「词句公平竞争」的核心。
        # 多上文：[一个, 吃了] 这样的尾部词列表交给 score_word 衰减打分，
        # Viterbi 首词仍只挂最近词（整句内部词序自带 bigram）。
        prevs = self.ctx_prevs or ([self.last_word] if self.last_word else [])
        prev = prevs[0] if prevs else ""
        pool = {}
        if self.de.loaded:
            dict_cands = []
            full_py = ""
            if n >= 2 and n % 2 == 0:
                full_py = " ".join(decode_syllable(code[i:i + 2])
                                   for i in range(0, n, 2))
                dict_cands += self.de.lookup_pinyin(full_py, self.n_pool)
            if n >= 2:
                # 简拼召回放宽到 2 键：bz→包子、mb→面包 是最高频场景，
                # 之前 n>=3 把它们全部挡在召回之外。2 键另需更深召回：
                # by_initial['b z'] 里 豹子 排 78（470 条含重复），默认
                # n_pool=45 的截断把它挡在池外（主人 追上了一个bz→豹子 案）。
                # 【2026-09-06 主人报案修复】去掉 not mb_exact 前置条件：
                # 码表二简字命中（bz 必有）不该挡掉底库简拼召回——原条件
                # 导致 2 键词池整个为空（动态区零候选），裁判无从复核，
                # 包子永远出不来。码表字仍居固频区最前，底库词跟在动态区，
                # 两区共存不冲突。
                dict_cands += self.de.lookup_initial(" ".join(code),
                                                     90 if n == 2 else self.n_pool)
            for w in dict_cands:
                if w in seen or w in pool:
                    continue
                pym = self.de.word_py.get(w) or ""
                m = max(1, len(pym.split()))
                c = self.rr.score_word(w, prevs) / m
                # 全码整词提权（2026-09-07 主人 jylwviwu「菌类植物」案）：
                # 用户打满整词音节（4 音码→4 音词）时这个词就是目标本身——
                # 押过后简拼召回里缩水的 2 音节词，更压过虚词拼装伪句。
                if n % 2 == 0 and full_py and m == n // 2:
                    c -= WORD_BOOST / m
                pool[w] = c
        if n >= 2:
            # 整句切分放开到任意码长：短码也要能预测词库外组合
            vpool = []
            if n % 2 == 0 and n >= 4:
                keys = [decode_syllable(code[i:i + 2]) for i in range(0, n, 2)]
                vpool += self.rr.viterbi(keys, "py", 8, ret_cost=True, prev=prev)
                # py 全码同样走锚定召回（2026-09-06 双拼全码批测案）：py
                # 模式原本只有 viterbi，书面长词链垄断的病一样存在
                if n >= 5:
                    vpool += self.rr.anchor_sentences(keys, "py", 20, prev=prev)
            vpool += self.rr.viterbi(list(code), "ini", 8, ret_cost=True, prev=prev)
            # 长词锚定召回：3~4 音节键串 top1 强制成句（前缀 top3 变体）。
            # 主 beam 的统计代价被「文件体现|出」类 4 字词条链垄断，口语串
            # （我今天想吃西湖醋鱼）全程被剪——锚定不与 beam 竞争，直接把
            # 「我今天想吃西湖醋鱼」们塞进池，排序交给神经裁决。
            if n >= 5:
                vpool += self.rr.anchor_sentences(list(code), "ini", 20, prev=prev)
                # 贪心最大匹配通道：目标句每段都是词库真实词（给你=7、
                # 东西=3、测试一下=1、西湖醋鱼=1），统计 beam 却被「文件
                # 体现|出」类书面长词链垄断（beam256 也召不回）——本通道
                # 不管代价只管用真实词填满键串，正反两向各产出一条骨架句，
                # 排序交伪似然终审（整句组纯神经排序置顶）
                vpool += self.rr.max_match_sentences(list(code), "ini")
                # 口语字链通道（2026-09-05 回归案 wilygbz 钉死）：词库/口语
                # 2gram 里 吃了/我吃/了个 全在，但 viterbi 书面代价、锚点
                # 书面门槛、max_match 的「外出旅游」贪心三路都召不回
                # 「我吃了一个包子」。真人打真实句子时逐键字链就是目标句
                # 本身——本通道只管召回（top6 整链进池），排序交 LLM 终审
                vpool += self.rr.char_chains(list(code), "ini", 6, prevs=prevs)
            vpool.sort(key=lambda t: t[1] / max(1, t[2]))
            for s, c, m in vpool:
                if s in seen:
                    continue
                cps = c / max(1, m)
                if s not in pool or cps < pool[s]:
                    pool[s] = cps  # 同串取更优代价（词条 vs 切分谁准谁上）
        ranked_pool = sorted(pool.items(), key=lambda kv: kv[1])
        # 词截断、整句豁免：骨架句（锚点串接/最大匹配）的统计均价结构性
        # 偏高——「的|乡村|是一项」类高频短词链每字便宜 ~2 nat，正是口语
        # 串的死因。截断会让目标句见不到神经终审（2026-09-05 实测钉死），
        # 句子全保留进池，排序交给 _on_neural 整句组纯神经裁决。
        # 码表固频区额度（2026-09-06 主人报案 bz→包子）：2 键简码时 prefix
        # 字上百个，不设限会吃满候选池、把底库智能词全部挤出（包子在
        # _cache_pool_scores 里却进不了 base），且 n_mb>池长让 _on_neural
        # 的 n_mb>=len(base) 守卫直接丢弃裁判结果。固频区封顶两页半。
        # 【2026-09-07 二修，主人报案「前 8 全是四字词」】未完成码（<4 键）的
        # prefix 是声母前缀大爆炸（bz→56 个四字成语 暴躁不安/不醉不归…），
        # 它们不是背过的码、没有肌肉记忆价值，只占固频区坑位。未完成码的
        # 固频区只收短词（<=2 字，真简码词 变/不在/不再 全保留）+ 封顶一页。
        if n <= 3:
            mb_part = (mb_hits + [w for w in mb_prefix if len(w) <= 2])[: 9]
        else:
            mb_part = (mb_hits + mb_prefix)[: 24]
        n_mb = len(mb_part)
        self._cache_pool_scores = dict(ranked_pool)  # 神经精排的统计基底
        # 全量动态池（不截断）：统计排名靠后的词（bz→包子 在 35 位）也能
        # 送进 LLM 判别名单——词可以先被统计层截断，但判别分高就得回流。
        self._cache_dynall = [w for w, _ in ranked_pool]
        # 裁判可评分名单（2026-09-07 piye 案二修 + jylwviwu 案三修）：
        # viterbi 拼字伪句（普查员嗯/教育了我正成为是，word_py 查无此串或
        # 无语料证据）是拼装垃圾，Qwen 0.5B 会给它们打分出虚幻的「通顺」高分
        # 霸占前排。名单=词库真词 + 证据分达标的整句（_eval_sent>=EV_SENT_GOOD），
        # 伪句/证据不足句一律不裁——裁判的核心价值在整句判别，短码靠统计层。
        real = lambda w: (w in self.de.word_py) or (
            len(w) >= 5 and not any(ch in COIN_BLOCK for ch in w)
            and self.rr._eval_sent(w) >= EV_SENT_GOOD)
        self._cache_dynscore = [w for w, _ in ranked_pool if real(w)]
        py_len = lambda w: len((self.de.word_py.get(w) or "").split())
        # 候选三档（2026-09-07 主人定案：伪句一个都不能有）：
        #   一档 短真词：1~2 音节且有拼音（变/包子/皮也）。viterbi 拼出的
        #        非词串（普查员嗯 这类 word_py 查无拼音）不算词，不进候选。
        #   二档 长真词：3~4 音节且 weight 过闸的真词（菌类植物）。低权长词
        #        （菌类职务/菌类织物/菌类之屋 这类自然二连字组合）不占坑位。
        #        py_len 限定 >=3：**杜绝与一档重复**（piye 案——僻野/屁也
        #        1~2 音节词曾同时进 short_words 与 heavy，屏上 6~9 位与 1~5
        #        位原样重复）。
        #   三档 真句：证据分达标的整句（词典整词=1.0，或口语 3gram 命中率高）。
        #        弱句（0<ev<EV_SENT_GOOD）与无拼音伪句（ev=0）**彻底不进候选**，
        #        不再翻页可见——打长码时屏上只有词和真话，不掺一句假。
        short_words = [w for w, _ in ranked_pool
                       if len(w) < 5 and 1 <= py_len(w) <= 2]
        long_words = [w for w, _ in ranked_pool
                      if len(w) < 5 and py_len(w) >= 3]
        heavy = [w for w in long_words
                 if self.de.weight(w) >= LONGWORD_MIN_A]
        cap_dyn = 48 if n == 2 else (self.n_pool - n_mb)
        dict_side = (short_words + heavy)[: max(cap_dyn, 16)]
        base = mb_part + dict_side
        seen_base = set(base)
        ev = lambda w: self.rr._eval_sent(w)
        sents_good = [w for w, _ in ranked_pool
                      if len(w) >= 5 and ev(w) >= EV_SENT_GOOD]
        base += [w for w in sents_good if w not in seen_base]
        # AI 顺延：静态侧命中 k 个，AI 从第 k+1 位起
        merged = base[:]
        for w in self.ai.peek(self.context_key(), code):
            if w not in merged:
                merged.append(w)
        merged = merged[: self.n_pool]
        self._cache_code = code
        self._cache_ctx = tuple(self.ctx_prevs)  # 上文变了结果必须重算（同码不同上文）
        self._cache_cands = merged
        self._cache_nmb = n_mb
        return merged, n_mb

    def context_key(self):
        return "".join(self.context)[-32:]

    # ---- 钩子回调（主线程 Tk pump 中被调用，必须快进快出）----
    def _hook_cb(self, ncode, wparam, lparam):
        try:
            return self._on_key(ncode, wparam, lparam)
        except Exception as e:  # 钩子里绝不抛异常，否则全局键盘卡死
            self.log("[钩子] 异常已兜底: %r" % e)
            return user32.CallNextHookEx(None, ncode, wparam, lparam)

    def _on_key(self, ncode, wparam, lparam):
        if ncode != 0:
            return user32.CallNextHookEx(None, ncode, wparam, lparam)
        msg = wparam
        info = ctypes.cast(lparam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
        vk = info.vkCode
        self.stats["keys"] += 1
        if info.flags & LLKHF_INJECTED and not _ACCEPT_INJECTED:
            return user32.CallNextHookEx(None, ncode, wparam, lparam)  # 自己注入的，放行

        # Shift 单击：组码中=上屏已敲的英文（搜狗/微软惯例）；空码=中英切换。
        # 注意 LL 钩子 vkCode 是 VK_LSHIFT/VK_RSHIFT，不折叠成 VK_SHIFT。
        if vk in (VK_SHIFT, VK_LSHIFT, VK_RSHIFT):
            if msg in (WM_KEYDOWN, WM_SYSKEYDOWN):
                self.shift_t0 = kernel32.GetTickCount64()
                self.shift_alone = True
            elif msg == WM_KEYUP and self.shift_alone:
                self.shift_alone = False
                if self.enabled and kernel32.GetTickCount64() - self.shift_t0 <= 400:
                    if self.buffer:
                        raw = self.buffer
                        self.buffer = ""
                        self.page = 0
                        self.q.put(("hide",))
                        self.stats["commits"] += 1
                        self._streak = []  # 英文上屏：断造词链（不记录英文）
                        send_unicode(raw)
                    else:
                        self.cn_mode = not self.cn_mode
                        self.liaison = None
                        self.q.put(("hide",))
                        self.q.put(("flash", "[中]" if self.cn_mode else "[EN]"))
                        self.log("[模式] " + ("中文" if self.cn_mode else "英文"))
            return user32.CallNextHookEx(None, ncode, wparam, lparam)  # Shift 永远放行
        if msg in (WM_KEYDOWN, WM_SYSKEYDOWN):
            self.shift_alone = False  # 期间按了别的键 → 不是单击

        ctrl, alt = key_down(VK_CONTROL), key_down(VK_MENU)
        if ctrl and alt and vk == VK_L:
            self.enabled = not self.enabled
            self.buffer = ""
            self.q.put(("hide",))
            self.log("[开关] " + ("已启用" if self.enabled else "已暂停 (Ctrl+Alt+L 恢复)"))
            return 1

        if not self.enabled:
            return user32.CallNextHookEx(None, ncode, wparam, lparam)
        if msg not in (WM_KEYDOWN, WM_SYSKEYDOWN):
            return user32.CallNextHookEx(None, ncode, wparam, lparam)
        # `~ 键完全透传：不参与组码/清码/上屏任何处理（2026-09-07 报案：
        # 按 ~ 候选窗消失=旧逻辑把它当组码标点触发 hide）。想输出 ~ 就输出。
        if vk == 0xC0:
            return user32.CallNextHookEx(None, ncode, wparam, lparam)
        # 组合键/系统键组合一律放行：Alt+字母 以 WM_SYSKEYDOWN 上报（PowerToys
        # 的 alt+wasd→方向键全靠这类事件，输入法不得截胡）。GetAsyncKeyState
        # 异步状态在钩子回调里偶发漏检（alt+a 打出 aaaa 案），事件本身的
        # 系统语义更可靠。CapsLock 大写锁定时整体放行：目标应用自管大小写，
        # 输入法不拦截不转码（中文输入法惯例）。
        if msg == WM_SYSKEYDOWN or ctrl or alt \
                or key_down(VK_LWIN) or key_down(VK_RWIN) \
                or bool(user32.GetKeyState(VK_CAPITAL) & 0x01):
            if self.buffer:
                self.buffer = ""
                self.q.put(("hide",))
            return user32.CallNextHookEx(None, ncode, wparam, lparam)

        is_letter = 0x41 <= vk <= 0x5A
        is_digit = 0x30 <= vk <= 0x39
        fg = user32.GetForegroundWindow()
        if fg != self.last_fg:
            self.last_fg = fg
            self.context.clear()  # 换窗口=换语境
            self.ctx_prevs = []   # 上下文也随窗口失效
            self.ctx_tail_text = ""
            self._streak_clear()  # 造词链同样随窗口失效（先结算再清）
            self.liaison = None
            self.q.put(("hide",))

        if not self.cn_mode:  # 英文态：全部透传
            return user32.CallNextHookEx(None, ncode, wparam, lparam)

        # 联想词生命周期：被空格（上屏首选）或数字（选第 n 个）消费，
        # 其他任何键都取消。数字键不放行到这里，否则联想会被提前清掉。
        if self.liaison and vk != VK_SPACE and not is_digit:
            self.liaison = None
            if not self.buffer:
                self.q.put(("hide",))

        if is_letter:
            if key_down(VK_SHIFT):  # 大写意图：放弃组码，透传
                if self.buffer:
                    self.buffer = ""
                    self.q.put(("hide",))
                return user32.CallNextHookEx(None, ncode, wparam, lparam)
            if not self.buffer:
                # 新码开始：读一次光标处上下文（每组码只读一次，摊薄开销）。
                # 这是豆包「指哪打哪」的关键——同样的键串在不同上文下应出不同
                # 结果。tail_words 切出最后 3 个词（限长 2 保住动词），供
                # 多上文衰减打分；同码不同上文会触发重算（缓存键含上文）。
                try:
                    before, _after = caret_ctx.read_context(before=64, after=0)
                    self.ctx_prevs = caret_ctx.tail_words(
                        before, self.de.word_py if self.de.loaded else None, 3)
                    self.ctx_tail_text = before[-32:]
                except Exception:
                    self.ctx_prevs = []
                    self.ctx_tail_text = ""
            if len(self.buffer) < self.max_len:
                self.buffer += chr(vk).lower()
            # 第 5 码起继续组码（纯双拼续词）；超 max_len 后吞键防漏字母
            self._after_edit()
            return self._eat()

        if vk == VK_BACK:
            if self.buffer:
                self.buffer = self.buffer[:-1]
                self._after_edit()
                return self._eat()
            return user32.CallNextHookEx(None, ncode, wparam, lparam)

        if vk == VK_SPACE:
            if self.buffer:
                cands, n_mb = self.compute(self.buffer)
                if cands:
                    self._commit(cands[0], 0, n_mb)
                else:
                    self.buffer = ""
                    self.q.put(("hide",))
                return self._eat()
            if self.liaison:  # 联想态：空格上屏首选联想词（并继续联想下一词）
                w = self.liaison[0]
                self.liaison = None
                if self.de.loaded:
                    self.de.remember(w, self.de.word_py.get(w, ""))  # 联想上屏也调频
                self._commit(w)
                return self._eat()
            return user32.CallNextHookEx(None, ncode, wparam, lparam)

        if is_digit and not self.buffer and self.liaison:
            # 联想态下数字键 = 选第 n 个联想词（豆包式），同样调频并续联想
            idx = vk - 0x31
            if 0 <= idx < len(self.liaison):
                w = self.liaison[idx]
                self.liaison = None
                if self.de.loaded:
                    self.de.remember(w, self.de.word_py.get(w, ""))
                self._commit(w)
                return self._eat()

        if is_digit and self.buffer:
            idx = vk - 0x31
            cands, n_mb = self.compute(self.buffer)
            idx += self.page * self.n_show  # 数字选字作用于当前页
            if idx < len(cands):
                self._commit(cands[idx], idx, n_mb)
            return self._eat()

        # 翻页：=/Tab 下一页（末页回卷），- 上一页；无组码时透传
        if self.buffer and vk in (VK_OEM_PLUS, VK_OEM_MINUS, VK_TAB):
            cands, _ = self.compute(self.buffer)
            n_pages = max(1, (len(cands) + self.n_show - 1) // self.n_show)
            if vk == VK_OEM_MINUS:
                self.page = max(0, self.page - 1)
            else:
                self.page = 0 if self.page + 1 >= n_pages else self.page + 1
            return self._eat()

        if vk in (VK_OEM_1, VK_OEM_7) and self.buffer:
            # 分号=2选，单引号=3选（音形重码选择键惯例，作用于当前页）
            idx = 1 if vk == VK_OEM_1 else 2
            cands, n_mb = self.compute(self.buffer)
            idx += self.page * self.n_show
            if len(cands) > idx:
                self._commit(cands[idx], idx, n_mb)
            return self._eat()  # 无对应候选时吞键忽略，;/' 不漏进目标窗口

        if vk == VK_ESCAPE and self.buffer:
            self.buffer = ""
            self._streak_clear()  # 放弃组码：链先结算再清
            self.q.put(("hide",))
            return self._eat()

        if vk == VK_RETURN and self.buffer:
            # 回车 = 上屏英文（字母原文直进目标应用，回车本身吞掉不换行）
            raw = self.buffer
            self.buffer = ""
            self.q.put(("hide",))
            self.stats["commits"] += 1
            self._streak = []  # 英文上屏：断造词链（不记录英文）
            send_unicode(raw)
            return self._eat()

        if self.buffer and vk in PUNCT_VKS:
            cands, n_mb = self.compute(self.buffer)
            if cands:  # 组码中标点：上屏首选，随后标点照常进应用
                self._commit(cands[0], 0, n_mb)
            self._streak_clear()  # 标点=句子边界：链先结算再清
            return user32.CallNextHookEx(None, ncode, wparam, lparam)

        if self.buffer:
            self.buffer = ""
            self.q.put(("hide",))
        return user32.CallNextHookEx(None, ncode, wparam, lparam)

    def _on_nr_result(self, code, scores, ms):
        """RBT3 返回分流（worker 线程）：裁判在位＝两级裁决的粗筛级，整句按
        RBT3 分取 top prune_keep 转送裁判终审；裁判不在位＝兜底精排（整句组
        保持统计序，RBT3 的新闻语料偏差不得裁决整句，见 _on_neural）。"""
        if self.judge.ready:
            if self.buffer != code or not scores:
                return
            sents = sorted(((v, w) for w, v in scores.items() if len(w) >= 5),
                           reverse=True)
            words = sorted(((v, w) for w, v in scores.items() if len(w) < 5),
                           reverse=True)
            keep = [w for _, w in sents[:self.prune_keep]] + \
                   [w for _, w in words[:self.prune_keep]]
            self.judge.request(code, self.ctx_tail_text, keep)
        else:
            self._on_neural(code, scores, ms, judge=False)

    def _on_neural(self, code, scores, ms, judge=True):
        """精排返回（worker 线程）：统计分 + λ·每字logP 重排动态区，刷新当前页。

        只重排池内已有的候选（MLM 打分是判别式的：不生成、只裁决）。
        buffer 已变（用户继续击键/已上屏）则丢弃本包结果。

        回流（2026-09-07 bz→包子 案）：评分名单已扩至全量动态池，base 外
        的判别高分词组（包子 统计 35 位被截断、LLM 判第 1）按融合分与池内
        词组同场排序，自然回流进动态区前部——「统计层管召回，裁判管裁决」。
        """
        if self.buffer != code or not scores:
            return
        base = self._cache_cands
        n_mb = self._cache_nmb
        if not base or n_mb >= len(base):
            return
        # 有明确上文时裁判主导（λ 放大）：判别式整句 logP 在「我吃了一个+bz」
        # 下对 包子 是决定性的（-6.09 vs 不再 -7.58，λ>1.05 才翻盘，1.5 留
        # 余量）。无上文时 2 字词判别不可靠（诗仙案：来吧 -4.94 压过 李白），
        # λ 维持原值不动，避免乱序回归。
        lam = 1.5 if (judge and self._judge_ctx and len(self._judge_ctx) >= 2) \
            else self.neural_lambda
        tail = base[n_mb:]
        tail_set = set(tail)
        sents = [(scores[w], w) for w in tail
                 if w in scores and w in self._cache_pool_scores and len(w) >= 5]
        # 词组融合 + 回流：池内词组与评分名单里 base 外的词组（须有统计基底
        # 分可比）统一按融合分排序。被挤出槽位的池内原词仍保留（不消失）。
        fused = {}
        for w in tail:
            if len(w) < 5 and w in scores and w in self._cache_pool_scores:
                fused[w] = self._cache_pool_scores[w] - lam * scores[w]
        for w, s in scores.items():
            if len(w) >= 5 or w in tail_set or w not in self._cache_pool_scores:
                continue
            fused[w] = self._cache_pool_scores[w] - lam * s
        words = sorted(fused.items(), key=lambda kv: kv[1])
        cand_set = {w for _, w in words}
        rest = [w for w in tail if w not in cand_set]
        if judge:
            sents.sort(reverse=True)   # logP 越大（越接近 0）越通顺
            sents_ranked = [w for _, w in sents]
        else:
            sents_ranked = [w for w in tail
                            if w in self._cache_pool_scores and len(w) >= 5]
        cap = max(0, self.n_pool - n_mb - len(sents_ranked))
        # 词先句后（2026-09-07 主人「菌类植物」案）：整句绝不无条件压词。
        # 打全码（jylwviwu）时目标词 菌类植物 是被 WORD_BOOST 抬到动态区
        # 最前的——若裁判回流把整句组提到词前，全码真词反而被挤后。
        # 用户要的是「词和句都按常用度排」：词比句常用，词在前、真句在后。
        merged = base[:n_mb] + [w for w, _ in words[:cap]] + sents_ranked + rest
        self._cache_cands = merged
        self.q.put(("show", self.buffer, merged, n_mb, self._last_pos))
        self.log("[%s] %s 精排整句%d 词%d %dms" % (
            "裁判" if judge else "RBT3", code, len(sents), len(words), ms))

    def _on_cloud_generate(self, code, items, ms):
        """云端生成首选到达（豆包式）：直接写答案，不依赖候选池。

        与判别式精排的分水岭：判别只能在池里挑（池里没有的永远出不来），
        生成把「用户最可能想打的」写进候选窗。落位规则：
        - 码表固频区（n_mb 个，用户背熟的音形码）永远压顶，不被动；
        - 生成结果插在固频区之后、动态区之前——简拼/未全码时就是首选；
        - 已在动态区里的同名候选去重（避免候选窗重复）。
        buffer 已变（用户继续击键/已上屏）则丢弃本包结果。
        """
        if self.buffer != code or not items:
            return
        base = self._cache_cands
        n_mb = self._cache_nmb
        if not base or n_mb >= len(base):
            return
        # 全码/固频区占满时不插（用户打满码是确定性输入，码表说了算）
        head = base[:n_mb]
        rest = [w for w in base[n_mb:] if w not in items]
        merged = head + list(items) + rest
        self._cache_cands = merged
        self.q.put(("show", self.buffer, merged, n_mb, self._last_pos))
        self.log("[云端生成] %s → %s %dms" % (code, " / ".join(items), ms))

    def _eat(self):
        self.stats["eaten"] += 1
        return 1  # 吞掉，不让原始键进入目标应用

    def _after_edit(self):
        self.page = 0  # 组码内容变化 → 回第一页
        if not self.buffer:
            self.q.put(("hide",))
            return
        cands, n_mb = self.compute(self.buffer)
        pos = caret_pos()
        self._last_pos = pos
        if cands:
            self.q.put(("show", self.buffer, cands, n_mb, pos))
        else:
            self.q.put(("think", self.buffer, pos))  # 静态零命中，AI 在途
        # 精排直送裁判（2026-09-07 主人判死速度：「不可能等一个候选六秒」）：
        # 裁判上 GPU fp16 后全池 66 句一遍 ~0.3s——两级粗筛（RBT3 每键 2s）
        # 在稳态整体退役，裁判直接全量终审，整句零截断零粗筛。裁判未就绪
        # （启动加载窗口）才回退 RBT3 词组融合（整句组保持统计序，见
        # _on_neural 门禁）。异步不阻塞击键，worker 只留最新请求。
        if len(self.buffer) >= 2 and len(cands) > n_mb:
            self._judge_ctx = (self.ctx_tail_text or "").strip()
            # 评分名单用全量动态池：词可以先被统计截断（bz→包子 35 位被
            # dict_side 截断线挡掉），但不能被挡在裁判门外——判别分高的
            # 词由 _on_neural 回流进动态区前部。名单本身剔除语气词粘合
            # 伪句（_cache_dynscore），裁判不为垃圾背书。
            dyn_all = getattr(self, "_cache_dynscore", None) \
                or getattr(self, "_cache_dynall", None) or cands[n_mb:]
            if self.cloud_judge.ready:
                # 云端接管：只发生成首选请求（豆包式，_on_cloud_generate
                # 压顶动态区）；动态区顺序由 RBT3 兜底，不再双重精排。
                self.cloud_judge.request_generate(self.buffer, self.ctx_tail_text)
            elif self.judge.ready:
                self.judge.request(self.buffer, self.ctx_tail_text, dyn_all)
            elif self.nr.ready:
                self.nr.request(self.buffer, self.ctx_tail_text, dyn_all)
        # AI 只在长码（第 5 码起）时补位：短码静态侧（码表+底库+重排）已足够强，
        # 生成式 LLM 也物理上进不了打字节奏（200ms/字 vs 300ms+ 热调用）。
        # 整句场景有天然停顿（打完一串键才看结果），AI 300ms 能赶上。
        if len(self.buffer) > 4 and self.cfg.get("ai", {}).get("enabled", True):
            self.ai.request(self.buffer, "".join(self.context))

    def _commit(self, word, sel_idx=None, n_mb=0):
        """上屏一个词。

        sel_idx/n_mb：从候选池第 sel_idx 位选中、其中前 n_mb 位是码表固频。
        码表固频区的选中**不做底库调频**（remember）——码表字（吧/做/在 这类）
        的 user_count 是肌肉记忆的副产品，写进用户词库既无排序收益（固频永远
        在前），又会通过 rerank 的 unigram/个性化通道污染统计层。
        bigram 学习（rr.learn）保留： 了→在 这类真实接龙对联想有价值。
        """
        # 自动记忆/调频：偶数长码可还原拼音串，音节合法才入 user_dict
        py = None
        from_mabiao = sel_idx is not None and sel_idx < n_mb
        if self.de.loaded and not from_mabiao and len(self.buffer) >= 2 \
                and len(self.buffer) % 2 == 0:
            syls = [decode_syllable(self.buffer[i:i + 2]) for i in range(0, len(self.buffer), 2)]
            cand = " ".join(syls)
            if all(s in self.de.valid_sylls for s in syls):
                py = cand
        # 造词链的拼音（必须在 buffer 清空前取）：底库词走 word_py；
        # 码表单字/辅码筛选字走 buffer 前两键反解（音形码前两键=双拼音节）
        coin_py = self._coin_py(word) if self.de.loaded else ""
        self.buffer = ""
        self.page = 0
        self.q.put(("hide",))
        self.context.append(word)
        self.stats["commits"] += 1
        # bigram 学习：上一个上屏词 → 本词（越用越准的来源）
        self.rr.learn(self.last_word, word)
        # 多上文学习：尾部各位置的词 → 本词（位置信息由读取端的衰减处理）。
        # 这是「打碎了一个 bz→杯子」的第二次必中机制：第一次没有动宾统计
        # 证据排不到前面，用户翻页选了杯子，此处的 (打碎,杯子) 用户共现
        # （B=4，强于一切预训练证据）让它下次直接冲到前排。
        for p in self.ctx_prevs:
            if p and p != word:
                self.rr.learn(p, word)
        self.last_word = word
        # 上屏后光标前的尾部词 = 本词 + 原上文前两个（多上文滚动的近似，
        # 免去再读一次屏幕）
        self.ctx_prevs = [word] + self.ctx_prevs[:2]
        # 裁判上文同步滚动：ctx_tail_text 只在新码开始时读屏（caret_ctx），
        # 自己上屏的内容若不主动拼进去，读屏失败的应用里裁判永远看不到
        # 刚上屏的句子——接龙案（我吃了一个+bz→包子）的断点之一
        self.ctx_tail_text = (self.ctx_tail_text + word)[-32:]
        if py:
            self.de.remember(word, py)  # 新词入库/旧词调频，批量落盘
        send_unicode(word)  # 注入事件自带 INJECTED 标志，会被钩子放行
        self._streak_push(word, coin_py)  # 连续片段链：自动造词原料
        # 上屏联想已按主人 2026-09-07 指示移除：上屏后不提示任何后续词，
        # 联想的空格/数字消费分支恒不触发（self.liaison 保持 None）。

    # ---- 自动造词：连续上屏片段链 ----
    # 原理：词库里没有的组合，用户用辅码逐字打出来（张|布|斯、试|作|古|华），
    # 这本身就是「这是个词」的最强证据——打字人比任何语料库都清楚自己要什么。
    # 豆包靠云端热词表盖住这类专名；我们端侧对应物是两件事：用户打一遍
    # 永远拥有（造词）+ 公共热词管线定期注入（dicts/hotwords_inc）。

    @staticmethod
    def _is_hanzi(word):
        return bool(word) and all("\u4e00" <= ch <= "\u9fff" for ch in word)

    def _coin_py(self, word):
        """造词用的拼音：底库 word_py 优先，单字音形码走 buffer 前两键反解。

        buffer 前两键=双拼音节对小鹤音形码恒成立（音码 2 键在前、形码在后），
        所以 vlh→vl→zhang、bus→bu、siv→si 都能反解。查不到返回空串（造词
        放弃该链，防脏数据）。
        """
        p = self.de.word_py.get(word)
        if p:
            return p
        if len(word) == 1 and len(self.buffer) >= 2:
            s = decode_syllable(self.buffer[:2])
            if s in self.de.valid_sylls:
                return s
        return ""

    def _streak_push(self, word, py):
        """上屏词推入片段链。**只积累，不判定**——判定统一在断链事件
        （标点/Esc/切窗/多字词边界）里做，否则「试|作|古」会先造出中间
        垃圾词「试作古」并清链，「试作古华」永远造不出来。

        多字词=天然边界：先结算旧链再重启。非中文（英文/混合）直接断链
        ——主人明令：不记录英文。
        """
        if not py or not self._is_hanzi(word):
            self._streak_clear()
            return
        if len(word) >= 2:
            self._coin_try()
            self._streak = [(word, py)]
        else:
            self._streak.append((word, py))
            if len(self._streak) > 8:
                self._streak.pop(0)

    def _streak_clear(self):
        """断链（标点/英文/Esc/切窗）。断前先把可造的造掉——「我|吃」接
        标点也是主人的自用词，不能白打。"""
        self._coin_try()
        self._streak = []

    def _coin_try(self):
        if not self.de.loaded or len(self._streak) < 2:
            return
        hit = coin_pick(self._streak, self._streak_seen, self.de)
        if hit:
            s, py_join = hit
            self.de.remember(s, py_join)  # 入 user_dict：weight 起步 10 万，
            # 之后每打一次 remember 调频——越用越靠前，与手选词同机制
            self._streak = []
            self.q.put(("flash", "[已造词:%s]" % s))
            self.log("[造词] %s (%s)" % (s, py_join))

    # ---- 安装/卸载 ----
    def install_hook(self):
        # WH_KEYBOARD_LL 的 hMod 必须为 NULL（钩子过程在本进程内，不加载 DLL）
        self._hook = user32.SetWindowsHookExW(WH_KEYBOARD_LL, self._proc, None, 0)
        return bool(self._hook)

    def uninstall_hook(self):
        if self._hook:
            user32.UnhookWindowsHookEx(self._hook)
            self._hook = None


# ---------------- 主程序 ----------------

def load_cfg(base):
    path = os.path.join(base, "config.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


_single_instance = None  # 持有互斥体句柄，进程存活期间不释放


def acquire_single_instance():
    """单实例互斥（Windows 命名互斥体）。返回 False 表示已有灵鹤在跑。

    多实例叠加会装多个 WH_KEYBOARD_LL 钩子互相抢键——实测表现为
    "只能打一个字母"（一实例吃键组码、另一实例把后续键吞掉/干扰）。
    进程退出（含崩溃）时系统自动回收互斥体，无需显式清理。
    """
    global _single_instance
    if os.name != "nt":
        return True
    import ctypes
    # use_last_error=True + get_last_error()：ctypes 内部调用会污染 GetLastError，
    # 直接 windll.kernel32.GetLastError() 读到的是残留码（曾致无实例也误判 183）
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateMutexW.restype = ctypes.c_void_p
    k32.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_bool, ctypes.c_wchar_p]
    ERROR_ALREADY_EXISTS = 183
    h = k32.CreateMutexW(None, False, "Local\\LingHe_IME_SingleInstance")
    if not h:
        return True  # 创建失败不拦正常流程，宁可放过不可错杀
    if ctypes.get_last_error() == ERROR_ALREADY_EXISTS:
        k32.CloseHandle(ctypes.c_void_p(h))
        return False
    _single_instance = h  # 仅引用防 GC；句柄随进程销毁
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="自检模式：加载码表/探测AI/装钩子后自动退出")
    ap.add_argument("--smoke-seconds", type=float, default=3.0)
    args = ap.parse_args()

    if getattr(sys, "frozen", False):
        # PyInstaller 打包：数据目录（dicts/mabiao/ai_neural）放在 exe 旁边
        base = os.path.dirname(os.path.abspath(sys.executable))
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    # 日志双写：控制台在（源码 run.bat）也同步落盘 linghe.log——真机排障
    # 需要证据（2026-09-06 主人 run.bat 案：控制台一关，现场全丢）。
    # 追加模式：即使第二实例瞬间启动又退出，也不截断第一实例的日志
    try:
        _logf = open(os.path.join(base, "linghe.log"), "a", encoding="utf-8",
                     buffering=1)
    except OSError:
        _logf = None

    class _Tee:
        def __init__(self, *fps):
            self._fps = [f for f in fps if f is not None]

        def write(self, s):
            for f in self._fps:
                try:
                    f.write(s)
                    f.flush()
                except Exception:
                    pass
            return len(s)

        def flush(self):
            for f in self._fps:
                try:
                    f.flush()
                except Exception:
                    pass

    _console = sys.stdout
    if _console is None:  # windowed 打包无控制台
        sys.stdout = _logf if _logf else open(os.devnull, "w", encoding="utf-8")
    else:
        sys.stdout = _Tee(_console, _logf) if _logf else _console
    sys.stderr = sys.stdout
    sys.path.insert(0, base)
    cfg = load_cfg(base)

    if not args.smoke and not acquire_single_instance():
        print("[灵鹤] 已有实例在运行，本次启动自动退出（单实例互斥）")
        return

    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass

    root = tk.Tk()
    root.withdraw()

    lines = []
    eng = Engine(base, cfg, log=lambda s: (lines.append(s), print(s)))
    ui = CandidateWindow(root)
    last_pos = (300, 300)

    def poll():
        nonlocal last_pos
        try:
            while True:
                item = eng.q.get_nowait()
                kind = item[0]
                if kind == "show":
                    _, code, cands, n_mb, pos = item
                    last_pos = pos
                    off = eng.page * eng.n_show
                    chunk = cands[off: off + eng.n_show]
                    shown = [(w, i == 0) for i, w in enumerate(chunk)]
                    ui.update(code, shown, pos)
                elif kind == "hide":
                    ui.hide()
                elif kind == "think":
                    _, code, pos = item
                    last_pos = pos
                    ui.thinking(code, pos)
                elif kind == "flash":
                    _, text, = item[0], item[1]
                    ui.flash(text, last_pos)
                    root.after(900, ui.hide)
                elif kind == "liaison":
                    # 上屏联想：候选窗显示 [已上屏词 + 联想词列表]，
                    # 空格=上屏首选，数字键=选第 n 个联想
                    _, prev, nxt, pos = item
                    last_pos = pos
                    ui.update(prev, [(w, i == 0) for i, w in enumerate(nxt)], pos)
                elif kind == "ai":
                    # AI 结果返回：失效缓存后重算（把 AI 候选并入），按当前页显示
                    if eng.buffer:
                        eng._cache_code = None
                        cands, n_mb = eng.compute(eng.buffer)
                        off = eng.page * eng.n_show
                        chunk = cands[off: off + eng.n_show]
                        shown = [(w, i == 0) for i, w in enumerate(chunk)]
                        if shown:
                            ui.update(eng.buffer, shown, last_pos)
                        else:
                            ui.thinking(eng.buffer, last_pos)
                    else:
                        ui.hide()
        except queue.Empty:
            pass
        root.after(16, poll)

    eng.install_hook()
    if not eng._hook:
        print("[致命] 低级键盘钩子安装失败 err=%d" % kernel32.GetLastError())
        return 1
    print("=" * 56)
    print(" 灵鹤 LingHe  外挂式 AI 输入法（小鹤音形向）")
    print("  开关热键: Ctrl+Alt+L    退出: 控制台 Ctrl+C")
    print("  空格=首选  ;=2选  '=3选  数字1-9=选字  Esc=清码")
    print("  =/Tab=下一页  -=上一页  回车=上屏英文  Shift单击=上屏英文/切中英")
    print("=" * 56)
    for s in lines:
        print(s)

    if args.smoke:
        def done():
            print("[冒烟] 钩子=OK  统计=", eng.stats)
            print("[冒烟] AI:", eng.ai.endpoint_desc, "ready=", eng.ai.ready)
            print("[冒烟] 样例查询 aih ->", eng.mb.exact("aih")[:5])
            print("[冒烟] 样例查询 aq  ->", eng.mb.exact("aq")[:5])
            eng.uninstall_hook()
            root.destroy()
        root.after(int(args.smoke_seconds * 1000), done)
    else:
        root.protocol("WM_DELETE_WINDOW", lambda: (eng.uninstall_hook(), root.destroy()))
    poll()
    root.mainloop()
    # 退出前把用户词频与 bigram 共现落盘（防丢）；造词链也做最后一次结算
    eng._coin_try()
    eng.de.flush()
    eng.rr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
