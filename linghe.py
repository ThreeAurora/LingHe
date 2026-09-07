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
import itertools
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
from gram_model import GramModel, NC_PEN
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
VK_LMENU, VK_RMENU = 0xA4, 0xA5   # 左/右 Alt（LL 钩子不折叠，VK_MENU 仅作状态查询）
VK_TAB = 0x09
VK_L = 0x4C
VK_OEM_1 = 0xBA    # ; :
VK_OEM_7 = 0xDE    # ' "
VK_OEM_MINUS = 0xBD  # - _（上一页）
VK_OEM_PLUS = 0xBB   # = +（下一页）
VK_CAPITAL = 0x14    # CapsLock（大写锁定）

# 中文态全角标点（US 布局 vk，2026-09-07 主人定案：中文态所有符号一律全角）：
# 元组 = (无Shift半角, 无Shift全角, Shift半角, Shift全角)。
FULLWIDTH_PUNCT = {
    0xBC: (",", "，", "<", "《"),   # , <
    0xBE: (".", "。", ">", "》"),   # . >
    0xBA: (";", "；", ":", "："),   # ; :
    0xDE: ("'", "’", '"', "“"),    # ' "
    0xBF: ("/", "／", "?", "？"),   # / ?
    0xDB: ("[", "【", "{", "｛"),   # [ {
    0xDC: ("\\", "、", "|", "｜"),  # \ |
    0xDD: ("]", "】", "}", "｝"),   # ] }
    0xBD: ("-", "—", "_", "——"),   # - _
    0xBB: ("=", "＝", "+", "＋"),   # = +
}

# 数字行上档符号的全角形态（Shift+1..0）。数字本身半角直通——只有符号要全角。
DIGIT_SYM = {
    0x31: "！", 0x32: "＠", 0x33: "＃", 0x34: "＄", 0x35: "％",
    0x36: "＾", 0x37: "＆", 0x38: "＊", 0x39: "（", 0x30: "）",
}


def cn_punct(vk, shift):
    """中文态下 vk+Shift 对应的全角标点；非标点键返回 None。"""
    m = FULLWIDTH_PUNCT.get(vk)
    if m:
        return m[3] if shift else m[1]
    if shift:
        return DIGIT_SYM.get(vk)  # Shift+数字 = 上档符号，全角
    return None

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
SENTS4_MIN = 0.4         # 4 字整句通道闸门（低于 EV_SENT_GOOD，单设）：4 字
                         # 串 trigram 数=2，rate=1.0 时才可能 ≥0.4——实际放行
                         # 的只有「两个 3gram 全命中」的强证据组合（这是一部
                         # 0.45，切分成 这是一|部 被孤字罚 0.2，仍过线）；菌类
                         # 职务类 3gram 零命中 ev0 照挡。碎片句混入风险由长句
                         # 闸（≥5 字 EV_SENT_GOOD 0.55）承担，与本通道无关。
GRAM_MAX = 12.0         # 万象语法证据减价封顶：证据 ev=s-(-12) 上限 12（约
                        # 对应搭配对数概率 12/万位级），乘以 GRAM_ALPHA 后最多
                        # 减 3.0 代价单位——够把上下文里的词顶到动态区前排，
                        # 又不至于掀翻码表固频区。
GRAM_ALPHA = 0.25       # 语法证据权重：0.25 × 证据 = 减价。吃了一个|包子
                        # ev=18 → 减 3.0；无证据词 0 减价保持统计序。
GRAM_TOP_EV = 6.0       # 动态调频压顶闸：上下文证据 ev（=gram 分+12）≥6 才允许
                        # 压过码表固频做首选（一部史诗 ev=22）。低于 6 的弱
                        # 搭配不压——固频区是肌肉记忆，证据不足不得掀翻。

# 动态调频开关（2026-09-07 主人定案）：odt 加空格开启/关闭。
# odt 作为词条固定在首选（词条文案提示当前模式），按空格即切换：
#   odt+空格 → 动态模式开（四码起一切候选动态排序，固频区让位）
#   odt+空格 → 动态模式关（回到码表固频词库）
# 动态排序只影响候选窗本次排序，绝不写码表——固频词库词序不受影响。
DYN_SWITCH_CODE = "odt"
DYN_TAG_ON = "动态调频:开"    # 当前已开 → 显示开（再按空格=关）
DYN_TAG_OFF = "动态调频:关"   # 当前已关 → 显示关（再按空格=开）
DYN_MIN_LEN = 4              # 四码起（含四码）一切动态


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
        elif not singles:
            # 多字词+多字词（这是|一部）：词库没有的组合连续上屏=自用词。
            # 2026-09-07 报案「长词不去造了」：此前该形态无任何分支，永远不造。
            # 高频词头（这是/东西，≥COIN_MULTI_MAX）像句子流，首次不造、
            # 重复才造；含低频词头（专名特征，布加迪|威航）第一次就造。
            hi = [m for m in multi if de.weight(m) >= COIN_MULTI_MAX]
            ok = (len(hi) < len(multi)) or seen_n >= 1
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
user32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
user32.GetWindowThreadProcessId.restype = wt.DWORD
user32.SendInput.argtypes = [wt.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.SendInput.restype = wt.UINT

# 系统输入法（IME）控制：灵鹤启动后接管输入，把系统输入法「打开状态」置 False
# （切回英文直通），防止微软拼音等与灵鹤同时弹候选窗抢占按键（2026-09-07 报案）。
imm32 = ctypes.WinDLL("imm32", use_last_error=True)
imm32.ImmGetContext.argtypes = [wt.HWND]
imm32.ImmGetContext.restype = ctypes.c_void_p
imm32.ImmSetOpenStatus.argtypes = [ctypes.c_void_p, ctypes.c_int]
imm32.ImmSetOpenStatus.restype = ctypes.c_int
imm32.ImmReleaseContext.argtypes = [wt.HWND, ctypes.c_void_p]
imm32.ImmReleaseContext.restype = ctypes.c_int


def ime_close():
    """关闭系统中文输入法（打开状态 → False）。

    只对「前台窗口的输入上下文」生效；窗口切换后系统输入法可能自己恢复
    打开状态，届时灵鹤的低级钩子仍在拦截，双输入法抢键问题由钩子侧兜底
    （系统输入法拿不到已拦截的按键）。
    """
    fg = user32.GetForegroundWindow()
    if not fg:
        return
    try:
        himc = imm32.ImmGetContext(fg)
        if himc:
            imm32.ImmSetOpenStatus(himc, False)
            imm32.ImmReleaseContext(fg, himc)
    except Exception:
        pass


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
    """取前台应用光标屏幕坐标；取不到则回退鼠标位置。

    关键：GetGUIThreadInfo 必须传**前台窗口所属线程** ID——传 0（当前线程）
    查的是钩子线程自己，hwndCaret 恒空，回退鼠标会让候选窗盖在光标上
    （2026-09-07 主人报案「打字位置覆盖光标」）。
    """
    fg = user32.GetForegroundWindow()
    if fg:
        tid = user32.GetWindowThreadProcessId(fg, None)
        if tid:
            info = GUITHREADINFO()
            info.cbSize = ctypes.sizeof(GUITHREADINFO)
            if user32.GetGUIThreadInfo(tid, ctypes.byref(info)) and info.hwndCaret:
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
        self._cache_dyn = None       # 缓存对应的动态调频状态（开/关结果不同）
        self._cache_smart0 = False   # 四码契约：首选位被智能词占据（选中仍调频）
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
        self._composed_py = {}       # 组词通道：词外词→拼音（选中造词/云端校验）
        self.buffer = ""
        self.cn_mode = True          # Shift 单击切换中/英
        self.shift_t0 = 0
        self.shift_alone = False
        self.alt_down = False        # Alt 键自维护状态（2026-09-07：GetAsyncKeyState
                                     # 异步漏检导致 alt+a 打出 aaaa，改事件状态机）
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

        # 终审裁判：云端大模型（豆包式生成首选）。2026-09-07 主人判死本地：
        # 本地 QwenJudge/RBT3 从未给出可用智能（几十轮修补仍是垃圾），整条
        # 本地智能链路已删（见 _after_edit 注释）。云端是唯一智能层：
        # 配置了 api_key → 云端生成通道接管首选；未配置 → 纯静态检索
        # （码表+底库+整句召回），不再回落任何本地排序智能。
        jcfg = cfg.get("judge", {})
        self.cloud_judge = CloudJudge(jcfg, log=log)
        self.cloud_judge.on_generate = self._on_cloud_generate
        self.cloud_judge.on_ready = self._on_backend_ready
        if jcfg.get("enabled", True) and jcfg.get("api_key"):
            self.cloud_judge.load_async()
        else:
            log("[云端裁判] 已停用（judge.enabled=false 或未配 api_key），纯端侧：万象语法 + 统计检索")

        # 万象语法模型（端侧 n-gram 智能首选，2026-09-07 集成）：0.1s 智能
        # 不靠云端——.gram 直接 mmap 直读（420MB 文件 2ms 就绪），查询微秒级，
        # compute 里同步打上下文分，把「上文里最该出现的词」浮到动态区顶。
        # 无需 Rime 框架，纯格式解析（gram_model.py）。云端仍可异步精排，
        # 但首选速度由万象语法兜底。
        gcfg = cfg.get("gram", {})
        self.gram = None
        if gcfg.get("enabled", True) and gcfg.get("model"):
            try:
                self.gram = GramModel(gcfg["model"])
                log("[万象语法] 加载成功：%s" % gcfg["model"])
            except Exception as e:
                log("[万象语法] 加载失败：%s" % e)

        self.buffer = ""
        self.dyn_mode = False   # 动态调频模式（odt 空格切换，四码起全动态）
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

    # ---- 辅码逐字锁定组词（2026-09-07 主人 xlgmko→象鸣 案）----
    def _mb_single_chars(self, syl, aux, cap=6):
        """码表音码桶里的单字（固频序≈常用序），辅码首键过滤。

        象 = xl 桶 ∩ 辅码g；鸣 = mk 桶 ∩ 辅码o。辅码表（手心）与码表
        （简单鹤）形码体系独立，前缀桶只保「音对」，辅码必须经
        fm.word_match 复核——两个体系在主人实测里恰好一致（xlg→象、
        mko→鸣 双双命中），过滤后剩下的就是身份确定的字。
        """
        out, seen = [], set()
        for w in self.mb._prefix.get(syl, ()):
            if len(w) != 1 or w in seen:
                continue
            if aux is not None and not self.fm.word_match(w, aux):
                continue
            seen.add(w)
            out.append(w)
            if len(out) >= cap:
                break
        return out

    @staticmethod
    def _cross(per, cap):
        """逐位字表叉乘，按「固频名次和」升序——每字都取最常用字的组合在前。"""
        combos = [(sum(idxs), idxs)
                  for idxs in itertools.product(*(range(len(c)) for c in per))]
        combos.sort()
        return ["".join(per[i][j] for i, j in enumerate(idxs))
                for _, idxs in combos[:cap]]

    def _compose_locked(self, code):
        """词库里没有的词（象鸣）也要能组出来——辅码逐字锁定即字身份确定。

        模式3（音2+辅1/字）把每个字的音和辅码全部打满时，逐字查
        「音码桶∩辅码首键」得到身份确定的字，按位组词——词库有没有
        这个词无关紧要（象鸣 不在任何词表，xl+g/mk+o 已唯一锁定）。
        返回 (全锁定组词, 部分锁定组词)：
          全锁定：模式3 每音节都有本字辅码 → 确凿组词，尾随码表/底库
                  真词之后进固频区（xlgmko 无真词命中时 象鸣 即首选）；
          部分锁定：模式4 仅首字辅码 → 首字锁定+余字固频 top 组合，进
                  动态区前部（「至少打 xlgmk 得有词」）。
        无本字辅码的解释（模式1/2）不组：纯音节的兜底已有整句/字链通道。
        组词拼音记入 _composed_py：选中上屏即自动造词入库（_commit），
        云端候选校验同源放行（_cloud_code_ok）——造词闭环。
        """
        if not self.fm.loaded or not self.mb.loaded:
            return [], []
        if len(self._composed_py) > 4096:
            self._composed_py.clear()   # 会话级缓存防爆
        full_out, part_out = [], []
        for syls, fuses in key_parser.parse(code):
            if len(syls) < 2 or len(syls) > 6:
                continue        # 组词至少两字；超长码叉乘爆炸，不组
            locks = {i: k for i, k in fuses if 0 <= i < len(syls)}
            if not locks:
                continue
            cap = 6 if len(syls) <= 3 else 3   # 限制叉乘规模
            per, n_lock = [], 0
            for i, s in enumerate(syls):
                k = locks.get(i)
                chars = self._mb_single_chars(s, k, cap=cap)
                if not chars:
                    per = None
                    break       # 锁定音节无字可解 → 该解释不成立
                if k:
                    n_lock += 1
                per.append(chars)
            if not per:
                continue
            dst = full_out if n_lock == len(syls) else part_out
            for w in self._cross(per, 8):
                if w not in dst:
                    dst.append(w)
                    if len(dst) >= 8:
                        break
            if dst:
                py = " ".join(decode_syllable(s) for s in syls)
                for w in dst:
                    self._composed_py.setdefault(w, py)
        return full_out[:8], part_out[:8]

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
                and self.dyn_mode == self._cache_dyn \
                and self._cache_cands is not None:
            return self._cache_cands, self._cache_nmb
        # 动态调频开关词条（2026-09-07 主人定案）：odt 固定首选，空格切换。
        if code == DYN_SWITCH_CODE:
            tag = DYN_TAG_ON if self.dyn_mode else DYN_TAG_OFF
            self._cache_code = code
            self._cache_ctx = tuple(self.ctx_prevs)
            self._cache_dyn = self.dyn_mode
            self._cache_cands = [tag]
            self._cache_nmb = 1
            self._cache_smart0 = False
            return [tag], 1
        n = len(code)
        fused = self._fused_candidates(code)
        mb_exact = list(self.mb.exact(code))
        seen = set(fused) | set(mb_exact)
        # 辅码逐字锁定组词（2026-09-07 主人 xlgmko→象鸣 案）：全锁定组词
        # 尾随码表/底库真词进固频区——真词永远优先（异议 案：yityic 的
        # fused 真词压住组出的 异意 等组合），无真词时 象鸣 即首选。
        comp_full, comp_part = self._compose_locked(code)
        comp_full = [w for w in comp_full if w not in seen]
        comp_part = [w for w in comp_part if w not in seen]
        seen.update(comp_full)
        seen.update(comp_part)
        mb_hits = fused + mb_exact + comp_full
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
        # 【2026-09-07 动态调频·一修】主人实测钉死：把码表词按统计代价
        # 丢进 pool 融化固频，书面高频虚词（的了是就）代价最小霸榜首选，
        # 真词全被挤到翻页。统计代价没有用户意图，顶不动固频的确定性——
        # 动态调频的正确姿势是云端压顶（见 _on_cloud_generate），本地
        # 候选保持固频+统计稳定序。故此处不再注入码表词。
        ranked_pool = sorted(pool.items(), key=lambda kv: kv[1])
        # 【2026-09-07 万象语法·端侧智能首选】0.1s 不靠云端：gram 模型 mmap
        # 直读、查询微秒级，同步给动态池打上下文分。证据 ev = max(0, s-NC_PEN)
        # （s 是「上文|词」的搭配对数概率，无证据=0）；有搭配证据的词
        # 减价 GRAM_ALPHA×min(GRAM_MAX, ev)，无证据词保持统计序——上文
        # 里最该出现的词自然浮顶，码表固频区不受影响（肌肉记忆优先）。
        # 实测：我今天想吃|西湖醋鱼=10.3 vs 我今天想喝|西湖醋鱼=-12（区分
        # 「吃/喝」的搭配语义），吃了一个|包子=6.0（真搭配证据）。
        if self.gram is not None:
            gctx = self.ctx_tail_text or "".join(self.context)[-8:]
            gwin = ([gctx] + [w for w in self.ctx_prevs[:3]
                              if w and w not in (gctx,)]) if gctx else []
            gboost, gev = {}, {}
            for w, cps in ranked_pool:
                # 【2026-09-07 蘑菇喷出了bczi 案】多窗口上下文：语法模型只认
                # 尾缀搭配，「喷出了孢子」无词条时整段证据归零，藏在更早的
                # 「蘑菇→孢子 / 喷出→孢子」搭配被中间词截断。逐上文词单独
                # 查窗取 max——真搭配自然浮出（蘑菇孢子 ev17 vs 包子 13.7），
                # 无关词窗零证据不会反超。tail_words 只切 3 个词（右起），
                # 主语「蘑菇」常落在第 3 位，故全取（喷出/蘑菇 都能各自贡献）。
                if gwin:
                    ev = max(self.gram.query(c, w, is_rear=(len(w) >= 5))
                             for c in gwin) - NC_PEN
                else:
                    ev = 0.0
                gev[w] = ev
                if ev > 0:
                    gboost[w] = GRAM_ALPHA * min(GRAM_MAX, ev)
                # 【2026-09-07 蘑菇喷出了孢子 整句一次性识别案】整句候选的
                # 尾词由句内前词单独确证：蘑菇喷出了|包子/豹子/孢子 三句同音
                # 同构统计同价，语料只有「蘑菇→孢子」是真搭配（ev17.3 vs
                # 包子 13.5 vs 豹子 0）——用句内前词当窗口查尾词，真搭配句
                # 浮到最前。外部窗口证据顶不动码表固频才封顶 GRAM_MAX；整句
                # 内部排序是统计地界，证据差必须完整反映，封顶放宽到 2×
                # （否则同音三句全顶到上限分不出先后）。只对零孤字词链句
                # 放行（伪句带单字残渣不参与）；史诗级→对决 ev22.3 顺带把
                # uiuijidvjt 提到整句区首位（回归用户原始诉求）。
                if len(w) >= 5 and self.rr._clean_chain(w):
                    words = self.rr._cut_words(w)
                    if len(words) >= 2:
                        tail = words[-1]
                        if tail in self.de.word_py and self.de.weight(tail) > 0:
                            iwins = list(dict.fromkeys(words[:-1]))
                            iev = max(self.gram.query(
                                c, tail, is_rear=(len(tail) >= 5))
                                for c in iwins) - NC_PEN
                            if iev > 0:
                                gboost[w] = max(
                                    gboost.get(w, 0.0),
                                    GRAM_ALPHA * min(GRAM_MAX * 2, iev))
            if gboost:
                ranked_pool = sorted(
                    ranked_pool,
                    key=lambda kv: kv[1] - gboost.get(kv[0], 0.0),
                )
        else:
            gctx, gev = "", {}
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
        # 【2026-09-07 动态调频·一修】固频区不再融化（见上注释）：四码起
        # 的「动态」交给云端压顶（_on_cloud_generate），本地始终保持
        # 固频+统计稳定序——用户背过的码永远在，不因统计代价翻车。
        if n <= 3:
            mb_part = (mb_hits + [w for w in mb_prefix if len(w) <= 2])[: 9]
            n_mb = len(mb_part)
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
        # 4 字整句通道（2026-09-07 主人报案「打不出'这是一部'」）：weight 无
        # 记录的 4 音节长词（viterbi/锚点切出的 这是|一部）被权重闸整段挡在
        # 候选外。口语 3gram 证据达标的 4 字组合算真词——"这是一部"（这是一
        # /是一部 双命中 ev0.65）放行，"菌类职务"类自然二连（3gram 零命中
        # ev0）照挡。
        # 【二修】原实现从 long_words 取——long_words 要求 py_len≥3（有
        # word_py 注音的字典词），viterbi 拼出的非词典 4 字组合 py_len=0
        # 根本进不了 long_words，通道对目标组合永远空转。改从 ranked_pool
        # 直接取（与 sents_good 同源），w not in heavy 防与真词重复。
        sents4 = [w for w, _ in ranked_pool
                  if len(w) == 4 and w not in heavy
                  and self.rr._eval_sent(w) >= SENTS4_MIN]
        cap_dyn = 48 if n == 2 else (self.n_pool - n_mb)
        # 部分锁定组词（象+固频top：xlgmk→象明/象名/象鸣…）进动态区最前：
        # 带着主人敲出的首字辅码，比纯统计词更接近意图；排序不再下沉。
        dict_side = (comp_part + short_words + heavy + sents4)[: max(cap_dyn, 16)]
        base = mb_part + dict_side
        seen_base = set(base)
        ev = lambda w: self.rr._eval_sent(w)
        # 整句三档：①词典整词（_eval_sent=1.0）②口语 3gram 证据达标
        # （>=EV_SENT_GOOD）③零孤字词链（_clean_chain：史诗级|对决 这种
        # 词典词干净拼接，2026-09-07 uiuijidvjt 案）。③单独存在是因为 3+2/
        # 2+3 拆法的 5 字短语中间跨词 3gram 天然是残渣，证据分结构性上不了
        # 0.55——真短语与拼字伪句（是十几对绝）在 ev 上同分，只有词链干不
        # 干净能分开。③只进候选池尾部，不替代统计排序、不进裁判名单。
        sents_good = [w for w, _ in ranked_pool
                      if len(w) >= 5 and (ev(w) >= EV_SENT_GOOD
                                          or self.rr._clean_chain(w))]
        base += [w for w in sents_good if w not in seen_base]
        # 【2026-09-07 主人定案·四码排序契约】四码候选形态：
        #   [智能首选词] + [码表四码单字（固频序）] + [码表其余] + [动态其余]…
        # 主人原话：几码都应该智能调频，仅前两码跟在固定词库后面；三码跟在
        # 固定词库三码单字后面（现状已满足：mb_part 前置）；第四码智能首选
        # 词去第一位——码表该码首选是①占位或词语时直接替换它；码表单字
        # 永远固定在次选、三选、四选…（肌肉记忆位置不漂移，想打单字翻位
        # 可盲选）。动态池空时码表原序全权接管。①不是汉字，单字判定走
        # _is_hanzi。_cache_smart0 标记首选被智能词占据：选中首选仍走
        # 智能调频（remember），不受 from_mabiao 豁免。
        self._cache_smart0 = False
        if n == 4:
            singles4 = [w for w in mb_exact
                        if len(w) == 1 and self._is_hanzi(w)]
            # 智能首选词 = 动态池里打满音节数（4键=2音节）的真词之首：
            # 单字虚词（的一是）py_len=1 不配顶四码首选——用户打满 2 音节
            # 意图就是 2 音节词（与 WORD_BOOST / 云端压顶的整词音节约定一致）
            smarts = [w for w in dict_side
                      if len((self.de.word_py.get(w) or "").split()) == n // 2]
            smart = smarts[0] if smarts else ""
            # 主人契约：单字**始终**固定在四码的次选起（多个单字依次三选、
            # 四选…）；码表①占位/词组本是占位，退到单字之后（yiyi 案：
            # exact=意义/刈/异议 —— 意义占位词不许把单字 刈 顶离次选）；
            # 有智能首选词时再顶到第 1 位（①/词语被替换）。动态池空且
            # 无单字时码表原序不动。
            if smart or singles4:
                head = ([smart] if smart else []) + singles4
                rest = [w for w in base if w not in head]
                base = (head + rest)[: self.n_pool]
                if smart:
                    self._cache_smart0 = True
                    if mb_part:
                        n_mb += 1   # 码表固频区整体被插队后移一位
        # 【2026-09-07 端侧动态调频压顶】dyn_mode 四码起（odt 空格开）：
        # 云端已关，压顶由万象语法接棒——「上下文证据最强」的候选提到码表
        # 固频之前做首选（这是一部+uiui → 史诗，gram 10.3/ev22 压过码表
        # 事实/实施/试试）。证据不够 GRAM_TOP_EV 不压：固频是肌肉记忆，
        # 只有语言模型确证「下文就该是这个」才让位（防冷门词霸榜重演）。
        if self.dyn_mode and n >= DYN_MIN_LEN and gctx and gev:
            # 【2026-09-07 蘑菇喷出了bczi 案】压顶只认词库真词且只压「打满整词
            # 音节」的词：viterbi 拼字伪句（不存在出 w=0，靠 存在→不存在 类词条
            # 蹭证据 ev17.6）与简拼前缀混进池的异长词（菠菜甾醇 4 音节蹭
            # 蘑菇→甾醇 语料证据 ev>17）都不配——用户 4 键双拼只打满了 2 音节，
            # 压顶只能把 2 音节真词提到固频之前（与 WORD_BOOST 的整词音节约定
            # 一致：m == n//2）。压顶是「上下文确证该出哪个词」的最高权力。
            n_syl = n // 2
            real_gev = {w: ev for w, ev in gev.items()
                        if w in self.de.word_py
                        and len((self.de.word_py.get(w) or "").split())
                        in (n_syl, n_syl + n % 2)}
            if real_gev:
                top_w = max(real_gev.items(), key=lambda kv: kv[1])
                if top_w[1] >= GRAM_TOP_EV and top_w[0] not in mb_part:
                    base = [top_w[0]] + [w for w in base if w != top_w[0]]
        # AI 顺延：静态侧命中 k 个，AI 从第 k+1 位起
        merged = base[:]
        for w in self.ai.peek(self.context_key(), code):
            if w not in merged:
                merged.append(w)
        merged = merged[: self.n_pool]
        self._cache_code = code
        self._cache_ctx = tuple(self.ctx_prevs)  # 上文变了结果必须重算（同码不同上文）
        self._cache_dyn = self.dyn_mode
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

        # Alt 键状态机（2026-09-07 主人报案：alt+a 打出 aaaa）。GetAsyncKeyState
        # 是异步状态，钩子回调里偶发漏检；改由按下/释放事件自维护。按住 Alt
        # 期间的任何键都以系统组合放行（PowerToys 的 alt+wasd→方向键映射全靠
        # 这些事件，输入法不得截胡，也不得把 A 当组码）。
        if vk in (VK_LMENU, VK_RMENU):
            if msg in (WM_KEYDOWN, WM_SYSKEYDOWN):
                self.alt_down = True
            elif msg == WM_KEYUP:
                self.alt_down = False
            return user32.CallNextHookEx(None, ncode, wparam, lparam)

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
        if msg == WM_SYSKEYDOWN or ctrl or self.alt_down \
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
            ime_close()           # 新窗口的系统输入法若自恢复打开，一并关闭

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

        # 翻页：=/Tab 下一页（末页回卷），- 上一页；无组码时透传。
        # 2026-09-07 报案「翻页键没用」：此前只改 self.page 不重发 show，
        # 候选窗永不刷新——翻页必须重发当前页（UI 按 eng.page 切 chunk）。
        if self.buffer and vk in (VK_OEM_PLUS, VK_OEM_MINUS, VK_TAB):
            cands, n_mb = self.compute(self.buffer)
            n_pages = max(1, (len(cands) + self.n_show - 1) // self.n_show)
            if vk == VK_OEM_MINUS:
                self.page = max(0, self.page - 1)
            else:
                self.page = 0 if self.page + 1 >= n_pages else self.page + 1
            pos = caret_pos()
            self._last_pos = pos
            self.q.put(("show", self.buffer, cands, n_mb, pos))
            return self._eat()

        # 中文态全角标点（2026-09-07 主人定案：中文态所有符号一律全角）。
        # 组码中：,./[\ 等先上屏首选再注入全角；;' 保留 2选/3选 惯例。
        # 无组码：标点键直接注入全角。Shift 上档（《》？…）跟随。
        punc = cn_punct(vk, key_down(VK_SHIFT))
        if punc is not None and not (self.buffer and vk in (VK_OEM_1, VK_OEM_7)):
            if self.buffer:
                cands, n_mb = self.compute(self.buffer)
                if cands:
                    self._commit(cands[0], 0, n_mb)
                self._streak_clear()  # 标点=句子边界：链先结算再清
            self._eat()
            send_unicode(punc)
            return 1

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

        if self.buffer:
            self.buffer = ""
            self.q.put(("hide",))
        return user32.CallNextHookEx(None, ncode, wparam, lparam)

    def _on_nr_result(self, code, scores, ms):
        """RBT3 返回分流——2026-09-07 主人判死：本地神经判别已删（neural.
        enabled=false，不加载不请求），本回调永远不会被触发，留作死路径
        守卫：万一旧配置仍加载 RBT3，结果直接丢弃，绝不进候选流。"""

    def _on_neural(self, code, scores, ms, judge=True):
        """精排返回（worker 线程）：统计分 + λ·每字logP 重排动态区，刷新当前页。

        只重排池内已有的候选（MLM 打分是判别式的：不生成、只裁决）。
        buffer 已变（用户继续击键/已上屏）则丢弃本包结果。

        回流（2026-09-07 bz→包子 案）：评分名单已扩至全量动态池，base 外
        的判别高分词组（包子 统计 35 位被截断、LLM 判第 1）按融合分与池内
        词组同场排序，自然回流进动态区前部——「统计层管召回，裁判管裁决」。
        """
        # 【2026-09-07 动态调频·一修】动态模式四码起：云端答案已压顶首选，
        # 本地神经/裁判回流不再干预——0.5B/RBT3 判别不过云端语义，回插会
        # 把云端首选顶下去（主人实测钉死：首选变冷门词）。云端说了算。
        if self.dyn_mode and len(code) >= DYN_MIN_LEN:
            return
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

    @staticmethod
    def _cloud_norm(s):
        """去拼音声调/变体：bǎo→bao、zhèng→zheng、lǜ→lv。留 a-z 和 v。"""
        s = (s.replace("ü", "v").replace("ǖ", "v").replace("ǘ", "v")
             .replace("ǚ", "v").replace("ǜ", "v"))
        return "".join(c for c in s if ("a" <= c <= "z") or c == "v")

    def _cloud_code_ok(self, code, word):
        """云端生成的候选必须「音对得上码」，否则丢弃（2026-09-07 主人 ilyg 案）。

        code 有两条合法解读（xiaohe）：
        - 简拼：每键一声母（zh/ch/sh 记 v/i/u，零声母音节取首字母 a/o/e）；
        - 双拼：每 2 键一完整音节（decode_syllable 逐对解码）。
        读音来源：多字词词条固定读音 word_py 优先，其后逐字 char_py 全读音
        （带声调，先归一）兜底——万=wàn/mò 这类首读错的字能救回。任一解读
        对上即放行；有字查不到读音直接拒（宁缺毋滥，幻觉绝不进候选窗）。
        ilyg 案钉死：云端出「一起一个/以后有空/有了一个」读音（yqyg/yhyk/
        ylyg）与码（ilyg）全不对应——没有这道闸，云端幻觉直接上屏霸占首选。
        """
        if not word:
            return False
        # 组词通道的词外词（象鸣）自带拼音，同源放行云端校验
        py = self._composed_py.get(word) or self.de.word_py.get(word)
        if py and len(py.split()) > 1:
            # 词条固定读音快速通道：每音节包一层列表（_cloud_matches 按
            # 「每字一个读音列表」展开——扁平串会把音节拆成逐字符遍历，
            # zheng 被读成 z/h/e/n/g 导致 zh 词错误放行，2026-09-07 钉死）。
            if self._cloud_matches(code, [[self._cloud_norm(s)] for s in py.split()]):
                return True
        choices = []
        for ch in word:
            cpy = self.de.char_py.get(ch)
            if not cpy:
                return False   # 有字查不到拼音 → 不敢放行（宁缺毋滥）
            choices.append([self._cloud_norm(s) for s in cpy.split()])
        return self._cloud_matches(code, choices)

    def _cloud_matches(self, code, choices):
        """choices: 每字一个读音列表（已归一）。简拼/双拼任一解读对上即过。"""
        # 简拼：每位 ini 键 == code[i]
        if len(choices) == len(code):
            for i, chs in enumerate(choices):
                if not any(self._cloud_ini(s) == code[i] for s in chs):
                    break
            else:
                return True
        # 双拼：每位 decode(code 两键) ∈ 该字读音
        if len(choices) == len(code) // 2:
            for i, chs in enumerate(choices):
                d = decode_syllable(code[i * 2:i * 2 + 2])
                if not any(s == d for s in chs):
                    break
            else:
                return True
        return False

    @staticmethod
    def _cloud_ini(s):
        """归一音节 → 简拼键（zh/ch/sh 记 v/i/u，零声母音节取首字母 a/o/e）。"""
        if s[:2] in ("zh", "ch", "sh"):
            return {"zh": "v", "ch": "i", "sh": "u"}[s[:2]]
        return s[0] if s else ""

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
        # 【2026-09-07 云端码形校验】音对不上码的幻觉直接丢，绝不进候选窗
        # （ilyg 案钉死：一起一个/以后有空/有了一个 读音与码全不对应）。
        kept = [w for w in items if self._cloud_code_ok(code, w)]
        if len(kept) != len(items):
            self.log("[云端校验] %s 丢弃音形不符: %s" % (
                code, " / ".join(w for w in items if w not in kept)))
            items = kept
            if not items:
                self.log("[云端校验] %s 全部不符，保持本地静态候选" % code)
                return
        base = self._cache_cands or []
        n_mb = self._cache_nmb
        # 【2026-09-07 动态调频·一修】动态模式四码起：云端答案=首选，直接
        # 压顶全部候选（含固频区）——这才是「从四码起一切动态」的正确形态。
        # 一修前把固频融化给统计代价排序，书面高频虚词霸榜（主人实测钉死：
        # 首选全是冷门词、真词不在第一页）；云端有语义+常识+上文，压顶可信。
        # 本地 0.1s 先出稳定序，云端 ~1.4s 到后升级首选。
        if self.dyn_mode and len(code) >= DYN_MIN_LEN:
            # 【2026-09-07 四码契约】云端词也是智能词，去首选；但码表四码
            # 单字固定次选起（肌肉记忆位置不漂移），不能被云端词整段压走。
            if len(code) == 4 and self.mb.loaded:
                singles4 = [w for w in self.mb.exact(code)
                            if len(w) == 1 and self._is_hanzi(w)
                            and w not in items]
                merged = (list(items) + singles4
                          + [w for w in base
                             if w not in items and w not in singles4])
            else:
                merged = list(items) + [w for w in base if w not in items]
            merged = merged[: self.n_pool]
            self._cache_cands = merged
            self._cache_nmb = 0   # 云端词视为动态区：选中走调频，不入固频
            self.q.put(("show", self.buffer, merged, 0, self._last_pos))
            self.log("[动态调频·云端压顶] %s → %s %dms" % (code, " / ".join(items), ms))
            return
        # 本地零命中（简拼整句在任何词表都召不回，cands 空）：生成结果
        # 直接成为候选窗全部内容——这正是豆包式「直接写答案」的意义。
        if not base:
            merged = list(items)[: self.n_pool]
            self._cache_cands = merged
            self._cache_nmb = 0
            self.q.put(("show", self.buffer, merged, 0, self._last_pos))
            self.log("[云端生成] %s → %s %dms" % (code, " / ".join(items), ms))
            return
        if n_mb >= len(base):
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
        elif self.ai.ready or self.cloud_judge.ready:
            self.q.put(("think", self.buffer, pos))  # 静态零命中，智能层在途
        else:
            # 【2026-09-07 主人报案「一直三个点」】AI 与云端裁判都没在途时，
            # 「…」永远不会有人来替换——挂着省略号让主人白等。直接收窗，
            # 打不出就是打不出，不装思考。
            self.q.put(("hide",))
        # 精排直送裁判（2026-09-07 主人判死速度：「不可能等一个候选六秒」）：
        # 裁判上 GPU fp16 后全池 66 句一遍 ~0.3s——两级粗筛（RBT3 每键 2s）
        # 在稳态整体退役，裁判直接全量终审，整句零截断零粗筛。裁判未就绪
        # （启动加载窗口）才回退 RBT3 词组融合（整句组保持统计序，见
        # _on_neural 门禁）。异步不阻塞击键，worker 只留最新请求。
        if len(self.buffer) >= 2 and self.buffer != DYN_SWITCH_CODE:
            self._judge_ctx = (self.ctx_tail_text or "").strip()
            # 【2026-09-07 主人判死本地】本地 Qwen/RBT3 智能从未给出可用结果，
            # 已整体删出智能链路（neural.enabled=false，本地裁判不加载）。
            # 云端生成是唯一智能层：豆包式直接写答案，_on_cloud_generate 压顶
            # 动态区（动态模式四码起压顶全部含固频）。云端未就绪/失败时保持
            # 静态结果（码表+底库+整句召回），静态层 0.1s 即时响应，云端
            # ~1.4s 到后升级首选——绝不回落本地垃圾排序。
            if self.cloud_judge.ready:
                self.cloud_judge.request_generate(self.buffer, self.ctx_tail_text)
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
        # 动态调频开关词条（odt 首选，2026-09-07 主人定案）：不真正上屏，
        # 只切换模式。缓存作废——同码在不同模式下结果不同。
        if word in (DYN_TAG_ON, DYN_TAG_OFF):
            self.dyn_mode = not self.dyn_mode
            self.buffer = ""
            self.page = 0
            self._cache_code = None
            self.q.put(("hide",))
            self.q.put(("flash", DYN_TAG_ON if self.dyn_mode else DYN_TAG_OFF))
            self.log("[动态调频] %s" % ("四码起全动态；固频词库不受影响"
                     if self.dyn_mode else "关闭（回到固频词库）"))
            return
        # 自动记忆/调频：偶数长码可还原拼音串，音节合法才入 user_dict
        # 【2026-09-07 组词造词闭环】辅码逐字锁定组出的词（象鸣）自带拼音：
        # 选中即 remember 入库，第二次打就是词库真词——「用户用辅码逐字
        # 打出来」本就是这是个词的最强证据（见下方自动造词条）。组词不是
        # 背过的码，不受 from_mabiao 豁免约束。
        py = self._composed_py.get(word)
        from_mabiao = sel_idx is not None and sel_idx < n_mb
        # 【2026-09-07 四码契约】首选位被智能词占据（_cache_smart0）时，
        # 选中首选=选智能词，照常 remember 调频——这就是四码智能调频本身
        if getattr(self, "_cache_smart0", False) and sel_idx == 0:
            from_mabiao = False
        if py:
            from_mabiao = False   # 组词不是背过的码，选中=最强造词证据
        elif self.de.loaded and not from_mabiao and len(self.buffer) >= 2 \
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
        if py and self.de.loaded:
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
        （标点/Esc/切窗/下一个多字词）里做，否则「试|作|古」会先造出中间
        垃圾词「试作古」并清链，「试作古华」永远造不出来。

        2026-09-07 主人「长词不去造了」案二修：原实现多字词入链即清链重启
        （这是→[这是]，一部→[一部]），「这是|一部」这类多字词+多字词组合
        永远凑不成链，coin_pick 的多字词组合分支是死代码——长词一个都造
        不出来。改为多字词也追加入链（链头权重闸+COIN_COMMON 兜底防句子
        流误造），组合由下一断链事件统一结算。

        非中文（英文/混合）直接断链——主人明令：不记录英文。
        """
        if not py or not self._is_hanzi(word):
            self._streak_clear()
            return
        if len(word) >= 2:
            self._coin_try()  # 旧链先结算（试|作|古|华 在此成词）
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
    # 灵鹤接管输入：关闭系统输入法打开状态，避免双输入法抢占（2026-09-07 报案）
    try:
        ime_close()
        print("[输入法] 已关闭系统输入法打开状态，灵鹤接管输入")
    except Exception as e:
        print("[输入法] 关闭系统输入法失败: %r" % e)
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
