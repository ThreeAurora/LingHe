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
from candidate_ui import CandidateWindow
from dict_engine import DictEngine
from fuma import FuMa
from mabiao import MaBiao
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

PUNCT_VKS = {
    0xBC,  # , <
    0xBE,  # . >
    0xBF,  # / ?
    0xC0,  # ` ~
    0xDB, 0xDC, 0xDD,  # [ \
}


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

        self.ai = AIEngine(cfg.get("ai", {}), log=log)
        self.ai.on_result = lambda seq: self.q.put(("ai", seq))
        if cfg.get("ai", {}).get("enabled", True):
            self.ai.probe()
        else:
            self.ai.ready = False
            log("[AI] 已按配置停用（ai.enabled=false），纯静态模式")

        self.buffer = ""
        self.enabled = True
        self.cn_mode = True          # Shift 单击切换中/英
        self.shift_t0 = 0
        self.shift_alone = False
        self.last_fg = None
        self.context = deque(maxlen=int(cfg.get("context_max", 120)))
        self.q = queue.Queue()
        self.stats = {"keys": 0, "eaten": 0, "commits": 0, "ai_hits": 0}
        self._hook = None
        self._proc = HOOKPROC(self._hook_cb)

    def _on_dict_ready(self, ok):
        if ok:
            self._cache_code = None  # 底库上线，作废旧候选
            self.q.put(("flash", "[底库就绪 %d词]" % self.de.size))
            self.log("[底库] 后台加载完成：%d 词条" % self.de.size)

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
            if n >= 2 and n % 2 == 0:
                py = " ".join(decode_syllable(code[i:i + 2]) for i in range(0, n, 2))
                dict_cands += self.de.lookup_pinyin(py, self.n_pool)
            if n >= 2 and not mb_exact:
                # 简拼召回放宽到 2 键：bz→包子、mb→面包 是最高频场景，
                # 之前 n>=3 把它们全部挡在召回之外。2 键另需更深召回：
                # by_initial['b z'] 里 豹子 排 78（470 条含重复），默认
                # n_pool=45 的截断把它挡在池外（主人 追上了一个bz→豹子 案）。
                dict_cands += self.de.lookup_initial(" ".join(code),
                                                     90 if n == 2 else self.n_pool)
            for w in dict_cands:
                if w in seen or w in pool:
                    continue
                pym = self.de.word_py.get(w) or ""
                m = max(1, len(pym.split()))
                pool[w] = self.rr.score_word(w, prevs) / m
        if n >= 2:
            # 整句切分放开到任意码长：短码也要能预测词库外组合
            vpool = []
            if n % 2 == 0 and n >= 4:
                keys = [decode_syllable(code[i:i + 2]) for i in range(0, n, 2)]
                vpool += self.rr.viterbi(keys, "py", 5, ret_cost=True, prev=prev)
            vpool += self.rr.viterbi(list(code), "ini", 5, ret_cost=True, prev=prev)
            vpool.sort(key=lambda t: t[1] / max(1, t[2]))
            for s, c, m in vpool:
                if s in seen:
                    continue
                cps = c / max(1, m)
                if s not in pool or cps < pool[s]:
                    pool[s] = cps  # 同串取更优代价（词条 vs 切分谁准谁上）
        dict_side = [w for w, _ in sorted(pool.items(), key=lambda kv: kv[1])][: self.n_pool]
        base = mb_hits + mb_prefix + dict_side
        base = base[: self.n_pool]
        n_mb = len(mb_hits) + len(mb_prefix)
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
        if info.flags & LLKHF_INJECTED:
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
        if ctrl or alt or key_down(VK_LWIN) or key_down(VK_RWIN):
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
                except Exception:
                    self.ctx_prevs = []
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
            self.q.put(("hide",))
            return self._eat()

        if vk == VK_RETURN and self.buffer:
            # 回车 = 上屏英文（字母原文直进目标应用，回车本身吞掉不换行）
            raw = self.buffer
            self.buffer = ""
            self.q.put(("hide",))
            self.stats["commits"] += 1
            send_unicode(raw)
            return self._eat()

        if self.buffer and vk in PUNCT_VKS:
            cands, n_mb = self.compute(self.buffer)
            if cands:  # 组码中标点：上屏首选，随后标点照常进应用
                self._commit(cands[0], 0, n_mb)
            return user32.CallNextHookEx(None, ncode, wparam, lparam)

        if self.buffer:
            self.buffer = ""
            self.q.put(("hide",))
        return user32.CallNextHookEx(None, ncode, wparam, lparam)

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
        if cands:
            self.q.put(("show", self.buffer, cands, n_mb, pos))
        else:
            self.q.put(("think", self.buffer, pos))  # 静态零命中，AI 在途
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
        if py:
            self.de.remember(word, py)  # 新词入库/旧词调频，批量落盘
        send_unicode(word)  # 注入事件自带 INJECTED 标志，会被钩子放行
        # 上屏联想（豆包式）：提示后面可能的词，空格上屏首选，数字键选其余
        self.liaison = None
        if self.cfg.get("liaison", True):
            n = int(self.cfg.get("liaison_count", 6))
            nxt = self.rr.next_word(word, n)
            if nxt:
                self.liaison = nxt
                self.q.put(("liaison", word, self.liaison, caret_pos()))

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="自检模式：加载码表/探测AI/装钩子后自动退出")
    ap.add_argument("--smoke-seconds", type=float, default=3.0)
    args = ap.parse_args()

    base = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, base)
    cfg = load_cfg(base)

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
    # 退出前把用户词频与 bigram 共现落盘（防丢）
    eng.de.flush()
    eng.rr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
