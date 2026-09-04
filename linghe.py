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
VK_LWIN, VK_RWIN = 0x5B, 0x5C
VK_L = 0x4C
VK_OEM_1 = 0xBA    # ; :
VK_OEM_7 = 0xDE    # ' "

PUNCT_VKS = {
    0xBB,  # = +
    0xBC,  # , <
    0xBD,  # - _
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

        self.ai = AIEngine(cfg.get("ai", {}), log=log)
        self.ai.on_result = lambda seq: self.q.put(("ai", seq))
        self.ai.probe()

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
            self.q.put(("flash", "[底库就绪 %d词]" % self.de.size))
            self.log("[底库] 后台加载完成：%d 词条" % self.de.size)

    # ---- 候选组装：排序公式的唯一实现 ----
    def _fused_candidates(self, code):
        """辅码筛选词（最高优先）：多假设解析出 音节+辅码 解释，
        查码表词后校验辅码（mid=第 idx 字，word=-1 任意字），通过者按固频序。"""
        if not self.fm.loaded or len(code) < 3:
            return []
        out, seen = [], set()
        for syls, fuses in key_parser.parse(code):
            if not fuses or len(syls) < 2:
                continue
            ss = "".join(syls)
            if len(ss) > 4:
                continue  # 词最长 4 码
            for w in self.mb.exact(ss):
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
            if len(out) >= self.n_show:
                break
        return out

    def compute(self, code):
        # 第 1 优先：辅码筛选出的词（打了辅码 = 用户在钉死这个词）
        fused = self._fused_candidates(code)
        # 第 2 优先：码表侧（精确码 + 前缀命中，固频序）
        mb_hits = list(self.mb.exact(code))
        seen = set(fused) | set(mb_hits)
        for w in self.mb.prefix(code, self.n_show * 3):
            if w not in seen:
                mb_hits.append(w)
                seen.add(w)
        base = fused + mb_hits
        # 第 3 优先：底库（152 万词，后台就绪后生效）——码表没有的词在这里出，
        # 如 6 码 vhmusi -> "zhan mu si" -> 詹姆斯；奇数长（第5码起续码中）用简拼兜
        n = len(code)
        if self.de.loaded and len(base) < self.n_show:
            if n >= 2 and n % 2 == 0:
                py = " ".join(decode_syllable(code[i:i + 2]) for i in range(0, n, 2))
                for w in self.de.lookup_pinyin(py, self.n_show):
                    if w not in seen:
                        base.append(w)
                        seen.add(w)
            if len(base) < self.n_show and n >= 3 and not self.mb.exact(code) and not self.mb.prefix(code, 1):
                for w in self.de.lookup_initial(" ".join(code), self.n_show):
                    if w not in seen:
                        base.append(w)
                        seen.add(w)
        base = base[: self.n_show]
        n_mb = len(base)
        # AI 顺延：静态引擎命中 k 个，AI 从第 k+1 位起
        merged = base[:]
        for w in self.ai.peek(self.context_key(), code):
            if w not in merged:
                merged.append(w)
        return merged[: self.n_show], n_mb

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

        # Shift 单击 = 中英切换（按下后 400ms 内无其他键、抬起时触发）
        if vk == VK_SHIFT:
            if msg in (WM_KEYDOWN, WM_SYSKEYDOWN):
                self.shift_t0 = kernel32.GetTickCount64()
                self.shift_alone = True
            elif msg == WM_KEYUP and self.shift_alone:
                self.shift_alone = False
                if self.enabled and kernel32.GetTickCount64() - self.shift_t0 <= 400:
                    self.cn_mode = not self.cn_mode
                    self.buffer = ""
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

        if not self.cn_mode:  # 英文态：全部透传
            return user32.CallNextHookEx(None, ncode, wparam, lparam)

        if is_letter:
            if key_down(VK_SHIFT):  # 大写意图：放弃组码，透传
                if self.buffer:
                    self.buffer = ""
                    self.q.put(("hide",))
                return user32.CallNextHookEx(None, ncode, wparam, lparam)
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
                cands, _ = self.compute(self.buffer)
                if cands:
                    self._commit(cands[0])
                else:
                    self.buffer = ""
                    self.q.put(("hide",))
                return self._eat()
            return user32.CallNextHookEx(None, ncode, wparam, lparam)

        if is_digit and self.buffer:
            idx = vk - 0x31
            cands, _ = self.compute(self.buffer)
            if idx < len(cands):
                self._commit(cands[idx])
            return self._eat()

        if vk in (VK_OEM_1, VK_OEM_7) and self.buffer:
            # 分号=2选，单引号=3选（音形重码选择键惯例）
            idx = 1 if vk == VK_OEM_1 else 2
            cands, _ = self.compute(self.buffer)
            if len(cands) > idx:
                self._commit(cands[idx])
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
            cands, _ = self.compute(self.buffer)
            if cands:  # 组码中标点：上屏首选，随后标点照常进应用
                self._commit(cands[0])
            return user32.CallNextHookEx(None, ncode, wparam, lparam)

        if self.buffer:
            self.buffer = ""
            self.q.put(("hide",))
        return user32.CallNextHookEx(None, ncode, wparam, lparam)

    def _eat(self):
        self.stats["eaten"] += 1
        return 1  # 吞掉，不让原始键进入目标应用

    def _after_edit(self):
        if not self.buffer:
            self.q.put(("hide",))
            return
        cands, n_mb = self.compute(self.buffer)
        pos = caret_pos()
        if cands:
            self.q.put(("show", self.buffer, cands, n_mb, pos))
        else:
            self.q.put(("think", self.buffer, pos))  # 码表零命中且 AI 在途
        if not n_mb or len(cands) < self.n_show or len(self.buffer) > 4:
            # 长码（第5码起纯双拼续词）主要靠 AI 出长词
            self.ai.request(self.buffer, "".join(self.context))

    def _commit(self, word):
        # 自动记忆/调频：偶数长码可还原拼音串，音节合法才入 user_dict
        py = None
        if self.de.loaded and len(self.buffer) >= 2 and len(self.buffer) % 2 == 0:
            syls = [decode_syllable(self.buffer[i:i + 2]) for i in range(0, len(self.buffer), 2)]
            cand = " ".join(syls)
            if all(s in self.de.valid_sylls for s in syls):
                py = cand
        self.buffer = ""
        self.q.put(("hide",))
        self.context.append(word)
        self.stats["commits"] += 1
        if py:
            self.de.remember(word, py)  # 新词落盘 user_dict.txt，旧词会话内调频
        send_unicode(word)  # 注入事件自带 INJECTED 标志，会被钩子放行

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
                    shown = [(w, i == 0) for i, w in enumerate(cands)]
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
                elif kind == "ai":
                    # AI 结果返回：用当前键入码刷新（码表 k 项之后顺延插入）
                    if eng.buffer:
                        cands, n_mb = eng.compute(eng.buffer)
                        shown = [(w, i == 0) for i, w in enumerate(cands)]
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
