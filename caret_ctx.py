# -*- coding: utf-8 -*-
"""读取目标应用「光标处上下文」——豆包「指哪打哪」的关键能力。

为什么需要这个模块
------------------
孤立预测的天花板很低：同样是敲 `dnynkcltydqd`（但你也能看出来它有多强大），
纯靠语言模型是在全空间里盲猜；而豆包会先读光标前后已有的文字，
把候选压到「这句话接着该说什么」的窄空间里，所以几乎不出错。

灵鹤原本只有 self.context（**自己上屏的**历史，切窗口即清空），
完全没读目标应用里已经存在的文字——这是与豆包最大的体验差距。

实现方式（零第三方依赖，纯 ctypes）
------------------------------------
1. GetGUIThreadInfo  → 取真正的焦点控件句柄（比 GetForegroundWindow 精确，
   能拿到输入框子窗口，而不是顶层窗口）；
2. WM_GETTEXT        → 读控件全文；
3. EM_GETSEL         → 读光标/选区位置；
4. 按位置切出光标前后文。

局限与兜底
----------
- 标准 Win32 Edit / RichEdit 有效（记事本、多数桌面软件输入框、IDE）；
- 自绘控件（浏览器地址栏、微信输入框、Word）可能返回空串——此时静默降级为
  「无上下文」，不影响打字，只是少了这层加成；
- 超长文本（>TEXT_LIMIT）直接放弃，避免大 buffer 拷贝卡住按键；
- 读取有开销，调用方应当缓存（见 Engine 的「每组码只读一次」策略）。
"""

import ctypes
from ctypes import wintypes

user32 = ctypes.WinDLL("user32", use_last_error=True)

WM_GETTEXT = 0x000D
WM_GETTEXTLENGTH = 0x000E
EM_GETSEL = 0x00B0

TEXT_LIMIT = 200000  # 超过此长度的文本放弃读取（防大文档卡按键）


class GUITHREADINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hwndActive", wintypes.HWND),
        ("hwndFocus", wintypes.HWND),     # 真正的焦点控件
        ("hwndCapture", wintypes.HWND),
        ("hwndMenuOwner", wintypes.HWND),
        ("hwndMoveSize", wintypes.HWND),
        ("hwndCaret", wintypes.HWND),
        ("rcCaret", wintypes.RECT),
    ]


user32.GetGUIThreadInfo.argtypes = [wintypes.DWORD, ctypes.POINTER(GUITHREADINFO)]
user32.GetGUIThreadInfo.restype = wintypes.BOOL
user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
user32.GetWindowThreadProcessId.restype = wintypes.DWORD
user32.GetForegroundWindow.argtypes = []
user32.GetForegroundWindow.restype = wintypes.HWND

# SendMessageW 需要两套签名（同为 64 位地址，但参数语义不同）：
# 一个第 4 参是缓冲区指针、第 3 参是整数长度；另一个第 3/4 参都是 out 指针。
# 用 WINFUNCTYPE 建原型，避免 argtypes 互相覆盖，也保证 64 位地址不被截断。
_proto_text = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT,
                                 wintypes.WPARAM, ctypes.c_void_p)
_send_text = _proto_text(("SendMessageW", user32))

_proto_sel = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT,
                                ctypes.c_void_p, ctypes.c_void_p)
_send_sel = _proto_sel(("SendMessageW", user32))


def focused_hwnd():
    """当前焦点控件句柄（失败返回 0）。"""
    try:
        fg = user32.GetForegroundWindow()
        if not fg:
            return 0
        tid = user32.GetWindowThreadProcessId(fg, None)
        info = GUITHREADINFO()
        info.cbSize = ctypes.sizeof(GUITHREADINFO)
        if user32.GetGUIThreadInfo(tid, ctypes.byref(info)):
            return info.hwndFocus or info.hwndActive or 0
        return int(fg)
    except Exception:
        return 0


def read_context(before=64, after=16):
    """读光标前后文，返回 (光标前文字, 光标后文字)。读不到返回 ('', '')。

    before/after 是要取的前后字数。整段文本过长时直接放弃（返回空）以免卡顿。
    """
    hwnd = focused_hwnd()
    if not hwnd:
        return "", ""
    try:
        n = _send_text(hwnd, WM_GETTEXTLENGTH, 0, None)
        if n <= 0 or n > TEXT_LIMIT:
            return "", ""
        buf = ctypes.create_unicode_buffer(n + 1)
        _send_text(hwnd, WM_GETTEXT, n + 1, ctypes.cast(buf, ctypes.c_void_p))
        text = buf.value
        if not text:
            return "", ""
        start = wintypes.DWORD()
        end = wintypes.DWORD()
        _send_sel(hwnd, EM_GETSEL,
                  ctypes.cast(ctypes.byref(start), ctypes.c_void_p),
                  ctypes.cast(ctypes.byref(end), ctypes.c_void_p))
        pos = start.value
        if pos < 0 or pos > len(text):
            pos = len(text)
        return text[max(0, pos - before):pos], text[pos:pos + after]
    except Exception:
        return "", ""


def last_word(before_text, dict_words=None, max_len=4):
    """从光标前文字里切出「最后一个词」，作为 bigram 上文。

    dict_words: 一个支持 `in` 判断的词集合（本引擎传 de.word_py）。
    用贪心最长匹配（4→1 字），命中词库即返回，没命中就退到最后一个字。
    """
    if not before_text:
        return ""
    tail = before_text[-max_len:]
    if dict_words:
        for k in range(len(tail), 0, -1):
            w = tail[-k:]
            if w in dict_words and len(w) >= 1:
                # 单字词只有在多字词都没命中时才用，优先长词
                if k > 1 or not any(tail[-j:] in dict_words for j in range(min(len(tail), max_len), 1, -1)):
                    return w
    return tail[-1] if tail else ""


def _hanzi(s):
    return all(0x4E00 <= ord(c) <= 0x9FFF for c in s)


def tail_words(before_text, dict_words=None, k=3, chunk=2):
    """切出光标前最后 k 个词（右起贪心，块长 ≤chunk，词库外退单字）。

    为什么块长限 2 而不是 last_word 的 4：万象长词表把 吃了/吃了一个 这类
    短语收作词，4 字贪心会把动词整个吞掉（「我吃了一个」→ 上文=吃了一个，
    bigram 左键是短语，任何共现表都查不到）。限 2 能切出
    [一个, 吃了] / [一个, 了, 打碎] 这种带动词的尾部结构，供多上文衰减
    打分用（rerank._bi_bonus_multi）。

    停止条件：撞到非汉字（标点/字母/空格）即停——跨句共现无意义。
    """
    if not before_text:
        return []
    out = []
    pos = len(before_text)
    while pos > 0 and len(out) < k:
        w = None
        if pos >= 2:
            two = before_text[pos - 2:pos]
            if _hanzi(two) and (dict_words is None or two in dict_words):
                w = two
        if w is None:
            one = before_text[pos - 1]
            if not _hanzi(one):
                break
            w = one
        out.append(w)
        pos -= len(w)
    return out


if __name__ == "__main__":
    b, a = read_context()
    print("光标前:", repr(b))
    print("光标后:", repr(a))
    print("末词  :", repr(last_word(b)))
