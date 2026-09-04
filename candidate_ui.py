# -*- coding: utf-8 -*-
"""候选窗：tkinter 无边框置顶小条，浅底黑字，跟随光标。"""

import tkinter as tk

BG = "#FBFBF8"
FG = "#1A1A1A"
HL = "#C0392B"      # 首选高亮
CODE_FG = "#8A8A8A"
BORDER = "#D8D8D2"


class CandidateWindow:
    def __init__(self, root: tk.Tk):
        self.top = tk.Toplevel(root)
        self.top.withdraw()
        self.top.overrideredirect(True)
        self.top.attributes("-topmost", True)
        self.top.configure(bg=BORDER, takefocus=0)
        self.code_var = tk.StringVar(value="")
        self.cand_var = tk.StringVar(value="")
        bar = tk.Frame(self.top, bg=BG)
        bar.pack(padx=1, pady=1)
        tk.Label(bar, textvariable=self.code_var, font=("Microsoft YaHei UI", 9),
                 fg=CODE_FG, bg=BG).pack(side="left", padx=(6, 8))
        tk.Label(bar, textvariable=self.cand_var, font=("Microsoft YaHei UI", 13),
                 fg=FG, bg=BG).pack(side="left", padx=(0, 6))
        self.visible = False

    def update(self, code: str, cands, pos=None):
        """cands: [(文本, 是否首选), ...] 已按位次排好。"""
        if not cands:
            self.hide()
            return
        self.code_var.set("[" + code + "]" if code else "")
        parts = []
        for i, (w, first) in enumerate(cands, 1):
            # 编号照常显示 1. 2. 3.（分号/单引号只是按键别名，不在窗上占位）
            parts.append(("%d.「%s」" % (i, w)) if first else ("%d.%s" % (i, w)))
        self.cand_var.set("  ".join(parts))
        self.top.update_idletasks()
        if pos:
            x, y = pos
            w, h = self.top.winfo_reqwidth(), self.top.winfo_reqheight()
            sw, sh = self.top.winfo_screenwidth(), self.top.winfo_screenheight()
            x = max(4, min(x, sw - w - 4))
            y = y + 14
            if y + h > sh - 8:
                y = max(4, y - h - 30)  # 下方放不下翻到上方
            self.top.geometry("+%d+%d" % (x, y))
        if not self.visible:
            self.top.deiconify()
            self.visible = True

    def thinking(self, code: str, pos=None):
        """码表零命中、AI 在途时的占位显示。"""
        self.code_var.set("[" + code + "]" if code else "")
        self.cand_var.set("…")
        self.top.update_idletasks()
        if pos:
            x, y = pos
            w, h = self.top.winfo_reqwidth(), self.top.winfo_reqheight()
            sw, sh = self.top.winfo_screenwidth(), self.top.winfo_screenheight()
            x = max(4, min(x, sw - w - 4))
            y = y + 14
            if y + h > sh - 8:
                y = max(4, y - h - 30)
            self.top.geometry("+%d+%d" % (x, y))
        if not self.visible:
            self.top.deiconify()
            self.visible = True

    def flash(self, text: str, pos=None):
        """短暂提示（如中英切换 [中]/[EN]），由主循环定时收回。"""
        self.code_var.set("")
        self.cand_var.set(text)
        self.top.update_idletasks()
        if pos:
            x, y = pos
            self.top.geometry("+%d+%d" % (x, y + 14))
        if not self.visible:
            self.top.deiconify()
            self.visible = True

    def hide(self):
        if self.visible:
            self.top.withdraw()
            self.visible = False
