import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tkinter as tk
import linghe
from linghe import Engine, send_unicode
base = os.path.dirname(os.path.abspath(linghe.__file__))
root = tk.Tk(); root.withdraw()
eng = Engine(base, {"ai": {"mode": "off"}, "mabiao_dir": "__none__"}, log=lambda s: None)
assert eng.install_hook(), "hook install failed"
def after():
    before = eng.stats["keys"]
    send_unicode("灵鹤")
    def later():
        gained = eng.stats["keys"] - before
        print("钩子观察到的注入事件:", gained, "(期望 4 = 2字 x down+up)")
        print("PASS" if gained == 4 else "FAIL")
        eng.uninstall_hook(); root.destroy()
    root.after(400, later)
root.after(400, after)
root.mainloop()
