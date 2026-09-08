# -*- coding: utf-8 -*-
"""训练数据生成器：语料句 → 词(金标) → 模拟击键 → 引擎真实召回 → 样本 JSONL。

数据流（对应主人真实打字）：
  语料(预分词, 词边界=金标) → 词→拼音(词库级 word_py 优先, 逐字 char_pinyin 兜底)
  → 每字双拼 2 键 (xiaohe_enc) + 辅码首键 (fuma, 多音字随机取)
  → 两种击键形态: 纯音 / 音+辅(模式3) → eng.compute() 真实召回池
  → 金标词在池内 top32 ⇒ 样本; 不在 ⇒ 漏召(统计, 丢弃)

输出 JSONL: {"code","mode","ctx","cands","gold","gold_idx","n_fixed","wlen"}
用法: python gen_data.py --lines 200 --out train/data/sample.jsonl
"""
import os, sys, json, random, argparse, time

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)
os.chdir(BASE)
sys.path.insert(0, os.path.join(BASE, "train"))

from linghe import Engine, load_cfg
from fuma import FuMa
from xiaohe_enc import encode_syllable

TONE = str.maketrans("āáǎàēéěèīíǐìōóǒòūúǔùǖǘǚǜ", "aaaaeeeeiiiioooouuuuvvvv")
PUNCT = set("，。！？；：、""''（）《》〈〉—…·「」『』【】")


def is_hanzi(tok):
    return tok and all("\u4e00" <= c <= "\u9fff" for c in tok)


def load_char_pinyin(path):
    """char_pinyin.txt -> {字: [读音...]}（无调）。"""
    m = {}
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or not line.startswith("U+"):
                continue
            head, rest = line.split(":", 1)
            pys = rest.split("#")[0]
            ch = chr(int(head[2:], 16))
            m[ch] = [p.strip().translate(TONE) for p in pys.split(",") if p.strip()]
    return m


class Gen:
    def __init__(self, seed=2026):
        cfg = load_cfg(BASE)
        cfg.setdefault("neural", {})["enabled"] = False
        self.eng = Engine(BASE, cfg, log=lambda s: None)
        self.eng._lite = True      # 跳过整句召回通道（词级样本用不到，提速 5-10x）
        self.eng.n_pool = 96       # 深池：金标可被压到 45+，重排器学「捞」
        self.fm = FuMa()
        self.fm.load_dir(os.path.join(BASE, "mabiao"))
        self.char_py = load_char_pinyin(os.path.join(BASE, "dicts", "char_pinyin.txt"))
        self.rng = random.Random(seed)
        self._fm_cache = {}
        # 统计
        self.stat = {"skip_len": 0, "skip_py": 0, "missed": 0, "hit": 0}

    def fuma_first(self, ch):
        """字的辅码首键（多音字随机取其一；无辅码返回 None）。"""
        if ch in self._fm_cache:
            return self._fm_cache[ch]
        keys = self.fm.first_keys(ch)
        v = self.rng.choice(sorted(keys)) if keys else None
        self._fm_cache[ch] = v
        return v

    def word_pys(self, word):
        """词 -> 拼音列表（词库级优先，逐字兜底）。失败返回 None。"""
        py = self.eng.de.word_py.get(word)
        if py:
            syls = py.split()
            if len(syls) == len(word):
                return syls
        out = []
        for ch in word:
            cands = self.char_py.get(ch)
            if not cands:
                return None
            out.append(cands[0])
        return out

    def keystrokes(self, word):
        """词 -> [(code, mode)] 三种击键形态（主人真实打法）。
        py : 双拼全码每字 2 键（xlmk 式）
        fm : 每字音 2 + 辅 1（模式 3，辅码锁定）
        ini: 声母简拼每字 1 键（bz 式，双拼声母键 zh→v/ch→i/sh→u）"""
        pys = self.word_pys(word)
        if not pys or len(pys) > 4:
            self.stat["skip_len" if len(pys or ()) > 4 else "skip_py"] += 1
            return []
        keys = [encode_syllable(p) for p in pys]
        if any(not k for k in keys):
            self.stat["skip_py"] += 1
            return []
        fus = [self.fuma_first(c) for c in word]
        forms = [("".join(keys), "py")]
        if all(fus):
            forms.append(("".join(k + f for k, f in zip(keys, fus)), "fm"))
        if len(word) >= 2:
            ini = "".join({"zh": "v", "ch": "i", "sh": "u"}.get(p[:2], p[0])
                          for p in pys)
            forms.append((ini, "ini"))
        return forms

    def make_samples(self, word, ctx):
        """词 -> [样本 dict]（每种命中形态各一个，不做 best 筛选）。

        2026-09-07 定案：旧的 best 逻辑总取金标最浅形态，导致 fm 63万/
        ini 352 的极端失衡（辅码锁定天然 rank 浅）+ 金标 95% 在 rank0。
        真实分布是：主人有时打全辅、有时只打双拼、更多时候打简拼——
        每种形态的池和难度都该进数据，深位样本是重排器的价值所在。
        """
        out = []
        for code, mode in self.keystrokes(word):
            cands, n_mb = self.eng.compute(code)
            # 滤掉码表功能模板词条（$ddcmd 日期/农历模板等，非可选词，
            # len>127 还会撑爆 memmap int8 wlen —— 09-08 训练崩溃根因）
            cands = [w for w in cands
                     if isinstance(w, str) and not any(c in w for c in "$<#(")]
            try:
                gi = cands.index(word)
            except ValueError:
                gi = -1
            if 0 <= gi < 48:
                out.append({"code": code, "mode": mode, "ctx": ctx,
                            "cands": cands[:48], "gold": word, "gold_idx": gi,
                            "n_fixed": n_mb, "wlen": len(word)})
        if out:
            self.stat["hit"] += 1
            return out
        self.stat["missed"] += 1
        return []

    def run(self, corpus_paths, out_path, max_lines=None,
            downsample_single=0.35, start=0, append=False):
        t0 = time.time()
        n_out = 0
        mode = "a" if append else "w"
        with open(out_path, mode, encoding="utf-8") as fo:
            for path in corpus_paths:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for li, line in enumerate(f):
                        if li < start:          # 断点续传：跳过已完成行
                            continue
                        if max_lines and li >= max_lines:
                            break
                        toks = line.split()
                        ctx_words = []
                        for tok in toks:
                            if not is_hanzi(tok):
                                if tok and tok[0] in PUNCT:
                                    ctx_words = []      # 标点断句
                                continue
                            if ctx_words:              # 句首词无上文，跳过生成但入上下文
                                if len(tok) == 1 and \
                                        self.rng.random() > downsample_single:
                                    ctx_words.append(tok)
                                    continue
                                for s in self.make_samples(tok, "".join(ctx_words)[-12:]):
                                    fo.write(json.dumps(s, ensure_ascii=False) + "\n")
                                    n_out += 1
                            ctx_words.append(tok)
                        if max_lines and li >= max_lines:
                            break
                        if li % 5000 == 0:
                            print("  [%s] line %d  out=%d  stat=%s  %.0fs"
                                  % (os.path.basename(path), li, n_out,
                                     self.stat, time.time() - t0), flush=True)
        print("DONE %d samples, stat=%s, %.0fs" % (n_out, self.stat, time.time() - t0))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=int, default=0, help="每文件最多行数(0=全部)")
    ap.add_argument("--out", default="train/data/gen.jsonl")
    ap.add_argument("--only", default="", help="只跑文件名含此子串的语料")
    ap.add_argument("--start", type=int, default=0, help="跳过每文件前 N 行(断点续传)")
    ap.add_argument("--append", action="store_true", help="输出追加模式")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    g = Gen()
    files = [p for p in ["cache/tmp/corpus/msr_training.utf8",
                         "cache/tmp/corpus/pku_training.utf8"]
             if a.only.lower() in os.path.basename(p).lower()]
    g.run(files or ["cache/tmp/corpus/msr_training.utf8",
                    "cache/tmp/corpus/pku_training.utf8"],
          a.out, a.lines or None, start=a.start, append=a.append)
