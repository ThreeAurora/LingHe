# -*- coding: utf-8 -*-
"""万象语法模型端侧加载器（Rime::Grammar/1.0 + Darts-clone 0.32）。

直接 mmap 读取 .gram 二进制，不依赖任何 Rime 框架，查询微秒级。
对照 librime-octagram 的二进制语义实现（仅对照格式，不复制其代码）：

文件布局：
    [0..31]    char format[32]                  "Rime::Grammar/1.0"
    [32..35]   uint32 db_checksum
    [36..39]   uint32 double_array_size         Darts 单元数（4 字节/单元）
    [40..43]   int32  double_array 偏移（相对自身地址 40）
    [44..]     Darts 双数组单元流（小端 uint32）

Darts-clone 0.32 单元位布局：
    label   = u & 0x800000FF     （bit31 为叶子标记）
    offset  = (u>>10) << ((u&0x200)>>6)
    value   = u & 0x7FFFFFFF
    has_leaf= (u>>8) & 1
    子节点位置 = node ^ offset ^ label（XOR 编码，非加法）
    叶子值   = 单元[node ^ offset] 的 value

键 = gram_encoding::encode(中文短语)，值 = int(log(频率) * 10000)。

查询语义（Octagram::Query 复刻，默认 GrammarConfig）：
    - 回看 n = min(8, collocation_max_length-1=8) 个字做上文；
    - 对每个上文后缀：traverse(context) 后 commonPrefixSearch(word 前 n 字)，
      取所有匹配中「模型分 + 搭配惩罚」的最大值；
    - is_rear：word 后有 "$" 键（句尾）则再比一轮 + rear_penalty；
    - context 为空或全不匹配 → non_collocation_penalty(-12)。
"""
import mmap
import struct

K_VALUE_SCALE = 10000
K_MAX_RESULTS = 8

# Octagram::GrammarConfig 默认值
# C_MAX_LEN 从默认 4 调到 9：万象是 LTS（长文本）模型，实测 8 字回看窗口
# 证据更强（我今天想吃|西湖醋鱼 3 字回看=6.7 → 8 字=10.3），查询仍微秒级。
C_MAX_LEN = 9          # collocation_max_length（回看 n = min(8, C_MAX_LEN-1)）
C_MIN_LEN = 3          # collocation_min_length
C_PEN = -12.0          # collocation_penalty
NC_PEN = -12.0         # non_collocation_penalty
WEAK_PEN = -24.0       # weak_collocation_penalty
REAR_PEN = -18.0       # rear_penalty

MAGIC = b"Rime::Grammar/"


class GramModel:
    """万象 .gram 语法模型。构造后即可 Query，线程安全（只读）。"""

    __slots__ = ("_mm", "_file", "_base", "_size", "_u32")

    def __init__(self, path):
        self._file = open(path, "rb")
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        fmt = self._mm[:32].split(b"\0")[0]
        if not fmt.startswith(MAGIC):
            raise ValueError("不是 Rime::Grammar 文件: %r" % fmt)
        checksum, da_size, da_off = struct.unpack_from("<III", self._mm, 32)
        self._size = da_size
        self._base = 40 + da_off          # 双数组起始
        self._u32 = struct.Struct("<I").unpack_from

    # ---------- 单元访问 ----------
    def _unit(self, idx):
        return self._u32(self._mm, self._base + (idx & 0xFFFFFFFF) * 4)[0]

    # ---------- 查询 ----------
    def _traverse(self, key, node_pos=0):
        """沿 key 逐字节走；中途 label 不符返回 None，否则返回最终节点。"""
        u = self._unit(node_pos)
        for c in key:
            node_pos ^= ((u >> 10) << ((u & 0x200) >> 6)) ^ c
            u = self._unit(node_pos)
            if (u & 0x800000FF) != c:
                return None
        return node_pos

    def _common_prefix(self, key, node_pos):
        """从 node_pos 起对 key 做公共前缀搜索，返回 [(value, 字节数)] ≤8 条。"""
        u = self._unit(node_pos)
        node_pos ^= (u >> 10) << ((u & 0x200) >> 6)
        out = []
        for i, c in enumerate(key):
            node_pos ^= c
            u = self._unit(node_pos)
            if (u & 0x800000FF) != c:
                break
            node_pos ^= (u >> 10) << ((u & 0x200) >> 6)
            if (u >> 8) & 1:
                out.append((self._unit(node_pos) & 0x7FFFFFFF, i + 1))
                if len(out) >= K_MAX_RESULTS:
                    break
        return out

    def _lookup(self, context, word):
        """GramDb::Lookup：traverse(context) 后对 word 做公共前缀搜索。"""
        np = self._traverse(context)
        if np is None:
            return []
        return self._common_prefix(word, np)

    def query(self, context, word, is_rear=False):
        """复刻 Octagram::Query。context=上屏中文原文，word=候选原文。"""
        n = min(8, C_MAX_LEN - 1)
        ctx_tail = context[-n:] if context else ""
        word_head = word[:n]
        ctx_enc = encode(ctx_tail)
        word_enc = encode(word_head)
        ctx_uni = len(ctx_tail)
        result = NC_PEN
        for ctx_len in range(ctx_uni, 0, -1):
            sub = encode(ctx_tail[-ctx_len:])
            for val, mlen in self._lookup(sub, word_enc):
                m_uni = unicode_len(word_enc, mlen)
                colloc = ctx_len + m_uni
                whole = (len(sub) == len(ctx_enc)) and (mlen == len(word_enc))
                pen = C_PEN if (colloc >= C_MIN_LEN or whole) else WEAK_PEN
                sc = (val / K_VALUE_SCALE if val >= 0 else -1.0) + pen
                if sc > result:
                    result = sc
        if is_rear:
            wl = len(word)
            if len(word_enc) == wl:
                m = self._lookup(word_enc, b"$")
                if m:
                    val = m[0][0]
                    sc = (val / K_VALUE_SCALE if val >= 0 else -1.0) + REAR_PEN
                    if sc > result:
                        result = sc
        return result


# ---------- gram_encoding::encode（对照 librime-octagram 语义） ----------
def encode(text):
    out = []
    for ch in text:
        u = ord(ch)
        if u < 0x80:
            out.append(0xE0 if u == 0 else u)
        elif 0x4000 <= u < 0xA000:
            if (u & 0xFF) == 0:
                out.append(0xE1)
                out.append((u >> 8) + 0x40)
            else:
                out.append((u >> 8) + 0x40)
                out.append(u & 0xFF)
        else:
            bits = 32
            v = u
            while bits > 0 and (v & 0xFE000000) == 0:
                bits -= 7
                v <<= 7
            nb = (bits + 6) // 7
            out.append(0xE0 | nb)
            while nb > 0:
                nb -= 1
                out.append(((v >> 25) & 0x7F) | 0x80)
                v <<= 7
    return bytes(out)


def unicode_len(enc, length):
    n = 0
    i = 0
    while i < length:
        b = enc[i]
        if b & 0x80 == 0:
            i += 1
        elif b & 0xF0 == 0xE0:
            i += (b & 0x0F) + 1
        else:
            i += 2
        n += 1
    return n


if __name__ == "__main__":
    import sys
    import time

    path = sys.argv[1] if len(sys.argv) > 1 else r"E:\AAAAA\wanxiang-lts-zh-hans.gram"
    t0 = time.time()
    gm = GramModel(path)
    print("加载 %.0fms" % ((time.time() - t0) * 1000))
    for ctx, w, rear in [
        ("我今天想吃", "西湖醋鱼", False),
        ("我今天想吃", "酸奶", False),
        ("我今天想", "吃", False),
        ("", "吃", False),
        ("这是我", "的", True),
    ]:
        print("Q(%s|%s%s) = %.3f" % (ctx, w, "$" if rear else "", gm.query(ctx, w, rear)))
    t = time.time()
    for _ in range(1000):
        gm.query("我今天想吃", "西湖醋鱼")
    print("1000 Query = %.1fms" % ((time.time() - t) * 1000))
