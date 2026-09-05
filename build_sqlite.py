# -*- coding: utf-8 -*-
"""把文本词库导出成 SQLite 数据库——这才是「瞬间加载」的正解。

为什么 marshal/pickle 都救不了加载
----------------------------------
marshal 缓存实测 26s，文本解析 41s。瓶颈不是解析，是**反序列化**：
152 万（现已 238 万）个词条要逐个重建 Python str 对象、装进 dict。
只要还要「把所有词读进内存」，就必然是几十秒量级。

SQLite 的做法完全不同
----------------------
不读进内存，而是让**操作系统按需分页映射**数据库文件（mmap）：
  - 打开数据库 = 读个文件头，微秒级完成，与词库大小无关；
  - 查询走 B 树索引，只把用到的那几个页面从磁盘调进来（且有 OS 页缓存）；
  - 这正是豆包/搜狗二进制词库（mmap 双数组 Trie）的原理，SQLite 是标准库
    能直接白嫖的等价物，零第三方依赖。

代价与取舍
----------
- 单次查询 ~1-5µs（vs 内存 dict ~0.05µs），慢 20~100 倍；
- 但一次按键只查几十~几百次（Viterbi 跨度×候选），合计仍 <2ms，
  且配合 LRU 缓存后热词退化到接近 dict；
- 换来的：启动从 40s → 近乎 0，且常驻内存从数百 MB 降到几 MB。

用法：
    python build_sqlite.py            # 生成 dicts/lexicon.db
"""
import os
import sqlite3
import sys
import time

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from dict_engine import DictEngine, _sm_key  # noqa: E402

DB_PATH = os.path.join(BASE, "dicts", "lexicon.db")


def build():
    # 复用 DictEngine 的全部解析逻辑（两列/三列格式、自动注音、用户词合并），
    # 只是把最终产物从「内存 dict」换成「数据库文件」。这次慢是一次性成本。
    de = DictEngine(log=print)
    t0 = time.perf_counter()
    de.load_dir(os.path.join(BASE, "dicts"))
    print("[构建] 文本词库解析完成 %.1fs，%d 词条" % (time.perf_counter() - t0, len(de.word_py)))

    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=OFF")   # 构建期不需要事务日志
    cur.execute("PRAGMA synchronous=OFF")
    cur.execute("""
        CREATE TABLE words (
            word   TEXT PRIMARY KEY,
            py     TEXT NOT NULL,
            ini    TEXT NOT NULL,
            weight INTEGER NOT NULL
        )
    """)
    # 覆盖索引：查询只需索引即可返回，不必回表
    cur.execute("CREATE INDEX idx_py  ON words(py,  weight DESC)")
    cur.execute("CREATE INDEX idx_ini ON words(ini, weight DESC)")

    rows = []
    t0 = time.perf_counter()
    n = 0
    for word, py in de.word_py.items():
        ini = " ".join(_sm_key(s) for s in py.split())
        rows.append((word, py, ini, int(de.word_weight.get(word, 0))))
        if len(rows) >= 50000:
            # 同词多音/多来源会冲突：保留权重最高的一条
            cur.executemany(
                "INSERT INTO words(word,py,ini,weight) VALUES(?,?,?,?) "
                "ON CONFLICT(word) DO UPDATE SET weight=MAX(weight, excluded.weight)",
                rows)
            n += len(rows)
            rows.clear()
    if rows:
        cur.executemany(
            "INSERT INTO words(word,py,ini,weight) VALUES(?,?,?,?) "
            "ON CONFLICT(word) DO UPDATE SET weight=MAX(weight, excluded.weight)",
            rows)
        n += len(rows)
    con.commit()
    print("[构建] 写入 %d 条，%.1fs" % (n, time.perf_counter() - t0))

    t0 = time.perf_counter()
    cur.execute("ANALYZE")
    con.commit()
    print("[构建] 索引分析 %.1fs" % (time.perf_counter() - t0))

    cur.execute("SELECT COUNT(*) FROM words")
    total = cur.fetchone()[0]
    con.close()
    print("[构建] 完成：%s  共 %d 词  %.1f MB"
          % (DB_PATH, total, os.path.getsize(DB_PATH) / 1048576))
    print("[提示] 让 dict_engine 走数据库：config.json 里设 \"dict_backend\": \"sqlite\"")


if __name__ == "__main__":
    build()
