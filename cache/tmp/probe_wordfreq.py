# -*- coding: utf-8 -*-
"""wordfreq 口语词频验证：xi 组竞争词的 zipf 序、目标词的相对位次。"""
from wordfreq import zipf_frequency

groups = {
    "xi组(想吃 rank11?)": ['想吃', '形成', '宣传', '新车', '现场', '消除', '行程', '县城', '薪酬', '写出', '下车'],
    "dx组(东西 rank2)": ['大型', '大学', '东西', '对象'],
    "其他": ['测试一下', '西湖醋鱼', '那个东西', '今天', '措施', '从事', '使用', '一些', '影响', '规定',
           '下次', '乡村', '内外', '那位', '能够', '高度', '测试', '一下', '文件', '危机', '体现出'],
}
for g, ws in groups.items():
    print("=" * 40)
    print(g)
    ranked = sorted(ws, key=lambda w: -zipf_frequency(w, 'zh'))
    for w in ranked:
        print("  %-6s zipf=%.2f" % (w, zipf_frequency(w, 'zh')))
