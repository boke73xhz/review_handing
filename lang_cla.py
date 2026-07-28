# -*- coding: utf-8 -*-
"""语言分类统计 + 筛选中英文"""
import pandas as pd
import os

ROOT = r"E:\完整项目"
OUT_DIR = os.path.join(ROOT, "lang_cla")
os.makedirs(OUT_DIR, exist_ok=True)
INFILE  = os.path.join(ROOT, "elt", "raw_reviews_clean.csv")
OUT_CLA = os.path.join(OUT_DIR, "lang_cla.csv")
OUT_LANG = os.path.join(OUT_DIR, "review_lang.csv")

KEEP = {'english', 'schinese', 'tchinese'}

# 读取
df = pd.read_csv(INFILE)
total = len(df)

# 分类统计
cla = df['language'].value_counts().reset_index()
cla.columns = ['language', 'num']
cla['percent'] = (cla['num'] / total * 100).round(2)
cla = cla.sort_values('num', ascending=False).reset_index(drop=True)

# 保存统计
cla.to_csv(OUT_CLA, index=False, encoding='utf-8-sig')
print(f"lang_cla.csv 已保存 ({len(cla)} 语种, {total} 行)")

# 筛选中英文
lang_df = df[df['language'].isin(KEEP)].copy()

lang_df.to_csv(OUT_LANG, index=False, encoding='utf-8-sig')
print(f"review_lang.csv 已保存 ({len(lang_df)} 行)")
