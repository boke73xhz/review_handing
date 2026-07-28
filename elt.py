#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据清洗 ETL — elt.py
======================
基于 review 文本长度的百分位分析，自动过滤异常评论。

参考：E:\完整项目\test.ipynb

用法：
  python elt.py                          # 默认 P2 / P99 阈值
  python elt.py --lo 1 --hi 99           # 自定义百分位
  python elt.py --lo_pct 1 --hi_pct 99
  python elt.py --min_len 3 --max_len 2000  # 绝对字符数阈值

输出：elt/raw_reviews_clean.csv、raw_reviews_low.csv、raw_reviews_high.csv
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd

# ── 路径 ──────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE  = os.path.join(ROOT, "crawler", "raw_reviews.csv")
OUTPUT_FILE = os.path.join(ROOT, "elt", "raw_reviews_clean.csv")
LOW_FILE    = os.path.join(ROOT, "elt", "raw_reviews_low.csv")
HIGH_FILE   = os.path.join(ROOT, "elt", "raw_reviews_high.csv")
REPORT_FILE = os.path.join(ROOT, "elt", "elt_report.txt")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

# ── 日志 ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("elt")


def load_data(path: str) -> pd.DataFrame:
    """读取 CSV，清理 review 文本，并计算长度"""
    log.info("读取: %s", path)
    df = pd.read_csv(path)

    # 文本清洗：换行 → 空格，压缩连续空格
    log.info("清洗 review 文本：换行符→空格，压缩连续空格")
    df["review"] = (
        df["review"]
        .fillna("")
        .str.replace(r"\r?\n", " ", regex=True)
        .str.replace(r"\s{2,}", " ", regex=True)
        .str.strip()
    )

    df["review_len"] = df["review"].str.len()
    return df


def percentile_report(df: pd.DataFrame) -> str:
    """生成百分位分析报告"""
    lengths = df["review_len"]
    lines = []
    lines.append("=" * 55)
    lines.append("Review 长度百分位分析")
    lines.append("=" * 55)
    lines.append(f"总评论数: {len(df):,}")
    lines.append(f"长度范围: {lengths.min()} ~ {lengths.max()}")
    lines.append(f"均值: {lengths.mean():.0f}  中位数: {lengths.median():.0f}")
    lines.append(f"标准差: {lengths.std():.0f}")
    lines.append("-" * 55)
    lines.append(f"{'百分位':>6}  {'长度(字符)':>10}  {'累计占比':>10}")
    lines.append("-" * 55)

    for p in [1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 100]:
        val = np.percentile(lengths, p)
        pct_below = (lengths <= val).mean() * 100
        lines.append(f"  P{p:>3}  {val:>10.0f}  {pct_below:>9.1f}%")

    lines.append("=" * 55)
    return "\n".join(lines)


def clean_by_percentile(df: pd.DataFrame,
                         lo_pct: float = 2,
                         hi_pct: float = 99):
    """基于百分位阈值清洗，返回 (clean, low, high) 三个 DataFrame"""
    lengths = df["review_len"]
    lo_val = np.percentile(lengths, lo_pct)
    hi_val = np.percentile(lengths, hi_pct)

    log.info("百分位阈值: P%.0f=%.0f 字符  ~  P%.0f=%.0f 字符", lo_pct, lo_val, hi_pct, hi_val)

    before = len(df)
    mask_clean = (lengths >= lo_val) & (lengths <= hi_val)
    mask_low   = lengths < lo_val
    mask_high  = lengths > hi_val

    df_clean = df[mask_clean].copy()
    df_low   = df[mask_low].copy()
    df_high  = df[mask_high].copy()

    log.info("保留: %d 条, 过短: %d 条 (<%.0f), 过长: %d 条 (>%.0f)",
             len(df_clean), len(df_low), lo_val, len(df_high), hi_val)

    return df_clean, df_low, df_high


def clean_by_length(df: pd.DataFrame,
                     min_len: int = 2,
                     max_len: int = 2261):
    """基于绝对字符数阈值清洗，返回 (clean, low, high) 三个 DataFrame"""
    before = len(df)
    mask_clean = (df["review_len"] >= min_len) & (df["review_len"] <= max_len)
    mask_low   = df["review_len"] < min_len
    mask_high  = df["review_len"] > max_len

    df_clean = df[mask_clean].copy()
    df_low   = df[mask_low].copy()
    df_high  = df[mask_high].copy()

    log.info("绝对阈值: %d ~ %d 字符", min_len, max_len)
    log.info("保留: %d 条, 过短: %d 条, 过长: %d 条",
             len(df_clean), len(df_low), len(df_high))

    return df_clean, df_low, df_high


def run_elt(lo_pct: float = 2, hi_pct: float = 99,
            min_len: int = 0, max_len: int = 0,
            input_file: str = INPUT_FILE,
            output_file: str = OUTPUT_FILE):
    """主流程"""

    # 1. 加载
    df = load_data(input_file)

    # 2. 分析报告
    report = percentile_report(df)
    print(report)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    log.info("报告已保存: %s", REPORT_FILE)

    # 3. 清洗
    if min_len > 0 or max_len > 0:
        _min = min_len if min_len > 0 else 0
        _max = max_len if max_len > 0 else df["review_len"].max()
        df_clean, df_low, df_high = clean_by_length(df, _min, _max)
    else:
        df_clean, df_low, df_high = clean_by_percentile(df, lo_pct, hi_pct)

    # 4. 输出（去掉辅助列）
    for fname, frame in [
        (output_file, df_clean),
        (LOW_FILE,    df_low),
        (HIGH_FILE,   df_high),
    ]:
        if "review_len" in frame.columns:
            frame = frame.drop(columns=["review_len"])
        log.info("写入 %d 条 -> %s", len(frame), fname)
        frame.to_csv(fname, index=False, encoding="utf-8")

    log.info("完成 — 清洗后 %d 条, 过短 %d 条, 过长 %d 条",
             len(df_clean), len(df_low), len(df_high))

    return df_clean


# ── CLI ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="数据清洗 ETL — 基于 review 长度百分位过滤"
    )
    parser.add_argument("--lo_pct", type=float, default=2,
                        help="低百分位阈值 (default: 2)")
    parser.add_argument("--hi_pct", type=float, default=99,
                        help="高百分位阈值 (default: 99)")
    parser.add_argument("--min_len", type=int, default=0,
                        help="绝对最小字符数（覆盖 --lo_pct）")
    parser.add_argument("--max_len", type=int, default=0,
                        help="绝对最大字符数（覆盖 --hi_pct）")
    parser.add_argument("--input", default=INPUT_FILE,
                        help="输入 CSV 路径")
    parser.add_argument("--output", default=OUTPUT_FILE,
                        help="输出 CSV 路径")

    args = parser.parse_args()
    run_elt(
        lo_pct=args.lo_pct,
        hi_pct=args.hi_pct,
        min_len=args.min_len,
        max_len=args.max_len,
        input_file=args.input,
        output_file=args.output,
    )
