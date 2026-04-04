from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_BACKTEST_DIR = Path('/data/Projects/Portal/data/quant/backtest/lgbm_ensemble_adaptive_v7c')
DEFAULT_DB_PATH = Path('/data/Projects/Portal/data/quant/data/quant.db')
DEFAULT_OUTPUT_DIR = Path('/data/Projects/Portal/data/quant/reports/v7c_market_cap_distribution')

MV_BUCKETS_YI = [0, 50, 100, 200, 500, 1000, float('inf')]
MV_BUCKET_LABELS = ['<50亿', '50-100亿', '100-200亿', '200-500亿', '500-1000亿', '>=1000亿']


def find_latest_trade_csv(backtest_dir: Path) -> Path:
    files = sorted(backtest_dir.glob('*-trade.csv'))
    if not files:
        raise FileNotFoundError(f'未在 {backtest_dir} 找到任何 *-trade.csv 文件')
    return files[-1]


def weighted_average(series: pd.Series, weights: pd.Series) -> float:
    valid = series.notna() & weights.notna()
    if valid.sum() == 0:
        return float('nan')
    s = series[valid]
    w = weights[valid]
    total_weight = w.sum()
    if total_weight == 0:
        return float('nan')
    return float((s * w).sum() / total_weight)


def summarize_market_cap(df: pd.DataFrame, mv_col_yi: str, bucket_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    quantiles = df[mv_col_yi].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).rename(index={
        0.05: 'p05',
        0.25: 'p25',
        0.50: 'p50',
        0.75: 'p75',
        0.95: 'p95',
    })
    summary = pd.DataFrame([
        {
            'trade_count': len(df),
            'unique_stocks': df['ts_code'].nunique(),
            'mean_yi': df[mv_col_yi].mean(),
            'median_yi': df[mv_col_yi].median(),
            'min_yi': df[mv_col_yi].min(),
            'max_yi': df[mv_col_yi].max(),
            'amount_weighted_avg_yi': weighted_average(df[mv_col_yi], df['amount']),
            **quantiles.to_dict(),
        }
    ])

    bucket = (
        df.groupby(bucket_col, observed=False)
        .agg(
            trade_count=('ts_code', 'size'),
            unique_stocks=('ts_code', 'nunique'),
            total_amount=('amount', 'sum'),
            mean_yi=(mv_col_yi, 'mean'),
            median_yi=(mv_col_yi, 'median'),
        )
        .reset_index()
        .rename(columns={bucket_col: 'bucket'})
    )
    bucket['trade_pct'] = bucket['trade_count'] / bucket['trade_count'].sum()
    bucket['amount_pct'] = bucket['total_amount'] / bucket['total_amount'].sum()
    return summary, bucket


def format_pct(x: float) -> str:
    if pd.isna(x):
        return 'NaN'
    return f'{x:.2%}'


def format_num(x: float) -> str:
    if pd.isna(x):
        return 'NaN'
    return f'{x:,.2f}'


def df_to_markdown(df: pd.DataFrame, float_digits: int = 2) -> str:
    display_df = df.copy()
    for col in display_df.columns:
        if pd.api.types.is_float_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: 'NaN' if pd.isna(x) else f'{x:.{float_digits}f}')
        elif pd.api.types.is_integer_dtype(display_df[col]):
            display_df[col] = display_df[col].map(lambda x: f'{x}')
        else:
            display_df[col] = display_df[col].astype(str)

    headers = [str(c) for c in display_df.columns]
    rows = display_df.values.tolist()
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt_row(row_vals: list[str]) -> str:
        return '| ' + ' | '.join(str(v).ljust(widths[i]) for i, v in enumerate(row_vals)) + ' |'

    sep = '| ' + ' | '.join('-' * w for w in widths) + ' |'
    parts = [fmt_row(headers), sep]
    parts.extend(fmt_row(row) for row in rows)
    return '\n'.join(parts)


def build_report(
    trades: pd.DataFrame,
    total_summary: pd.DataFrame,
    total_bucket: pd.DataFrame,
    circ_summary: pd.DataFrame,
    circ_bucket: pd.DataFrame,
    direction_summary: pd.DataFrame,
    yearly_summary: pd.DataFrame,
    trade_csv: Path,
) -> str:
    latest_date = trades['date'].max()
    earliest_date = trades['date'].min()
    top_total_bucket = total_bucket.sort_values('trade_count', ascending=False).head(1).iloc[0]
    top_circ_bucket = circ_bucket.sort_values('trade_count', ascending=False).head(1).iloc[0]

    lines: list[str] = []
    lines.append('# V7c 交易股票市值分布分析')
    lines.append('')
    lines.append('## 1. 数据范围')
    lines.append('')
    lines.append(f'- 成交文件：`{trade_csv}`')
    lines.append(f'- 交易区间：`{earliest_date}` ~ `{latest_date}`')
    lines.append(f'- 成交笔数：**{len(trades):,}**')
    lines.append(f'- 涉及股票数：**{trades["ts_code"].nunique():,}**')
    lines.append(f'- 仅统计状态为 `FILLED` 的成交，市值取自成交当日 `stock_daily.total_mv/circ_mv`，单位已换算为 **亿元**')
    lines.append('')
    lines.append('## 2. 关键结论')
    lines.append('')
    lines.append(
        f'- 总市值中位数约 **{format_num(total_summary.iloc[0]["median_yi"])} 亿**，金额加权平均约 **{format_num(total_summary.iloc[0]["amount_weighted_avg_yi"])} 亿**。'
    )
    lines.append(
        f'- 流通市值中位数约 **{format_num(circ_summary.iloc[0]["median_yi"])} 亿**，金额加权平均约 **{format_num(circ_summary.iloc[0]["amount_weighted_avg_yi"])} 亿**。'
    )
    lines.append(
        f'- 交易最集中的总市值区间是 **{top_total_bucket["bucket"]}**，占成交笔数 **{format_pct(top_total_bucket["trade_pct"])}**。'
    )
    lines.append(
        f'- 交易最集中的流通市值区间是 **{top_circ_bucket["bucket"]}**，占成交笔数 **{format_pct(top_circ_bucket["trade_pct"])}**。'
    )
    lines.append('')
    lines.append('## 3. 总市值分布')
    lines.append('')
    lines.append(df_to_markdown(total_summary, float_digits=2))
    lines.append('')
    lines.append(df_to_markdown(total_bucket, float_digits=4))
    lines.append('')
    lines.append('## 4. 流通市值分布')
    lines.append('')
    lines.append(df_to_markdown(circ_summary, float_digits=2))
    lines.append('')
    lines.append(df_to_markdown(circ_bucket, float_digits=4))
    lines.append('')
    lines.append('## 5. 按交易方向统计')
    lines.append('')
    lines.append(df_to_markdown(direction_summary, float_digits=2))
    lines.append('')
    lines.append('## 6. 按年份统计')
    lines.append('')
    lines.append(df_to_markdown(yearly_summary, float_digits=2))
    lines.append('')
    return '\n'.join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description='分析 V7c 交易股票的市值分布')
    parser.add_argument('--backtest-dir', type=Path, default=DEFAULT_BACKTEST_DIR, help='V7c 回测输出目录')
    parser.add_argument('--trade-csv', type=Path, default=None, help='指定成交文件；默认自动选取最新 *-trade.csv')
    parser.add_argument('--db-path', type=Path, default=DEFAULT_DB_PATH, help='SQLite 数据库路径')
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUTPUT_DIR, help='输出目录')
    args = parser.parse_args()

    trade_csv = args.trade_csv or find_latest_trade_csv(args.backtest_dir)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = pd.read_csv(trade_csv, dtype={'date': str, 'ts_code': str})
    trades = trades[trades['status'] == 'FILLED'].copy()
    if trades.empty:
        raise ValueError('成交文件中没有 FILLED 记录，无法分析')

    keys = trades[['ts_code', 'date']].drop_duplicates().copy()

    conn = sqlite3.connect(args.db_path)
    try:
        keys.to_sql('tmp_trade_keys', conn, index=False, if_exists='replace')
        market_caps = pd.read_sql_query(
            '''
            SELECT
                k.ts_code,
                k.date,
                d.total_mv,
                d.circ_mv,
                d.sw_l1,
                d.sw_l2,
                s.name,
                s.market,
                s.industry
            FROM tmp_trade_keys k
            LEFT JOIN stock_daily d
              ON d.ts_code = k.ts_code AND d.trade_date = k.date
            LEFT JOIN stock_info s
              ON s.ts_code = k.ts_code
            ''',
            conn,
        )
    finally:
        conn.close()

    trades = trades.merge(market_caps, on=['ts_code', 'date'], how='left')
    trades['total_mv_yi'] = trades['total_mv'] / 10000.0
    trades['circ_mv_yi'] = trades['circ_mv'] / 10000.0

    missing_mv = trades['total_mv_yi'].isna().sum()
    if missing_mv > 0:
        print(f'警告：有 {missing_mv} 笔成交未匹配到 total_mv，将在市值统计时忽略这些记录')

    trades_mv = trades.dropna(subset=['total_mv_yi', 'circ_mv_yi']).copy()
    trades_mv['year'] = trades_mv['date'].str.slice(0, 4)
    trades_mv['total_mv_bucket'] = pd.cut(
        trades_mv['total_mv_yi'],
        bins=MV_BUCKETS_YI,
        labels=MV_BUCKET_LABELS,
        right=False,
        include_lowest=True,
    )
    trades_mv['circ_mv_bucket'] = pd.cut(
        trades_mv['circ_mv_yi'],
        bins=MV_BUCKETS_YI,
        labels=MV_BUCKET_LABELS,
        right=False,
        include_lowest=True,
    )

    total_summary, total_bucket = summarize_market_cap(trades_mv, 'total_mv_yi', 'total_mv_bucket')
    circ_summary, circ_bucket = summarize_market_cap(trades_mv, 'circ_mv_yi', 'circ_mv_bucket')

    direction_summary = (
        trades_mv.groupby('direction')
        .agg(
            trade_count=('ts_code', 'size'),
            unique_stocks=('ts_code', 'nunique'),
            total_amount=('amount', 'sum'),
            total_mv_mean_yi=('total_mv_yi', 'mean'),
            total_mv_median_yi=('total_mv_yi', 'median'),
            total_mv_amt_weighted_yi=('total_mv_yi', lambda s: weighted_average(s, trades_mv.loc[s.index, 'amount'])),
            circ_mv_mean_yi=('circ_mv_yi', 'mean'),
            circ_mv_median_yi=('circ_mv_yi', 'median'),
            circ_mv_amt_weighted_yi=('circ_mv_yi', lambda s: weighted_average(s, trades_mv.loc[s.index, 'amount'])),
        )
        .reset_index()
    )

    yearly_summary = (
        trades_mv.groupby('year')
        .agg(
            trade_count=('ts_code', 'size'),
            unique_stocks=('ts_code', 'nunique'),
            total_amount=('amount', 'sum'),
            total_mv_mean_yi=('total_mv_yi', 'mean'),
            total_mv_median_yi=('total_mv_yi', 'median'),
            circ_mv_mean_yi=('circ_mv_yi', 'mean'),
            circ_mv_median_yi=('circ_mv_yi', 'median'),
        )
        .reset_index()
        .sort_values('year')
    )

    report = build_report(
        trades=trades_mv,
        total_summary=total_summary,
        total_bucket=total_bucket,
        circ_summary=circ_summary,
        circ_bucket=circ_bucket,
        direction_summary=direction_summary,
        yearly_summary=yearly_summary,
        trade_csv=trade_csv,
    )

    trades_mv.to_csv(out_dir / 'v7c_trade_with_market_cap.csv', index=False)
    total_summary.to_csv(out_dir / 'v7c_total_mv_summary.csv', index=False)
    total_bucket.to_csv(out_dir / 'v7c_total_mv_bucket_summary.csv', index=False)
    circ_summary.to_csv(out_dir / 'v7c_circ_mv_summary.csv', index=False)
    circ_bucket.to_csv(out_dir / 'v7c_circ_mv_bucket_summary.csv', index=False)
    direction_summary.to_csv(out_dir / 'v7c_direction_summary.csv', index=False)
    yearly_summary.to_csv(out_dir / 'v7c_yearly_summary.csv', index=False)
    (out_dir / 'v7c_market_cap_distribution_report.md').write_text(report, encoding='utf-8')

    print(f'分析完成：{trade_csv}')
    print(f'输出目录：{out_dir}')
    print('\n[总市值分布摘要]')
    print(total_summary.to_string(index=False))
    print('\n[总市值分桶]')
    print(total_bucket.to_string(index=False))
    print('\n[按方向统计]')
    print(direction_summary.to_string(index=False))


if __name__ == '__main__':
    main()
