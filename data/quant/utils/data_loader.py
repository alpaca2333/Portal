"""
简单的数据读取工具，直接读CSV，绕过qlib的pandas兼容性问题
后续因子计算、回测直接用这个接口
"""
import pandas as pd
import numpy as np
from pathlib import Path
from functools import lru_cache

DATA_FILE = Path("/root/quant_workspace/data/processed/all_stocks_daily.csv")

@lru_cache(maxsize=1)
def _load_all():
    """加载全量数据（缓存）"""
    print("加载数据中...")
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df["code"] = df["code"].str.upper()
    df = df.set_index(["date", "code"]).sort_index()
    print(f"数据加载完成: {len(df)} 行, {df.index.get_level_values('code').nunique()} 只股票")
    return df

def get_data(codes=None, start=None, end=None, fields=None):
    """
    获取股票数据
    
    参数:
        codes: 股票代码列表，如 ['SH600519', 'SZ000001']，None表示全部
        start: 开始日期，如 '2020-01-01'
        end: 结束日期，如 '2024-12-31'
        fields: 字段列表，如 ['close', 'volume']，None表示全部
    
    返回:
        MultiIndex DataFrame (date, code)
    """
    df = _load_all()
    
    if start:
        df = df[df.index.get_level_values("date") >= pd.Timestamp(start)]
    if end:
        df = df[df.index.get_level_values("date") <= pd.Timestamp(end)]
    if codes:
        codes = [c.upper() for c in codes]
        df = df[df.index.get_level_values("code").isin(codes)]
    if fields:
        df = df[fields]
    
    return df

def get_close(codes=None, start=None, end=None):
    """获取收盘价矩阵 (日期 x 股票)"""
    df = get_data(codes=codes, start=start, end=end, fields=["close"])
    return df["close"].unstack("code")

def get_returns(codes=None, start=None, end=None, periods=1):
    """获取收益率"""
    close = get_close(codes=codes, start=start, end=end)
    return close.pct_change(periods)

def get_universe(start=None, end=None, min_days=60):
    """获取满足条件的股票池"""
    df = get_data(start=start, end=end, fields=["close"])
    counts = df["close"].groupby(level="code").count()
    return counts[counts >= min_days].index.tolist()

if __name__ == "__main__":
    # 测试
    print("=== 测试数据读取 ===")
    
    # 1. 读取茅台数据
    df = get_data(codes=["SH600519"], start="2024-01-01", end="2024-03-01")
    print(f"\n茅台2024年Q1数据 ({len(df)} 行):")
    print(df.head())
    
    # 2. 收盘价矩阵
    close = get_close(codes=["SH600519", "SH600036", "SZ000001"], 
                      start="2024-01-01", end="2024-01-10")
    print(f"\n收盘价矩阵:")
    print(close)
    
    # 3. 收益率
    ret = get_returns(codes=["SH600519"], start="2024-01-01", end="2024-01-20")
    print(f"\n茅台日收益率:")
    print(ret.dropna().head())
