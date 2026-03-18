"""
直接用qlib的LocalProvider写入数据
将CSV转成qlib的二进制bin文件格式
"""
import os
import struct
import numpy as np
import pandas as pd
from pathlib import Path

QLIB_DIR = Path.home() / ".qlib" / "qlib_data" / "cn_data"
CSV_FILE = Path("/root/quant_workspace/data/processed/all_stocks_daily.csv")
FEATURES_DIR = QLIB_DIR / "features"

def write_bin(data: np.ndarray, filepath: Path):
    """写入qlib二进制格式: 第一个float是起始index，后面是float32数组"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        # 写入起始offset（float32）
        f.write(struct.pack("<f", 0.0))
        # 写入数据
        data.astype("<f").tofile(f)

def process_stock(stock_df, code, all_dates_index):
    """处理单只股票，转成按交易日对齐的数组"""
    stock_df = stock_df.set_index("date")
    
    fields = ["open", "high", "low", "close", "volume", "factor"]
    
    stock_dir = FEATURES_DIR / code.upper()
    stock_dir.mkdir(parents=True, exist_ok=True)
    
    # 找到该股票的日期范围
    start_date = stock_df.index.min()
    end_date = stock_df.index.max()
    
    # 获取该范围内的所有交易日
    start_idx = all_dates_index.get_loc(start_date) if start_date in all_dates_index else 0
    end_idx = all_dates_index.get_loc(end_date) if end_date in all_dates_index else len(all_dates_index) - 1
    date_range = all_dates_index[start_idx:end_idx+1]
    
    # 对齐数据（缺失日期填NaN）
    aligned = stock_df.reindex(date_range)
    
    for field in fields:
        if field not in aligned.columns:
            continue
        data = aligned[field].values.astype(np.float32)
        # 写入bin文件
        filepath = stock_dir / f"{field}.day.bin"
        with open(filepath, "wb") as f:
            f.write(struct.pack("<f", float(start_idx)))  # 起始calendar index
            data.tofile(f)

def main():
    print("加载处理后的数据...")
    df = pd.read_csv(CSV_FILE, dtype={
        "code": str, "date": str,
        "open": np.float32, "high": np.float32, 
        "low": np.float32, "close": np.float32,
        "volume": np.float32, "factor": np.float32
    })
    
    # 加载交易日历
    cal_file = QLIB_DIR / "calendars" / "day.txt"
    with open(cal_file) as f:
        all_dates = [line.strip() for line in f if line.strip()]
    all_dates_index = pd.Index(all_dates)
    print(f"交易日历: {len(all_dates)} 天")
    
    # 按股票分组处理
    stocks = df["code"].unique()
    print(f"开始写入 {len(stocks)} 只股票的bin文件...")
    
    grouped = df.groupby("code")
    
    for i, (code, stock_df) in enumerate(grouped):
        if i % 500 == 0:
            print(f"  {i}/{len(stocks)} - {code}")
        try:
            process_stock(stock_df, code, all_dates_index)
        except Exception as e:
            print(f"  错误 {code}: {e}")
    
    print(f"\n完成！数据已写入: {FEATURES_DIR}")
    print(f"股票数量: {len(stocks)}")

if __name__ == "__main__":
    main()
