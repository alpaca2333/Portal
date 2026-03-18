"""
将 ~/qlib_data/ 下的CSV文件转换并导入qlib格式
CSV格式: symbol,date,open,high,low,close,volume,factor
代码格式: 600000.SH / 000001.SZ / 920000.BJ
"""
import os
import glob
import pandas as pd
from pathlib import Path

RAW_DIR = Path.home() / "qlib_data"
QLIB_DIR = Path.home() / ".qlib" / "qlib_data" / "cn_data"

def get_exchange(filename):
    """从文件名提取交易所前缀"""
    name = Path(filename).stem  # e.g. 600000.SH
    parts = name.split(".")
    if len(parts) == 2:
        code, exchange = parts
        return exchange.lower(), code
    return None, None

def convert_code(filename):
    """600000.SH -> sh600000"""
    exchange, code = get_exchange(filename)
    if exchange:
        return f"{exchange}{code}"
    return None

def load_all_data():
    """加载所有CSV，合并成单一DataFrame"""
    files = glob.glob(str(RAW_DIR / "*.csv"))
    print(f"找到 {len(files)} 个CSV文件")
    
    dfs = []
    for i, f in enumerate(files):
        if i % 500 == 0:
            print(f"  处理 {i}/{len(files)}...")
        try:
            df = pd.read_csv(f)
            # 统一列名
            df.columns = [c.lower().strip() for c in df.columns]
            
            # 生成qlib格式的code
            qlib_code = convert_code(f)
            if qlib_code is None:
                continue
            df["code"] = qlib_code
            
            dfs.append(df)
        except Exception as e:
            print(f"  跳过 {f}: {e}")
    
    print(f"合并 {len(dfs)} 个文件...")
    all_data = pd.concat(dfs, ignore_index=True)
    return all_data

def prepare_qlib_data(df):
    """整理成qlib所需格式"""
    # 确保date列是字符串格式 YYYY-MM-DD
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    
    # 选择需要的列，计算vwap（如果没有就用close代替）
    cols_needed = ["code", "date", "open", "high", "low", "close", "volume", "factor"]
    available = [c for c in cols_needed if c in df.columns]
    df = df[available].copy()
    
    # 去重，按code+date排序
    df = df.drop_duplicates(subset=["code", "date"])
    df = df.sort_values(["code", "date"])
    
    return df

def build_calendars(df):
    """生成交易日历"""
    cal_dir = QLIB_DIR / "calendars"
    cal_dir.mkdir(parents=True, exist_ok=True)
    
    dates = sorted(df["date"].unique())
    with open(cal_dir / "day.txt", "w") as f:
        for d in dates:
            f.write(d + "\n")
    print(f"交易日历: {len(dates)} 个交易日 ({dates[0]} ~ {dates[-1]})")

def build_instruments(df):
    """生成股票列表"""
    inst_dir = QLIB_DIR / "instruments"
    inst_dir.mkdir(parents=True, exist_ok=True)
    
    # 每只股票的上市/退市日期
    stock_dates = df.groupby("code")["date"].agg(["min", "max"]).reset_index()
    
    with open(inst_dir / "all.txt", "w") as f:
        for _, row in stock_dates.iterrows():
            f.write(f"{row['code'].upper()}\t{row['min']}\t{row['max']}\n")
    
    # 按交易所分组
    sh_stocks = stock_dates[stock_dates["code"].str.startswith("sh")]
    sz_stocks = stock_dates[stock_dates["code"].str.startswith("sz")]
    bj_stocks = stock_dates[stock_dates["code"].str.startswith("bj")]
    
    for name, subset in [("sh", sh_stocks), ("sz", sz_stocks), ("bj", bj_stocks), ("csi300_placeholder", sh_stocks.head(0))]:
        with open(inst_dir / f"{name}.txt", "w") as f:
            for _, row in subset.iterrows():
                f.write(f"{row['code'].upper()}\t{row['min']}\t{row['max']}\n")
    
    print(f"股票列表: 共 {len(stock_dates)} 只 (SH:{len(sh_stocks)}, SZ:{len(sz_stocks)}, BJ:{len(bj_stocks)})")

def save_as_csv(df):
    """
    将数据保存为qlib CSV格式（按股票分文件）
    路径: ~/.qlib/qlib_data/cn_data/features/<code>/
    这里先保存到 data/ 目录下供后续qlib dump使用
    """
    out_dir = Path("/root/quant_workspace/data/processed")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存合并后的大文件
    out_file = out_dir / "all_stocks_daily.csv"
    df.to_csv(out_file, index=False)
    print(f"已保存到: {out_file} ({len(df)} 行)")
    return out_file

def main():
    print("=" * 50)
    print("开始导入A股日线数据到qlib")
    print("=" * 50)
    
    # 1. 加载数据
    df = load_all_data()
    print(f"\n原始数据: {len(df)} 行")
    
    # 2. 整理格式
    df = prepare_qlib_data(df)
    print(f"清洗后: {len(df)} 行, {df['code'].nunique()} 只股票")
    
    # 3. 生成calendars
    build_calendars(df)
    
    # 4. 生成instruments
    build_instruments(df)
    
    # 5. 保存处理后的数据
    out_file = save_as_csv(df)
    
    print("\n" + "=" * 50)
    print("基础整理完成！")
    print(f"下一步: 用 qlib dump 命令将数据转成二进制格式")
    print("=" * 50)
    
    return df

if __name__ == "__main__":
    df = main()
    print(f"\n数据预览:")
    print(df.head(3))
    print(f"\n数据范围: {df['date'].min()} ~ {df['date'].max()}")
