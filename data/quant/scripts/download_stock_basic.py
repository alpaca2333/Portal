"""
Download stock basic info from tushare.
Saves to: data/quant/data/stock_basic/stock_basic.csv
"""
import os
import sys
import argparse
import tushare as ts
import pandas as pd

# ─── paths ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
TOKEN_PATH = os.path.join(PROJECT_ROOT, "public", "stock-data", "tushare.token")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "stock_basic")

# ─── args ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="下载股票列表")
parser.add_argument("--since", type=str, default=None,
                    help="数据起始日期 (YYYYMMDD)，股票列表总是全量下载，此参数仅为接口统一")
parser.add_argument("--workers", type=int, default=5,
                    help="并发线程数 (默认: 5)，股票列表为单次请求，此参数仅为接口统一")
parser.add_argument("--tqdm-position", type=int, default=None,
                    help="tqdm进度条行位置，此参数仅为接口统一")
args = parser.parse_args()

# ─── init ────────────────────────────────────────────────────────────
with open(TOKEN_PATH, "r") as f:
    token = f.read().strip()
ts.set_token(token)
pro = ts.pro_api()

os.makedirs(DATA_DIR, exist_ok=True)

# ─── download ────────────────────────────────────────────────────────
print("正在下载股票列表 (stock_basic) ...")

# Listed stocks
df_l = pro.stock_basic(exchange="", list_status="L",
                       fields="ts_code,symbol,name,area,industry,market,list_date,list_status")
# Delisted stocks
df_d = pro.stock_basic(exchange="", list_status="D",
                       fields="ts_code,symbol,name,area,industry,market,list_date,delist_date,list_status")
# Paused stocks
df_p = pro.stock_basic(exchange="", list_status="P",
                       fields="ts_code,symbol,name,area,industry,market,list_date,list_status")

df = pd.concat([df_l, df_d, df_p], ignore_index=True)

save_path = os.path.join(DATA_DIR, "stock_basic.csv")
df.to_csv(save_path, index=False, encoding="utf-8-sig")
print(f"已保存股票列表，共 {len(df)} 条记录 -> {save_path}")
