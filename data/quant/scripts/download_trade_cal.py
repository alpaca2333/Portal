"""
Download trading calendar from tushare.
Saves to: data/quant/data/calendar/trade_cal.csv

tushare API: pro.trade_cal()
"""
import os
import argparse
import tushare as ts
import pandas as pd

# ─── paths ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
TOKEN_PATH = os.path.join(PROJECT_ROOT, "public", "stock-data", "tushare.token")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "calendar")

# ─── args ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="下载交易日历")
parser.add_argument("--since", type=str, default=None,
                    help="数据起始日期 (YYYYMMDD)，交易日历总是全量下载，此参数仅为接口统一")
parser.add_argument("--workers", type=int, default=5,
                    help="并发线程数 (默认: 5)，交易日历为单次请求，此参数仅为接口统一")
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
print("正在下载交易日历 (trade_cal) ...")

df = pro.trade_cal(exchange="SSE", start_date="20100101", end_date="20261231")
df = df.sort_values("cal_date").reset_index(drop=True)

save_path = os.path.join(DATA_DIR, "trade_cal.csv")
df.to_csv(save_path, index=False, encoding="utf-8-sig")
print(f"已保存交易日历，共 {len(df)} 条 -> {save_path}")
