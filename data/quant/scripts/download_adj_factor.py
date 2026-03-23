"""
Download adjust factor (复权因子) from tushare.
Saves to: data/quant/data/adj_factor/{ts_code}.csv   (one file per stock)

tushare API: pro.adj_factor()
Fields: ts_code, trade_date, adj_factor
"""
import os
import sys
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import tushare as ts
import pandas as pd
from tqdm import tqdm

# ─── paths ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
TOKEN_PATH = os.path.join(PROJECT_ROOT, "public", "stock-data", "tushare.token")
BASIC_PATH = os.path.join(SCRIPT_DIR, "..", "data", "stock_basic", "stock_basic.csv")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "adj_factor")

# ─── config ──────────────────────────────────────────────────────────
DEFAULT_START_DATE = "20100101"
DEFAULT_END_DATE = "20260322"
MAX_RETRIES = 10
RETRY_BASE_WAIT = 5  # seconds, will double on each retry

# ─── args ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="下载复权因子")
parser.add_argument("--since", type=str, default=None,
                    help="数据起始日期 (YYYYMMDD)，用于增量追加")
parser.add_argument("--workers", type=int, default=5,
                    help="并发线程数 (默认: 5)")
parser.add_argument("--tqdm-position", type=int, default=None,
                    help="tqdm进度条行位置 (由download_all自动传递)")
args = parser.parse_args()

START_DATE = args.since if args.since else DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE
NUM_WORKERS = args.workers
TQDM_POS = args.tqdm_position

# ─── init ────────────────────────────────────────────────────────────
with open(TOKEN_PATH, "r") as f:
    token = f.read().strip()
ts.set_token(token)

os.makedirs(DATA_DIR, exist_ok=True)

_local = threading.local()

def get_pro():
    if not hasattr(_local, "pro"):
        _local.pro = ts.pro_api()
    return _local.pro

# ─── load stock list ─────────────────────────────────────────────────
if not os.path.exists(BASIC_PATH):
    print("错误：请先运行 download_stock_basic.py 获取股票列表")
    sys.exit(1)

stock_list = pd.read_csv(BASIC_PATH)
ts_codes = stock_list["ts_code"].tolist()
total = len(ts_codes)


# ─── worker function ─────────────────────────────────────────────────
def download_one(ts_code):
    save_path = os.path.join(DATA_DIR, f"{ts_code}.csv")
    pro = get_pro()

    for retry in range(MAX_RETRIES):
        try:
            df_new = pro.adj_factor(ts_code=ts_code, start_date=START_DATE, end_date=END_DATE)
            if df_new is not None and not df_new.empty:
                if os.path.exists(save_path):
                    df_old = pd.read_csv(save_path, dtype={"trade_date": str})
                    df_new["trade_date"] = df_new["trade_date"].astype(str)
                    df_combined = pd.concat([df_old, df_new], ignore_index=True)
                    df_combined = df_combined.drop_duplicates(subset=["trade_date"], keep="last")
                else:
                    df_combined = df_new
                df_combined = df_combined.sort_values("trade_date").reset_index(drop=True)
                df_combined.to_csv(save_path, index=False, encoding="utf-8-sig")
            return (ts_code, True, None)
        except Exception as e:
            wait = RETRY_BASE_WAIT * (2 ** retry)
            time.sleep(wait)
    return (ts_code, False, f"在 {MAX_RETRIES} 次重试后仍失败")


# ─── download ────────────────────────────────────────────────────────
print(f"正在下载复权因子 (adj_factor)，共 {total} 只股票，起始日期: {START_DATE}，并发: {NUM_WORKERS} ...")

failed = []
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(download_one, code): code for code in ts_codes}
    tqdm_kwargs = dict(total=total, desc="复权因子", unit="只", ncols=80)
    if TQDM_POS is not None:
        tqdm_kwargs.update(position=TQDM_POS, leave=True)
    with tqdm(**tqdm_kwargs) as pbar:
        for future in as_completed(futures):
            ts_code, success, msg = future.result()
            if not success:
                if TQDM_POS is None:
                    tqdm.write(f"  [放弃] {ts_code}: {msg}")
                failed.append(ts_code)
            pbar.update(1)

if failed:
    print(f"共 {len(failed)} 只股票下载失败: {failed[:20]}{'...' if len(failed) > 20 else ''}")
print(f"复权因子下载完成 -> {DATA_DIR}")
