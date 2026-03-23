"""
Download suspend/resume info (停复牌信息) from tushare.
Saves to: data/quant/data/suspend/suspend_d.csv   (all stocks in one file, by date)

tushare API: pro.suspend_d()
Fields: ts_code, trade_date, suspend_timing, suspend_type
"""
import os
import sys
import time
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import tushare as ts
import pandas as pd
from tqdm import tqdm

# ─── paths ───────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
TOKEN_PATH = os.path.join(PROJECT_ROOT, "public", "stock-data", "tushare.token")
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "suspend")

# ─── config ──────────────────────────────────────────────────────────
DEFAULT_START_DATE = "20100101"
DEFAULT_END_DATE = "20260322"
MAX_RETRIES = 10
RETRY_BASE_WAIT = 5  # seconds, will double on each retry

# ─── args ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="下载停复牌信息")
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

# ─── build year tasks ────────────────────────────────────────────────
start = datetime.strptime(START_DATE, "%Y%m%d")
end = datetime.strptime(END_DATE, "%Y%m%d")

year_tasks = []
for year in range(start.year, end.year + 1):
    current = datetime(year, 1, 1)
    if year == start.year:
        current = start
    year_end = min(datetime(year, 12, 31), end)
    year_tasks.append((year, current.strftime("%Y%m%d"), year_end.strftime("%Y%m%d")))


# ─── worker function ─────────────────────────────────────────────────
def download_year(task):
    year, sd, ed = task
    pro = get_pro()
    for retry in range(MAX_RETRIES):
        try:
            df = pro.suspend_d(suspend_type="S", start_date=sd, end_date=ed)
            if df is not None and not df.empty:
                return (year, True, df)
            return (year, True, None)
        except Exception as e:
            wait = RETRY_BASE_WAIT * (2 ** retry)
            time.sleep(wait)
    return (year, False, None)


# ─── download ────────────────────────────────────────────────────────
print(f"正在下载停复牌信息 (suspend_d)，起始日期: {START_DATE}，并发: {NUM_WORKERS} ...")

all_data = []
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(download_year, t): t for t in year_tasks}
    tqdm_kwargs = dict(total=len(year_tasks), desc="停牌信息", unit="年", ncols=80)
    if TQDM_POS is not None:
        tqdm_kwargs.update(position=TQDM_POS, leave=True)
    with tqdm(**tqdm_kwargs) as pbar:
        for future in as_completed(futures):
            year, success, df = future.result()
            if not success:
                if TQDM_POS is None:
                    tqdm.write(f"  [放弃] {year} 在 {MAX_RETRIES} 次重试后仍失败")
            elif df is not None:
                all_data.append(df)
                if TQDM_POS is None:
                    tqdm.write(f"  {year}: {len(df)} 条停牌记录")
            pbar.update(1)

save_path = os.path.join(DATA_DIR, "suspend_d.csv")

if all_data:
    df_new = pd.concat(all_data, ignore_index=True)
    if os.path.exists(save_path):
        df_old = pd.read_csv(save_path, dtype={"trade_date": str, "ts_code": str})
        df_new["trade_date"] = df_new["trade_date"].astype(str)
        df_new["ts_code"] = df_new["ts_code"].astype(str)
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
    else:
        df_combined = df_new
    df_combined = df_combined.sort_values(["trade_date", "ts_code"]).reset_index(drop=True)
    df_combined.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"停复牌信息下载完成，共 {len(df_combined)} 条 -> {save_path}")
else:
    print("未获取到新的停复牌信息")
