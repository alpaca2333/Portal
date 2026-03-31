"""
Download index weights data from tushare.
Saves to: data/quant/data/index_weight/{index_code}.csv

tushare API: pro.index_weight()
Fields saved: index_code, con_code, trade_date, weight
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
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "index_weight")

# ─── config ──────────────────────────────────────────────────────────
DEFAULT_START_DATE = "20100101"
DEFAULT_END_DATE = "20260322"
MAX_RETRIES = 10
RETRY_BASE_WAIT = 5  # seconds, will double on each retry

TARGET_INDICES = {
    "000300.SH": "沪深300",
    "000905.SH": "中证500"
}

# ─── args ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="下载指数成分股权重数据")
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

# ─── worker function ─────────────────────────────────────────────────
def download_index(index_code):
    """Download index weight data for a single index by chunking months."""
    save_path = os.path.join(DATA_DIR, f"{index_code}.csv")
    pro = get_pro()
    
    # Generate month chunks
    start_dt = pd.to_datetime(START_DATE)
    end_dt = pd.to_datetime(END_DATE)
    
    # Create month start and end dates
    months = pd.date_range(start=start_dt.replace(day=1), end=end_dt, freq='MS')
    
    all_data = []
    
    # Add a nested progress bar for months
    month_pbar = tqdm(months, desc=f"  {index_code}", leave=False, position=TQDM_POS + 1 if TQDM_POS is not None else 1)
    
    for month_start in month_pbar:
        month_end = month_start + pd.offsets.MonthEnd(1)
        if month_end > end_dt:
            month_end = end_dt
            
        m_start_str = month_start.strftime("%Y%m%d")
        m_end_str = month_end.strftime("%Y%m%d")
        
        for retry in range(MAX_RETRIES):
            try:
                df_new = pro.index_weight(index_code=index_code, start_date=m_start_str, end_date=m_end_str)
                if df_new is not None and not df_new.empty:
                    all_data.append(df_new)
                time.sleep(0.35)  # Avoid hitting 200 requests/min limit
                break
            except Exception as e:
                if retry == MAX_RETRIES - 1:
                    month_pbar.close()
                    return (index_code, False, f"在 {MAX_RETRIES} 次重试后仍失败: {e}")
                
                # If it's a rate limit error, sleep longer
                error_msg = str(e)
                if "每分钟最多访问" in error_msg:
                    wait = 60  # Wait a full minute for rate limit to reset
                    month_pbar.set_postfix_str(f"触发限流，等待 {wait}s...")
                else:
                    wait = RETRY_BASE_WAIT * (2 ** retry)
                    month_pbar.set_postfix_str(f"重试 {retry+1}/{MAX_RETRIES}，等待 {wait}s...")
                
                time.sleep(wait)
                month_pbar.set_postfix_str("")
                
    month_pbar.close()
                
    if all_data:
        df_total = pd.concat(all_data, ignore_index=True)
        if os.path.exists(save_path):
            df_old = pd.read_csv(save_path, dtype={"index_code": str, "con_code": str, "trade_date": str})
            df_total["trade_date"] = df_total["trade_date"].astype(str)
            df_combined = pd.concat([df_old, df_total], ignore_index=True)
            df_combined = df_combined.drop_duplicates(subset=["index_code", "con_code", "trade_date"], keep="last")
        else:
            df_combined = df_total
            
        df_combined = df_combined.sort_values(["trade_date", "con_code"]).reset_index(drop=True)
        df_combined.to_csv(save_path, index=False, encoding="utf-8-sig")
        
    return (index_code, True, None)

# ─── download ────────────────────────────────────────────────────────
print(f"正在下载指数权重数据 (index_weight)，共 {len(TARGET_INDICES)} 个指数，起始日期: {START_DATE} ...")

failed = []
# Reduce workers to 1 to avoid hitting the 200 req/min limit across threads
with ThreadPoolExecutor(max_workers=1) as executor:
    futures = {executor.submit(download_index, code): code for code in TARGET_INDICES.keys()}
    tqdm_kwargs = dict(total=len(TARGET_INDICES), desc="指数权重", unit="个", ncols=80)
    if TQDM_POS is not None:
        tqdm_kwargs.update(position=TQDM_POS, leave=True)
    with tqdm(**tqdm_kwargs) as pbar:
        for future in as_completed(futures):
            index_code, success, msg = future.result()
            if not success:
                if TQDM_POS is None:
                    tqdm.write(f"  [放弃] {index_code}: {msg}")
                failed.append(index_code)
            pbar.update(1)

if failed:
    print(f"共 {len(failed)} 个指数下载失败: {failed}")
print(f"指数权重数据下载完成 -> {DATA_DIR}")
