"""
Download financial indicator data from tushare.
Saves to: data/quant/data/fina_indicator/{ts_code}.csv   (one file per stock)

tushare API: pro.fina_indicator()
Key fields: ts_code, ann_date, end_date, roe, roe_dt, roe_waa, roa,
            grossprofit_margin, netprofit_margin, eps, bps, cfps,
            debt_to_assets, current_ratio, quick_ratio,
            inv_turn, ar_turn, assets_turn, revenue_ps, etc.

Note: fina_indicator returns quarterly report data (end_date = report period).
      Each stock typically has ~60-80 records from 2010 to now.
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
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "fina_indicator")

# ─── config ──────────────────────────────────────────────────────────
DEFAULT_START_DATE = "20100101"
DEFAULT_END_DATE = "20260322"
MAX_RETRIES = 10
RETRY_BASE_WAIT = 5  # seconds, will double on each retry

FIELDS = ("ts_code,ann_date,end_date,"
          "eps,bps,cfps,revenue_ps,"
          "roe,roe_dt,roe_waa,roe_yearly,"
          "roa,roa_yearly,"
          "grossprofit_margin,netprofit_margin,profit_to_gr,"
          "debt_to_assets,current_ratio,quick_ratio,"
          "inv_turn,ar_turn,ca_turn,fa_turn,assets_turn,"
          "op_yoy,ebt_yoy,tr_yoy,or_yoy,equity_yoy,"
          "update_flag")

# ─── args ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="下载财务指标数据 (fina_indicator)")
parser.add_argument("--since", type=str, default=None,
                    help="数据起始日期 (YYYYMMDD)，用于增量追加（按公告日期 ann_date 过滤）")
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
            df_new = pro.fina_indicator(ts_code=ts_code,
                                        start_date=START_DATE,
                                        end_date=END_DATE,
                                        fields=FIELDS)
            if df_new is not None and not df_new.empty:
                df_new["end_date"] = df_new["end_date"].astype(str)
                if "ann_date" in df_new.columns:
                    df_new["ann_date"] = df_new["ann_date"].astype(str)

                if os.path.exists(save_path):
                    df_old = pd.read_csv(save_path, dtype={"end_date": str, "ann_date": str})
                    df_combined = pd.concat([df_old, df_new], ignore_index=True)
                    # Dedup by end_date + update_flag; keep the latest record
                    dedup_cols = ["end_date"]
                    if "update_flag" in df_combined.columns:
                        dedup_cols.append("update_flag")
                    df_combined = df_combined.drop_duplicates(subset=dedup_cols, keep="last")
                else:
                    df_combined = df_new

                df_combined = df_combined.sort_values("end_date").reset_index(drop=True)
                df_combined.to_csv(save_path, index=False, encoding="utf-8-sig")
            return (ts_code, True, None)
        except Exception as e:
            wait = RETRY_BASE_WAIT * (2 ** retry)
            time.sleep(wait)
    return (ts_code, False, f"在 {MAX_RETRIES} 次重试后仍失败")


# ─── download ────────────────────────────────────────────────────────
print(f"正在下载财务指标 (fina_indicator)，共 {total} 只股票，起始日期: {START_DATE}，并发: {NUM_WORKERS} ...")

failed = []
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {executor.submit(download_one, code): code for code in ts_codes}
    tqdm_kwargs = dict(total=total, desc="财务指标", unit="只", ncols=80)
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
print(f"财务指标下载完成 -> {DATA_DIR}")
