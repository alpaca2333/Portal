"""
Download industry classification from tushare (Shenwan / 申万行业分类).
Saves to:
  - data/quant/data/industry/sw_industry_index.csv   (industry index list)
  - data/quant/data/industry/sw_industry_member.csv   (stock -> industry mapping)

tushare APIs:
  - pro.index_classify()    : get Shenwan industry classification list
  - pro.index_member()      : get stocks in each industry index

Fields of interest:
  - industry_code (index_code): Shenwan industry code
  - industry_name (index_name): Shenwan industry name
  - ts_code (con_code): stock code
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
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data", "industry")

# ─── args ────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="下载申万行业分类")
parser.add_argument("--since", type=str, default=None,
                    help="数据起始日期 (YYYYMMDD)，行业分类为全量快照，此参数仅为接口统一")
parser.add_argument("--workers", type=int, default=5,
                    help="并发线程数 (默认: 5)")
parser.add_argument("--tqdm-position", type=int, default=None,
                    help="tqdm进度条行位置 (由download_all自动传递)")
args = parser.parse_args()

NUM_WORKERS = args.workers
TQDM_POS = args.tqdm_position

# ─── init ────────────────────────────────────────────────────────────
with open(TOKEN_PATH, "r") as f:
    token = f.read().strip()
ts.set_token(token)
pro = ts.pro_api()

os.makedirs(DATA_DIR, exist_ok=True)

MAX_RETRIES = 10
RETRY_BASE_WAIT = 5  # seconds, will double on each retry

_local = threading.local()

def get_pro():
    if not hasattr(_local, "pro"):
        _local.pro = ts.pro_api()
    return _local.pro

# ─── Step 1: Download Shenwan industry index list (L1 + L2 + L3) ────
print("正在下载申万行业分类列表 ...")

all_indices = []
tqdm_kwargs_l = dict(desc="行业分类", unit="级", ncols=80)
if TQDM_POS is not None:
    tqdm_kwargs_l.update(position=TQDM_POS, leave=True)
for level in tqdm(["L1", "L2", "L3"], **tqdm_kwargs_l):
    for retry in range(MAX_RETRIES):
        try:
            df = pro.index_classify(level=level, src="SW2021")
            if df is not None and not df.empty:
                df["level"] = level
                all_indices.append(df)
                if TQDM_POS is None:
                    tqdm.write(f"  {level}: {len(df)} 个行业")
            break
        except Exception as e:
            wait = RETRY_BASE_WAIT * (2 ** retry)
            if TQDM_POS is None:
                tqdm.write(f"  下载 {level} 失败 (重试 {retry + 1}/{MAX_RETRIES}): {e}，等待 {wait}s ...")
            time.sleep(wait)
    else:
        if TQDM_POS is None:
            tqdm.write(f"  [放弃] {level} 在 {MAX_RETRIES} 次重试后仍失败")

if all_indices:
    df_index = pd.concat(all_indices, ignore_index=True)
    save_path = os.path.join(DATA_DIR, "sw_industry_index.csv")
    df_index.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  已保存行业分类列表 -> {save_path}")
else:
    print("  错误：未获取到行业分类列表")
    sys.exit(1)


# ─── worker function for index_member ────────────────────────────────
def download_member(index_code, industry_name):
    """Download members for one industry index. Returns (index_code, success, df)."""
    _pro = get_pro()
    for retry in range(MAX_RETRIES):
        try:
            df_mem = _pro.index_member(index_code=index_code)
            if df_mem is not None and not df_mem.empty:
                df_mem["industry_code"] = index_code
                df_mem["industry_name"] = industry_name
                return (index_code, True, df_mem)
            return (index_code, True, None)
        except Exception as e:
            wait = RETRY_BASE_WAIT * (2 ** retry)
            time.sleep(wait)
    return (index_code, False, None)


# ─── Step 2: Download stock-industry membership (L1) ────────────────
print(f"正在下载一级行业成分股 (并发: {NUM_WORKERS}) ...")

df_l1 = df_index[df_index["level"] == "L1"]
all_members = []

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {
        executor.submit(download_member, row["index_code"], row["industry_name"]): row["index_code"]
        for _, row in df_l1.iterrows()
    }
    tqdm_kwargs_m = dict(total=len(df_l1), desc="一级行业成分股", unit="个", ncols=80)
    if TQDM_POS is not None:
        tqdm_kwargs_m.update(position=TQDM_POS, leave=True)
    with tqdm(**tqdm_kwargs_m) as pbar:
        for future in as_completed(futures):
            index_code, success, df_mem = future.result()
            if not success:
                if TQDM_POS is None:
                    tqdm.write(f"  [放弃] {index_code} 在 {MAX_RETRIES} 次重试后仍失败")
            elif df_mem is not None:
                all_members.append(df_mem)
            pbar.update(1)

if all_members:
    df_members = pd.concat(all_members, ignore_index=True)
    save_path = os.path.join(DATA_DIR, "sw_industry_member.csv")
    df_members.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  已保存行业成分股映射，共 {len(df_members)} 条 -> {save_path}")
else:
    print("  错误：未获取到行业成分股数据")

# ─── Step 3: Also download L2 membership for finer granularity ───────
print(f"正在下载二级行业成分股 (并发: {NUM_WORKERS}) ...")

df_l2 = df_index[df_index["level"] == "L2"]
all_members_l2 = []

with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = {
        executor.submit(download_member, row["index_code"], row["industry_name"]): row["index_code"]
        for _, row in df_l2.iterrows()
    }
    tqdm_kwargs_m2 = dict(total=len(df_l2), desc="二级行业成分股", unit="个", ncols=80)
    if TQDM_POS is not None:
        tqdm_kwargs_m2.update(position=TQDM_POS, leave=True)
    with tqdm(**tqdm_kwargs_m2) as pbar:
        for future in as_completed(futures):
            index_code, success, df_mem = future.result()
            if not success:
                if TQDM_POS is None:
                    tqdm.write(f"  [放弃] {index_code} 在 {MAX_RETRIES} 次重试后仍失败")
            elif df_mem is not None:
                all_members_l2.append(df_mem)
            pbar.update(1)

if all_members_l2:
    df_members_l2 = pd.concat(all_members_l2, ignore_index=True)
    save_path = os.path.join(DATA_DIR, "sw_industry_member_l2.csv")
    df_members_l2.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"  已保存二级行业成分股映射，共 {len(df_members_l2)} 条 -> {save_path}")

print("行业分类数据下载完成！")
