#!/usr/bin/env python3
"""
split_csv_chunks.py
把 all_stocks_daily.csv 拆成 chunk0/1/2，保留 header，每份 ~744 万行。
输出：
  data/quant/processed/all_stocks_daily_chunk0.csv
  data/quant/processed/all_stocks_daily_chunk1.csv
  data/quant/processed/all_stocks_daily_chunk2.csv
"""
import os

SRC  = "/projects/portal/data/quant/processed/all_stocks_daily.csv"
DEST = "/projects/portal/data/quant/processed"
N_CHUNKS = 3

print(f"读取源文件：{SRC}")
with open(SRC, "r") as f:
    header = f.readline()
    total_lines = sum(1 for _ in f)

chunk_size = (total_lines + N_CHUNKS - 1) // N_CHUNKS
print(f"总数据行：{total_lines}，每 chunk：{chunk_size} 行，共 {N_CHUNKS} 个文件")

# 打开所有输出文件
out_files = []
for i in range(N_CHUNKS):
    path = os.path.join(DEST, f"all_stocks_daily_chunk{i}.csv")
    fh = open(path, "w")
    fh.write(header)
    out_files.append((path, fh))

print("开始写入...")
with open(SRC, "r") as f:
    f.readline()  # skip header
    for idx, line in enumerate(f):
        chunk_idx = idx // chunk_size
        if chunk_idx >= N_CHUNKS:
            chunk_idx = N_CHUNKS - 1
        out_files[chunk_idx][1].write(line)
        if (idx + 1) % 2_000_000 == 0:
            print(f"  已处理 {idx+1:,} 行...")

for path, fh in out_files:
    fh.close()
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  {os.path.basename(path)}: {size_mb:.0f} MB")

print("完成。")
