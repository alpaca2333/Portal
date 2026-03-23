"""
Master script to download all stock data from tushare.
Run this script to execute all download steps in the correct order.
Steps that have no dependency on each other run concurrently.

Execution plan:
    Wave 1 (parallel): stock_basic + trade_cal
    Wave 2 (parallel): daily + daily_basic + adj_factor + industry + suspend
                        (all depend on stock_basic from Wave 1)

Usage:
    python download_all.py                              # download all data
    python download_all.py --wave 2                     # start from wave 2 (skip stock_basic/trade_cal)
    python download_all.py --since 20250101              # incremental update since 2025-01-01
    python download_all.py --workers 10                  # 10 concurrent threads per sub-script
    python download_all.py --since 20250101 --workers 8  # combine options
"""
import os
import sys
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Wave 1: no inter-dependencies, run in parallel
WAVE_1 = [
    ("股票列表", "download_stock_basic.py"),
    ("交易日历", "download_trade_cal.py"),
]

# Wave 2: all depend on stock_basic (Wave 1), but independent of each other
WAVE_2 = [
    ("日线行情", "download_daily.py"),
    ("每日指标 (pe_ttm/pb/流通市值等)", "download_daily_basic.py"),
    ("复权因子", "download_adj_factor.py"),
    ("财务指标 (roe/roa/eps等)", "download_fina_indicator.py"),
    ("行业分类 (申万)", "download_industry.py"),
    ("停复牌信息", "download_suspend.py"),
]


def run_script(desc, script, extra_args, position=None):
    """Run a single download script as subprocess.

    When *position* is given the sub-script's stdout is silenced (only the
    tqdm progress bar, which writes directly to stderr, will be visible).
    This prevents normal print() output from messing up multi-line bars.
    """
    script_path = os.path.join(SCRIPT_DIR, script)
    cmd = [sys.executable, "-u", script_path] + extra_args
    if position is not None:
        cmd.extend(["--tqdm-position", str(position)])
        # Silence stdout so that only tqdm bars (on stderr) are shown
        result = subprocess.run(cmd, cwd=SCRIPT_DIR,
                                stdout=subprocess.DEVNULL,
                                stderr=None)   # stderr passes through for tqdm
    else:
        result = subprocess.run(cmd, cwd=SCRIPT_DIR)
    return (desc, script, result.returncode)


def run_wave(wave_name, tasks, extra_args):
    """Run all tasks in a wave concurrently with multi-line tqdm bars."""
    if not tasks:
        return True

    descs = ", ".join(d for d, _ in tasks)
    print(f"\n{'━' * 60}")
    print(f"  {wave_name} — 并发执行: {descs}")
    print(f"{'━' * 60}")

    # Reserve terminal lines for progress bars
    n = len(tasks)
    sys.stderr.write("\n" * n)
    sys.stderr.flush()

    failed = []
    with ThreadPoolExecutor(max_workers=n) as executor:
        futures = {
            executor.submit(run_script, desc, script, extra_args, position=idx): desc
            for idx, (desc, script) in enumerate(tasks)
        }
        for future in as_completed(futures):
            desc, script, returncode = future.result()
            if returncode != 0:
                failed.append(desc)

    # Move cursor below the progress bar area
    sys.stderr.write("\n" * n)
    sys.stderr.flush()

    # Print summary
    for desc, _ in tasks:
        if desc in failed:
            print(f"  ✗ {desc} 失败")
        else:
            print(f"  ✓ {desc} 完成")

    if failed:
        print(f"\n[错误] 以下步骤执行失败: {', '.join(failed)}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="下载全部tushare股票数据")
    parser.add_argument("--wave", type=int, default=1, choices=[1, 2],
                        help="从第几波开始执行 (默认: 1)")
    parser.add_argument("--since", type=str, default=None,
                        help="数据起始日期 (YYYYMMDD)，用于增量追加数据")
    parser.add_argument("--workers", type=int, default=5,
                        help="并发线程数 (默认: 5)，透传给各子脚本")
    args = parser.parse_args()

    # Build extra args to pass to sub-scripts
    extra_args = ["--workers", str(args.workers)]
    if args.since:
        extra_args.extend(["--since", args.since])

    print("=" * 60)
    print("  Tushare 股票数据下载工具")
    print("=" * 60)
    print()
    print("数据将保存到: data/quant/data/")
    if args.since:
        print(f"增量模式: 从 {args.since} 开始下载")
    print(f"子脚本并发线程数: {args.workers}")
    print(f"执行计划:")
    print(f"  Wave 1 (并发): 股票列表 + 交易日历")
    print(f"  Wave 2 (并发): 日线 + 基本面 + 复权因子 + 财务指标 + 行业 + 停牌")

    # Wave 1 — stock_basic & trade_cal have no tqdm bars, run them simply
    if args.wave <= 1:
        descs = ", ".join(d for d, _ in WAVE_1)
        print(f"\n{'━' * 60}")
        print(f"  Wave 1 — 并发执行: {descs}")
        print(f"{'━' * 60}\n")

        failed = []
        with ThreadPoolExecutor(max_workers=len(WAVE_1)) as executor:
            futures = {
                executor.submit(run_script, desc, script, extra_args): desc
                for desc, script in WAVE_1
            }
            for future in as_completed(futures):
                desc, script, returncode = future.result()
                if returncode == 0:
                    print(f"  ✓ {desc} 完成")
                else:
                    print(f"  ✗ {desc} 失败 (返回码: {returncode})")
                    failed.append(desc)

        if failed:
            print(f"\n[错误] 以下步骤执行失败: {', '.join(failed)}")
            print("Wave 1 有步骤失败，可以使用 --wave 2 跳过已完成的步骤重试")
            sys.exit(1)
    else:
        print(f"\n[跳过] Wave 1 (股票列表 + 交易日历)")

    # Wave 2 — each sub-script has tqdm bars; use --tqdm-position for multi-line display
    if args.wave <= 2:
        ok = run_wave("Wave 2", WAVE_2, extra_args)
        if not ok:
            print("Wave 2 有步骤失败")
            sys.exit(1)

    print(f"\n{'=' * 60}")
    print("  全部数据下载完成！")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
