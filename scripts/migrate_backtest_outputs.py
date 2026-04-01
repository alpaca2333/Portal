import os
import shutil
import datetime

def migrate():
    base_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'quant', 'backtest')
    base_dir = os.path.abspath(base_dir)
    
    if not os.path.exists(base_dir):
        print(f"Directory not found: {base_dir}")
        return

    # Define the suffixes to look for
    suffixes = [
        "-trade.csv",
        "_nav.csv",
        "_monthly_returns.csv",
        "_report.md",
        "_factor_contrib.csv"
    ]

    # Group files by strategy name
    strategies = {}

    for filename in os.listdir(base_dir):
        filepath = os.path.join(base_dir, filename)
        if not os.path.isfile(filepath) or filename == "OUTPUT.md":
            continue

        # Find which suffix matches
        matched_suffix = None
        for suffix in suffixes:
            if filename.endswith(suffix):
                matched_suffix = suffix
                break
        
        if matched_suffix:
            strategy_name = filename[:-len(matched_suffix)]
            if strategy_name not in strategies:
                strategies[strategy_name] = []
            strategies[strategy_name].append((filename, matched_suffix, filepath))

    # Process each strategy
    for strategy_name, files in strategies.items():
        # Find the max modification time among these files to use as a unified timestamp
        max_mtime = 0
        for _, _, filepath in files:
            mtime = os.path.getmtime(filepath)
            if mtime > max_mtime:
                max_mtime = mtime
        
        timestamp = datetime.datetime.fromtimestamp(max_mtime).strftime("%Y%m%d_%H%M%S")
        
        # Create strategy directory
        strategy_dir = os.path.join(base_dir, strategy_name)
        os.makedirs(strategy_dir, exist_ok=True)
        
        # Move and rename files
        for filename, suffix, filepath in files:
            new_filename = f"{strategy_name}-{timestamp}{suffix}"
            new_filepath = os.path.join(strategy_dir, new_filename)
            
            print(f"Moving {filename} -> {strategy_name}/{new_filename}")
            shutil.move(filepath, new_filepath)

if __name__ == "__main__":
    migrate()
