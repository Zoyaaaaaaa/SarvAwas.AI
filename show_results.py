import re

# Read the results file
with open("model_training_results.txt", "r") as f:
    content = f.read()
    
print("=" * 80)
print("RERA RISK PREDICTION MODEL - TRAINING RESULTS SUMMARY")
print("=" * 80)
print(content)
print("=" * 80)

# Read latest training log
import os
from pathlib import Path

log_dir = Path("training_logs")
log_files = sorted(log_dir.glob("training_*.log"), key=os.path.getctime, reverse=True)

if log_files:
    latest_log = log_files[0]
    print(f"\n📝 Latest Training Log: {latest_log.name}\n")
    
    with open(latest_log, "r") as f:
        lines = f.readlines()
        
    # Print training summary from log
    print("Last 40 lines of training log:")
    print("-" * 80)
    for line in lines[-40:]:
        print(line.rstrip())
