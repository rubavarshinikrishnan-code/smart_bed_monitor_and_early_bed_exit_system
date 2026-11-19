import csv
import os
import subprocess
import sys
 
FOLDER = "videos"
EXT = (".mp4", ".avi", ".mov", ".mkv")
 
 
def label_for(name):
    n = name.lower()
    if n.startswith(("exit", "fall")):
        return "HIGH"
    if n.startswith("edge"):
        return "MEDIUM"
    if n.startswith(("lying", "sleeping", "sitting", "normal", "safe")):
        return "LOW"
    return None
 
 
if not os.path.isdir(FOLDER):
    raise SystemExit("Create a folder named 'videos' and put your test videos in it.")
 
files = sorted(f for f in os.listdir(FOLDER) if f.lower().endswith(EXT))
rows = [(f, label_for(f)) for f in files]
skipped = [f for f, l in rows if l is None]
rows = [(f, l) for f, l in rows if l]
if not rows:
    raise SystemExit("No videos found with names starting lying/sitting/edge/exit.")
if skipped:
    print("Skipped (name not recognised):", ", ".join(skipped))
 
if os.path.exists("results.csv"):
    os.remove("results.csv")  # fresh results for this evaluation
 
with open("labels.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source", "true_level"])
    w.writerows(rows)
 
for name, level in rows:
    print(f"\nRunning {name} (expected {level}) - press Esc to skip")
    subprocess.run([sys.executable, "bed_monitor.py", os.path.join(FOLDER, name)])
 
print("\n===== EVALUATION =====")
subprocess.run([sys.executable, "evaluate.py"])
