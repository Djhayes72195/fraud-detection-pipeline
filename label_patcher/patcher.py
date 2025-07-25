import time
import json
from pathlib import Path
from constants import PENDING_PATH, RELEASED_PATH, POLL_INTERVAL


def patch_released_labels(pending_path: Path, released_path: Path, now_fn=time.time):
    if not pending_path.exists():
        return

    still_pending = []
    released = []

    for entry in load_jsonl_lines(pending_path):
        if now_fn() >= entry["label_available_at"]:
            released.append(entry)
        else:
            still_pending.append(entry)

    if released:
        with open(released_path, "a") as out_f:
            for entry in released:
                out_f.write(json.dumps(entry) + "\n")

    with open(pending_path, "w") as f:  # "w" mode --> "released" records will be overwritten
        for entry in still_pending:
            f.write(json.dumps(entry) + "\n")

    if released:
        print(f"[{time.strftime('%X')}] Released {len(released)} label(s)")

def load_jsonl_lines(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"Skipping malformed line: {line}")
                continue

def run_loop():
    print("Starting ground truth patcher...")
    while True:
        patch_released_labels(PENDING_PATH, RELEASED_PATH)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run_loop()
