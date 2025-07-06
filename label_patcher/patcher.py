import time
import json
from pathlib import Path
from constants import PENDING_PATH, RELEASED_PATH, POLL_INTERVAL


def patch_released_labels(pending_path: Path, released_path: Path, now_fn=time.time):
    if not pending_path.exists():
        return

    still_pending = []
    released = []

    with open(pending_path, "r") as f:
        for line in f:
            entry = json.loads(line)
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


def run_loop():
    print("Starting ground truth patcher...")
    while True:
        patch_released_labels(PENDING_PATH, RELEASED_PATH)
        time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    run_loop()
