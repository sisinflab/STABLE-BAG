"""
Report progress of the running pipeline by inspecting checkpoints.
Works for both 02_select_lambda.py (validation) and 03_run_test.py (test).
"""

import os
import pickle

base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_path = os.path.join(base_path, "results")

CHECKPOINTS = {
    "validation (02_select_lambda)": os.path.join(results_path, "validation_intermediate.pkl"),
    "test       (03_run_test)":      os.path.join(results_path, "test_checkpoint.pkl"),
}


def report(label, path):
    if not os.path.exists(path):
        print(f"[{label}] no checkpoint at {path}")
        return
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception as exc:
        print(f"[{label}] failed to read checkpoint: {exc}")
        return
    n = len(data)
    print(f"[{label}] processed subjects: {n}")
    if n > 0:
        last = data[-1]
        sid = last.get("id", "?")
        y = last.get("y", "?")
        print(f"  last: id={sid} age={y}")


def main():
    for label, path in CHECKPOINTS.items():
        report(label, path)


if __name__ == "__main__":
    main()
