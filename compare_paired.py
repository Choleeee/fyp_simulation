#!/usr/bin/env python3
"""Compare policy and modular runs that share paired_starts.npz.

    python3 compare_paired.py

Pair k is the file index. A policy file and a modular file with the same
index are one paired approach.
"""

import argparse
import glob
import os
import re

import numpy as np

from task import CONTROL_PERIOD_S

HERE = os.path.dirname(os.path.abspath(__file__))
PAIR_NAME = re.compile(r"paired_(policy|modular)_(\d+)\.npz$")


def approach_time(action):
    return float(len(action)) * CONTROL_PERIOD_S


def command_smoothness(action):
    action = np.asarray(action, dtype=np.float64)
    if len(action) < 2:
        return float("nan")
    return float(np.abs(np.diff(action, axis=0)).mean())


def _text(value):
    array = np.asarray(value)
    if array.shape == () or array.size == 1:
        return str(array.reshape(-1)[0])
    return str(array)


def load_method(data_dir, method):
    records = {}
    for path in glob.glob(os.path.join(data_dir, f"paired_{method}_*.npz")):
        match = PAIR_NAME.search(os.path.basename(path))
        if match is None:
            continue
        episode = np.load(path)
        action = np.asarray(episode["action"])
        records[int(match.group(2))] = {
            "success": bool(np.asarray(episode["success"]).reshape(-1)[0]),
            "reason": _text(episode["reason"]),
            "time_s": approach_time(action),
            "smoothness": command_smoothness(action),
        }
    return records


def _mean(values):
    values = [value for value in values if value is not None and np.isfinite(value)]
    if not values:
        return float("nan")
    return float(np.mean(values))


def _counts(records):
    counts = {}
    for record in records.values():
        if record["success"]:
            continue
        counts[record["reason"]] = counts.get(record["reason"], 0) + 1
    return counts


def _line(method, records):
    if not records:
        return f"  {method:8} no episodes"
    success = sum(record["success"] for record in records.values())
    time_s = _mean([record["time_s"] for record in records.values()])
    smoothness = _mean([record["smoothness"] for record in records.values()])
    return (
        f"  {method:8} {success}/{len(records)} success, "
        f"mean time {time_s:.1f} s, mean smoothness {smoothness:.4f}"
    )


def report(data_dir):
    policy = load_method(data_dir, "policy")
    modular = load_method(data_dir, "modular")
    shared = sorted(set(policy) & set(modular))
    print(f"Paired approaches in {data_dir}")
    print(_line("policy", policy))
    print(_line("modular", modular))
    print(f"  matched pairs {len(shared)}")
    for method, records in (("policy", policy), ("modular", modular)):
        counts = _counts(records)
        if counts:
            listed = ", ".join(f"{reason} {count}" for reason, count in sorted(counts.items()))
            print(f"  {method} failures: {listed}")
    return {"policy": policy, "modular": modular, "shared": shared}


def main():
    parser = argparse.ArgumentParser(description="Compare paired policy and modular approaches.")
    parser.add_argument("--data", default=os.path.join(HERE, "data"))
    args = parser.parse_args()
    report(args.data)


if __name__ == "__main__":
    main()
