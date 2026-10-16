"""Mean and standard deviation of downstream evaluation summaries over seeds.

Reads the ``summary.json`` written by :mod:`downstream.evaluate` for each run and
reports top-1 accuracy (and worst-group accuracy for Waterbirds / Spawrious)
averaged over the runs.

    python -m downstream.aggregate --runs OUT/seed0/eval/summary.json OUT/seed1/eval/summary.json \
        OUT/seed2/eval/summary.json --output OUT/summary.json
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib

import numpy as np

logger = logging.getLogger("downstream.aggregate")

METRICS = ("top1_acc", "worst_group_acc")


def aggregate(paths: list[pathlib.Path]) -> dict[str, object]:
    runs = [json.loads(path.read_text()) for path in paths]
    summary: dict[str, object] = {"n_runs": len(runs), "runs": [str(path) for path in paths]}
    for metric in METRICS:
        values = [run[metric] for run in runs if metric in run]
        if not values:
            continue
        if len(values) != len(runs):
            raise ValueError(f"{metric} is missing from some runs.")
        array = np.asarray(values, dtype=np.float64)
        summary[metric] = {
            "mean": float(array.mean()),
            "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
            "values": array.tolist(),
        }
    return summary


def main(argv: list[str] | None = None) -> dict[str, object]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=pathlib.Path, nargs="+", required=True, help="per-seed eval summary.json files")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(name)s] %(message)s")
    summary = aggregate(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    for metric in METRICS:
        if metric in summary:
            stats = summary[metric]
            logger.info("%s over %d runs: %.2f +- %.2f", metric, summary["n_runs"], 100 * stats["mean"], 100 * stats["std"])
    return summary


if __name__ == "__main__":
    main()
