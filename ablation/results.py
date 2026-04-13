#!/usr/bin/env python3
"""Save and display ablation results.

Usage::

    python -m ablation.results          # run ablation + save + print
    python -m ablation.results --load   # just print from existing JSON
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from ablation.run_ablation import AblationResult
from ablation.run_ablation import main as run_ablation

_RESULTS_PATH = Path(__file__).parent / "results.json"


def save_results(results: list[AblationResult], path: Path = _RESULTS_PATH) -> None:
    """Persist ablation results to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [asdict(r) for r in results]
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"\nResults saved to {path}")


def load_results(path: Path = _RESULTS_PATH) -> list[AblationResult]:
    """Load ablation results from a JSON file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [AblationResult(**d) for d in data]


def print_table(results: list[AblationResult]) -> None:
    """Print a formatted comparison table to stdout."""
    header = (
        f"{'Variant':<10} {'Params':>10} {'Loss':>8} "
        f"{'PPL':>8} {'Tok/s':>10} {'Mem (MB)':>10}"
    )
    sep = "-" * len(header)

    print(f"\n{sep}")
    print(header)
    print(sep)

    for r in results:
        print(
            f"{r.variant.upper():<10} {r.total_params:>10,} "
            f"{r.final_loss:>8.4f} {r.final_perplexity:>8.2f} "
            f"{r.avg_tokens_per_sec:>10,.0f} {r.peak_memory_mb:>10.1f}"
        )

    print(sep)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load and display existing results instead of re-running.",
    )
    args = parser.parse_args()

    if args.load:
        results = load_results()
    else:
        results = run_ablation()
        save_results(results)

    print_table(results)


if __name__ == "__main__":
    main()
