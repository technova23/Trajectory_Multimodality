from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from pg3d.eval import success_rate_ci_rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    rows = success_rate_ci_rows(summary, methods=args.methods)
    if not rows:
        print("No method summaries found to plot.", file=sys.stderr)
        return 1
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(
            f"Failed to import matplotlib for plotting: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2

    output = args.output or args.summary.parent / "plots" / "comparative_success_ci.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    methods = _unique([str(row["method"]) for row in rows])
    metrics = _unique([str(row["metric"]) for row in rows])
    labels = {
        str(row["metric"]): str(row["label"])
        for row in rows
    }
    by_key = {
        (str(row["method"]), str(row["metric"])): row
        for row in rows
    }

    x = np.arange(len(methods))
    width = min(0.8 / max(len(metrics), 1), 0.25)
    fig, ax = plt.subplots(figsize=args.figsize)
    for idx, metric in enumerate(metrics):
        values: list[float] = []
        yerr: list[list[float]] = [[], []]
        for method in methods:
            row = by_key[(method, metric)]
            values.append(float(row["rate"]))
            yerr[0].append(float(row["err_low"]))
            yerr[1].append(float(row["err_high"]))
        ax.bar(
            x + (idx - (len(metrics) - 1) / 2.0) * width,
            values,
            width,
            label=labels[metric],
            yerr=yerr,
            capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Rate with Wilson 95% CI")
    ax.set_title(args.title)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)
    print(output)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot constrained-reach success rates with Wilson confidence intervals."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--methods", nargs="+", default=None)
    parser.add_argument("--title", default="Constrained reach validation comparison")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--figsize", type=float, nargs=2, default=(10.0, 5.0))
    return parser.parse_args(argv)


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
