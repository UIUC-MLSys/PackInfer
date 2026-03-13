import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

BACKEND_LABELS = {
    "packed_attn": "Packed Attn",
    "flash_attn": "Flash Attn",
}

BACKEND_COLORS = {
    "packed_attn": "#8c6bb1",
    "flash_attn": "#e3a86b",
}

STAT_LINESTYLES = {
    "avg": "-",
    "p95": "--",
}

plt.rcParams.update(
    {
        # general
        "figure.figsize": (8, 3.5),
        "figure.dpi": 600,
        "figure.constrained_layout.use": True,
        # font
        "font.family": "Arial",
        "font.size": 16,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        # lines
        "lines.linewidth": 1.2,
        "lines.markersize": 4,
        # axes
        "axes.linewidth": 1.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.top": False,
        "ytick.right": False,
        # grid
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.8,
        # legend
        "legend.frameon": False,
        "legend.loc": "best",
        # savefig
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
    }
)


def load_metric_from_dir(model_dir: Path, backend: str, metric: str, stat: str) -> float:
    pattern = f"*_ {backend}.json".replace(" ", "")
    files = sorted(model_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No results matching {pattern} under {model_dir}")

    values = []
    key = f"{stat}_{metric}"
    for path in files:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if key not in data:
            raise KeyError(f"Missing key '{key}' in {path}")
        values.append(float(data[key]))

    return float(np.mean(values))


def collect_experiments(base_dir: Path, prefix: str) -> list[tuple[int, Path]]:
    entries = []
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    for item in base_dir.iterdir():
        if not item.is_dir():
            continue
        match = pattern.match(item.name)
        if match:
            size = int(match.group(1))
            entries.append((size, item))
    return sorted(entries, key=lambda x: x[0])


def metric_label(metric: str) -> tuple[str, str]:
    if metric == "tpt":
        return "Throughput", "Throughput (tok/s)"
    return f"{metric.upper()} Latency", "Time (ms)"


def plot_ablation(base_dir: Path, model: str, backends: list[str], metric: str, stat: str):
    batch_entries = collect_experiments(base_dir, "batch_size")
    group_entries = collect_experiments(base_dir, "group_size")

    if not batch_entries and not group_entries:
        raise FileNotFoundError(f"No batch_size/group_size folders found under {base_dir}")

    title_suffix, y_label = metric_label(metric)

    fig, axes = plt.subplots(1, 2, figsize=(8, 2.5), squeeze=False)

    stats = [stat] + [s for s in ["p95"] if s != stat]
    stat_labels = {"avg": "Avg", "p95": "P95"}
    stat_markers = {"avg": "o", "p95": "s"}

    for ax, entries, x_label, title in [
        (axes[0][0], batch_entries, "Batch Size", "Batch Size Ablation"),
        (axes[0][1], group_entries, "Group Size", "Group Size Ablation"),
    ]:
        if not entries:
            ax.set_visible(False)
            continue

        xs = [size for size, _ in entries]
        for backend in backends:
            backend_label = BACKEND_LABELS.get(backend, backend)
            color = BACKEND_COLORS.get(backend)
            for s in stats:
                ys = [
                    load_metric_from_dir(exp_dir / model, backend, metric, s)
                    for _, exp_dir in entries
                ]

                ax.plot(
                    xs,
                    ys,
                    marker=stat_markers.get(s, "o"),
                    linestyle=STAT_LINESTYLES.get(s, "-"),
                    color=color,
                    label=f"{backend_label} {stat_labels.get(s, s.upper())}",
                )
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.ticklabel_format(style="plain", axis="x")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="center right")

    out_dir = Path("figures") / "ablation" / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model.split('-')[0]}_{metric}_{stat}.png"
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Plot ablation curves for batch_size and group_size experiments")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data/ablation"),
        help="Base directory containing batch_size_* and group_size_* folders",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen3-4B-fp16-pruned",
        help="Model subfolder name under each ablation folder",
    )
    parser.add_argument(
        "--backends",
        nargs="*",
        default=["packed_attn", "flash_attn"],
        help="Backend suffixes in json filenames (e.g., packed_attn flash_attn)",
    )
    parser.add_argument(
        "--stat",
        type=str,
        default="avg",
        choices=["avg", "p95", "p99", "p90"],
        help="Statistic to plot (avg/p95/p99/p90)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="ttft",
        choices=["tbt", "ttft", "ttlt", "tpt"],
        help="Metric suffix to plot (tbt/ttft/ttlt/tpt)",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=["tbt", "ttft", "ttlt", "tpt"],
        choices=["tbt", "ttft", "ttlt", "tpt"],
        help="Metrics to plot in one run (defaults to --metric if not set)",
    )
    args = parser.parse_args()

    metrics = args.metrics if args.metrics else [args.metric]
    for metric in metrics:
        plot_ablation(args.base_dir, args.model, args.backends, metric, args.stat)


if __name__ == "__main__":
    main()
