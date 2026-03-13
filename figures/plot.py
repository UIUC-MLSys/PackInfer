import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update(
    {
        # general
        "figure.figsize": (6, 4),
        "figure.dpi": 600,
        "figure.constrained_layout.use": True,
        # font
        "font.family": "Arial",
        "font.size": 16,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "legend.fontsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        # lines
        "lines.linewidth": 0.8,
        "lines.markersize": 2,
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

warnings.filterwarnings(
    "ignore",
    message="The figure layout has changed to tight",
    category=UserWarning,
)


DATASETS = [
    ("alpaca", "Alpaca"),
    ("lmsys", "LMSYS"),
    ("text2sql", "Text2SQL"),
]

BACKENDS = [
    ("flash_attn", "Flash Attn"),
    ("packed_attn", "Packed Attn"),
]

COLORS = {
    "flash_attn": "#e3a86b",
    "packed_attn": "#8c6bb1",
}


def load_metric(base_dir: Path, dataset: str, backend: str, metric: str, stat: str) -> float:
    path = base_dir / f"{dataset}_{backend}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    key = f"{stat}_{metric}"
    if key not in data:
        raise KeyError(f"Missing key '{key}' in {path}")
    return float(data[key])


def resolve_model_dirs(base_dir: Path, models: list[str] | None) -> list[tuple[str, Path]]:
    if models:
        model_dirs = []
        for model in models:
            model_path = Path(model)
            name = model_path.name if model_path.name else model_path.parent.name
            model_dirs.append((name, base_dir / name))
        return model_dirs

    subdirs = [p for p in base_dir.iterdir() if p.is_dir()]
    if subdirs:
        return [(p.name, p) for p in sorted(subdirs, key=lambda p: p.name)]

    return [(base_dir.name, base_dir)]


def plot_avg_p(base_dir: Path, metric: str, models: list[str] | None):
    labels = [label for _, label in DATASETS]
    x = np.arange(len(DATASETS))
    width = 0.25

    model_dirs = resolve_model_dirs(base_dir, models)
    for model_name, model_dir in model_dirs:
        fig, axes = plt.subplots(1, 2, figsize=(8, 2.5), squeeze=False)
        for col_idx, (stat, title) in enumerate(zip(["avg", "p95"], ["Average", "P95"])):
            ax = axes[0][col_idx]
            series = {}
            for backend, _ in BACKENDS:
                series[backend] = [
                    load_metric(model_dir, dataset, backend, metric, stat) for dataset, _ in DATASETS
                ]

            flash = series["flash_attn"]
            packed = series["packed_attn"]

            for (dataset, dataset_label), flash_val, packed_val in zip(DATASETS, flash, packed):
                if metric == "tpt":
                    improvement_pct = ((packed_val - flash_val) / flash_val * 100.0) if flash_val else 0.0
                else:
                    improvement_pct = ((flash_val - packed_val) / flash_val * 100.0) if flash_val else 0.0
                print(
                    f"[{model_name}] {metric} {stat} {dataset_label}: "
                    f"packed={packed_val:.2f}, flash={flash_val:.2f}, "
                    f"improvement={improvement_pct:.2f}%"
                )

            bars_flash = ax.bar(
                x - width / 2,
                flash,
                width,
                label="Flash Attn",
                color=COLORS["flash_attn"],
            )
            bars_packed = ax.bar(
                x + width / 2,
                packed,
                width,
                label="Packed Attn",
                color=COLORS["packed_attn"],
            )
            if metric == "tpt":
                ax.set_title(f"{title} Throughput")
            else:
                ax.set_title(f"{title} {metric.upper()} Latency")
            ax.set_xticks(x)
            ax.set_xticklabels(labels)
            if metric == "tpt":
                ax.set_ylabel("Throughput (tok/s)")
            else:
                ax.set_ylabel(f"{title} Time (ms)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            all_vals = flash + packed
            min_val = min(all_vals)
            max_val = max(all_vals)
            if min_val >= 2000:
                step = 2000
            elif min_val >= 100:
                step = 100
            elif min_val >= 50:
                step = 50
            else:
                step = 10
            y_min = (int(min_val // step) * step) if min_val > 0 else step
            if y_min == 0:
                y_min = step
            y_max = max_val * 1.05
            ax.set_ylim(bottom=y_min, top=y_max)

            for bars in (bars_flash, bars_packed):
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(
                        f"{height:.0f}" if metric == "tpt" else f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                    )

        handles, legend_labels = axes[0][0].get_legend_handles_labels()
        fig.legend(
            handles,
            legend_labels,
            loc="upper center",
            ncol=2,
            frameon=False,
            bbox_to_anchor=(0.5, 1.06),
        )
        fig.tight_layout()

        out_dir = Path("figures") / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = model_name.split("-")[0]
        fig.savefig(out_dir / f"{prefix}_{metric}.png", bbox_inches="tight")
        fig.savefig(out_dir / f"{prefix}_{metric}.pdf", bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description="Plot avg and p- latency bars.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing model subfolders with *_flash_attn.json and *_packed_attn.json",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=["Qwen3-4B-fp16-pruned", "Mistral-7B-fp16-pruned"],
        help="Model paths or names to plot",
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
        plot_avg_p(args.base_dir, metric, args.models)


if __name__ == "__main__":
    main()
