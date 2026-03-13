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
        "font.size": 20,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
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
    #("alpaca", "Alpaca"),
    #("lmsys", "LMSYS"),
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

P95_COLORS = {
    "flash_attn": "#ebc198",
    "packed_attn": "#ae97c7",
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


import matplotlib.patches as mpatches

def plot_avg_p(base_dir: Path, metric: str, models: list[str] | None):
    width = 0.15 
    group_gap = 0.12

    model_dirs = resolve_model_dirs(base_dir, models)
    for model_name, model_dir in model_dirs:
        fig, ax = plt.subplots(figsize=(5, 3.5)) 
        backends = ["flash_attn", "packed_attn"]
        backend_labels = ["Flash Attn", "PackInfer"]
        
        tick_positions = []
        min_val = float('inf')

        for i, backend in enumerate(backends):
            avg_val = load_metric(model_dir, DATASETS[0][0], backend, metric, "avg")
            p95_val = load_metric(model_dir, DATASETS[0][0], backend, metric, "p95")

            min_val = min(min_val, avg_val, p95_val)
            base_x = i * (2 * width + group_gap)

            tick_positions.append(base_x + width / 2)


            ax.bar(
                base_x, avg_val, width * 0.75,
                color=COLORS[backend],
            )

            ax.bar(
                base_x + width, p95_val, width * 0.75,
                color=P95_COLORS[backend],
                hatch="/",
                alpha=0.9,
            )

            for val, x_pos in zip([avg_val, p95_val], [base_x, base_x + width]):
                ax.annotate(
                    f"{val:.2f}",
                    xy=(x_pos, val),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=12
                )

        ax.set_xticks(tick_positions)
        ax.set_xticklabels(backend_labels)
        
        avg_patch = mpatches.Patch(facecolor="white", edgecolor="black", label="Avg")
        p95_patch = mpatches.Patch(facecolor="white", edgecolor="black", hatch="////", label="P95")
        ax.legend(handles=[avg_patch, p95_patch], loc="upper left", frameon=False)

        if metric == "tpt":
            ax.set_ylabel("Throughput (tok/s)")
        else:
            ax.set_ylabel("Latency (ms)")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

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
        ax.set_ylim(y_min, ax.get_ylim()[1])

        plt.tight_layout()

        out_dir = Path("figures") / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = model_name.split("-")[0]
        fig.savefig(out_dir / f"{prefix}_moe_{metric}.pdf", bbox_inches="tight")
        plt.close(fig)


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
        default=["Qwen3-30B-A3B-fp8-pruned"],
        help="Model paths or names to plot",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="tbt",
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
