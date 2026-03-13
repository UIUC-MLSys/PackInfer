import argparse
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

def get_light_color(hex_color, alpha=0.4):
    rgb = mcolors.to_rgb(hex_color)
    return (*rgb, alpha)

def plot_tpt_combined(base_dir: Path, models: list[str] | None):
    metric = "tpt"
    labels = [label for _, label in DATASETS]
    x = np.arange(len(DATASETS))
    width = 0.35 

    model_dirs = resolve_model_dirs(base_dir, models)
    
    for model_name, model_dir in model_dirs:
        fig, ax = plt.subplots(figsize=(4.28, 3))
        
        series_avg = {}
        series_p99 = {}
        
        for backend, _ in BACKENDS:
            series_avg[backend] = [
                load_metric(model_dir, dataset, backend, metric, "avg") for dataset, _ in DATASETS
            ]
            series_p99[backend] = [
                load_metric(model_dir, dataset, backend, metric, "p99") for dataset, _ in DATASETS
            ]

        for i, (backend, backend_label) in enumerate([("flash_attn", "Flash Attn"), ("packed_attn", "PackInfer")]):
            offset = -width/2 if i == 0 else width/2
            main_color = COLORS[backend]
            light_color = get_light_color(main_color, 0.3)

            bars_p99 = ax.bar(
                x + offset,
                series_p99[backend],
                width,
                color=light_color,
                edgecolor=main_color,
                linewidth=0.5,
                linestyle='--',
                label=f"{backend_label} (P99)"
            )
            
            bars_avg = ax.bar(
                x + offset,
                series_avg[backend],
                width,
                color=main_color,
                label=f"{backend_label} (Avg)"
            )

            for bar in bars_p99:
                height = bar.get_height()
                ax.annotate(
                    f'{height:.0f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10
                )

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Throughput (tok/s)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        all_vals = series_avg["flash_attn"] + series_p99["packed_attn"]
        ax.set_ylim(bottom=min(all_vals) * 0.8, top=max(all_vals) * 1.2)

        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.15),
            ncol=2,
            frameon=False,
            fontsize=12
        )

        fig.tight_layout()

        out_dir = Path("figures") / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = model_name.split("-")[0]
        fig.savefig(out_dir / f"{prefix}_tpt_combined.pdf", bbox_inches="tight")
        print(f"Saved: {out_dir / f'{prefix}_tpt_combined.pdf'}")


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
    args = parser.parse_args()

    plot_tpt_combined(args.base_dir, args.models)


if __name__ == "__main__":
    main()