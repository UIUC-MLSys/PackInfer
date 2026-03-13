import argparse
import json
import warnings
from pathlib import Path
from typing import List
from scipy.interpolate import PchipInterpolator

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
        "font.size": 18,
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
    ("alpaca", "Alpaca"),
    ("lmsys", "LMSYS"),
    ("text2sql", "Text2SQL"),
]

BACKENDS = [
    ("flash_attn", "Flash Attn"),
    ("packed_attn", "PackInfer"),
]

COLORS = {
    "flash_attn": "#e3a86b",
    "packed_attn": "#8c6bb1",
}


def load_metric(base_dir: Path, dataset: str, backend: str, metric: str) -> List[float]:
    path = base_dir / f"{dataset}_{backend}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing result file: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    key = f"{metric}"
    if key not in data:
        raise KeyError(f"Missing key '{key}' in {path}")
    return data[key]


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

def plot_ttlt_cdf(base_dir: Path, models: list[str] | None):
    metric = "ttlt"
    model_dirs = resolve_model_dirs(base_dir, models)
    
    for model_name, model_dir in model_dirs:
        fig, axes = plt.subplots(1, 3, figsize=(8, 3), sharey=True)

        for idx, (dataset_key, dataset_label) in enumerate(DATASETS):
            ax = axes[idx]
            
            for backend_key, backend_label in BACKENDS:
                try:
                    data = load_metric(model_dir, dataset_key, backend_key, metric)
                    if not data: continue

                    sorted_data = np.sort(data)
                    cutoff_percentile = 0.99 
                    keep_idx = int(len(sorted_data) * cutoff_percentile)
                    filtered_data = sorted_data[:keep_idx]
                    x_pts = filtered_data
                    y_pts = np.arange(1, len(filtered_data) + 1) / len(data)

                    x_unique, indices = np.unique(x_pts, return_index=True)
                    y_unique = y_pts[indices]

                    x_final = np.concatenate(([x_unique.min() * 0.999], x_unique))
                    y_final = np.concatenate(([0], y_unique))

                    interp_func = PchipInterpolator(x_final, y_final)
                    x_smooth = np.linspace(x_final.min(), x_final.max(), 1000)
                    y_smooth = np.clip(interp_func(x_smooth), 0, 1)

                    main_color = COLORS[backend_key]
                    ax.plot(
                        x_smooth, 
                        y_smooth, 
                        label=backend_label if idx == 0 else "", 
                        color=main_color,
                        linewidth=1.8,
                        zorder=3
                    )

                    idx_median = np.argmin(np.abs(y_smooth - 0.5))
                    x_median = x_smooth[idx_median]

                    ax.vlines(x=x_median, ymin=0, ymax=0.5, color=main_color, 
                              linestyle=':', linewidth=1.2, alpha=0.8, zorder=2)

                    ax.annotate(f'{x_median:.0f}', 
                                xy=(x_median, 0), 
                                xytext=(10, 8),
                                textcoords="offset points",
                                color=main_color,
                                fontsize=12,
                                ha='center',
                                fontweight='bold')

                    ax.fill_between(x_smooth, y_smooth, alpha=0.1, color=main_color)

                except Exception as e:
                    print(f"Error plotting {dataset_key}-{backend_key}: {e}")

            ax.axhline(y=0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.7, zorder=1)

            ax.set_title(dataset_label)
            ax.set_xlabel("TTLT (ms)")
            if idx == 0:
                ax.set_ylabel("CDF across Requests")
            
            ax.set_ylim(0, 1.05)
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, 1.05),
            ncol=2,
            frameon=False,
            fontsize=16
        )

        fig.tight_layout()

        out_dir = Path("figures") / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = model_name.split("-")[0]
        save_path = out_dir / f"{prefix}_ttlt_cdf.pdf"
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved CDF plot to: {save_path}")

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

    plot_ttlt_cdf(args.base_dir, args.models)


if __name__ == "__main__":
    main()