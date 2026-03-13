import argparse
import json
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

HARDWARE = ["A100", "A40", "H200"]
BACKENDS = [("flash_attn", "Flash Attn"), ("packed_attn", "PackInfer")]
COLORS = {"flash_attn": "#e3a86b", "packed_attn": "#8c6bb1"}

def load_metric(base_dir: Path, hardware: str, backend: str, metric: str, stat: str) -> float:
    path = base_dir / hardware / f"alpaca_{backend}.json"
    if not path.exists():
        return 0.0
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return float(data.get(f"{stat}_{metric}", 0.0))

def plot_avg_only(base_dir: Path, metric: str):
    x = np.arange(len(HARDWARE))
    width = 0.25

    fig, ax = plt.subplots(figsize=(5, 3.5))
    flash_data = [load_metric(base_dir, hw, "flash_attn", metric, "avg") for hw in HARDWARE]
    packed_data = [load_metric(base_dir, hw, "packed_attn", metric, "avg") for hw in HARDWARE]

    rects1 = ax.bar(x - width/2, flash_data, width, label="Flash Attn", color=COLORS["flash_attn"])
    rects2 = ax.bar(x + width/2, packed_data, width, label="PackInfer", color=COLORS["packed_attn"])

    ax.set_ylabel(f"Average {metric.upper()} (ms)")
    ax.set_xticks(x)
    ax.set_xticklabels(HARDWARE)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylim(0, max(flash_data + packed_data) * 1.2)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), 
                        textcoords="offset points",
                        ha="center", va="bottom", fontsize=12)

    autolabel(rects1)
    autolabel(rects2)
    ax.legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.15))

    fig.tight_layout()
    output_path = Path("figures")
    output_path.mkdir(exist_ok=True)
    fig.savefig(output_path / f"hardware-{metric}-avg.pdf", bbox_inches="tight")
    print(f"Plot saved to figures/hardware-{metric}-avg.pdf")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path("data/ablation-gpu"))
    parser.add_argument("--metric", type=str, default="tbt", choices=["tbt", "ttft", "ttlt"])
    args = parser.parse_args()
    plot_avg_only(args.base_dir, args.metric)

if __name__ == "__main__":
    main()