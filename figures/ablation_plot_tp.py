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
        "font.size": 22,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "axes.labelsize": 20,
        "axes.titlesize": 20,
        "legend.fontsize": 20,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
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


BACKENDS = [
    ("flash_attn", "Flash Attn"),
    ("packed_attn", "PackInfer"),
]

COLORS = {
    "flash_attn": "#e3a86b",
    "packed_attn": "#8c6bb1",
}

TPSIZE = [1, 2, 4]

def load_all_stats(base_dir: Path, tp_size: int, backend: str, metric: str):
    path = Path.joinpath(base_dir, f"tp{tp_size}-Qwen3-32B-fp8-pruned", f"lmsys_{backend}.json")
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    m = metric.lower()
    return {
        "whislo": data[f"p5_{m}"],
        "q1": data[f"p25_{m}"],
        "med": data[f"p50_{m}"],
        "q3": data[f"p75_{m}"],
        "whishi": data[f"p95_{m}"],
    }

def plot_latency_boxplots(base_dir: Path, metric: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    
    width = 0.2
    offset = 0.2

    legend_handles = []

    for i, tp in enumerate(TPSIZE):
        for j, (backend, label_name) in enumerate(BACKENDS):
            stats = load_all_stats(base_dir, tp, backend, metric)
            if stats is None: continue

            pos = i + (offset if j == 1 else -offset)

            bp = ax.bxp([stats], positions=[pos], widths=width, 
                        patch_artist=True, showfliers=False,
                        manage_ticks=False)
            
            color = COLORS[backend]
            for patch in bp['boxes']:
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
                if i == 0:
                    legend_handles.append(patch)
            
            for element in ['whiskers', 'caps', 'medians']:
                plt.setp(bp[element], color='#333333', linewidth=1.2)
            plt.setp(bp['fliers'], marker='o', markersize=3, markeredgecolor=color, alpha=0.5)


    ax.set_ylabel("Latency (ms)")
    ax.set_xticks(range(len(TPSIZE)))
    ax.set_xticklabels([f"TP={tp}" for tp in TPSIZE])
    
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(legend_handles, [b[1] for b in BACKENDS], loc='upper right', frameon=False)

    output_path = Path("figures")
    output_path.mkdir(exist_ok=True)
    fig.savefig(output_path / f"BoxPlot-TP-{metric}.pdf")
    print(f"Saved: BoxPlot-TP-{metric}.pdf")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=Path, default=Path("data/ablation-tp"))
    parser.add_argument("--metric", type=str, default="ttft", choices=["tbt", "ttft", "ttlt"])
    args, _ = parser.parse_known_args()

    plot_latency_boxplots(args.base_dir, args.metric)

if __name__ == "__main__":
    main()