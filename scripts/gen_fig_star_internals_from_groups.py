import argparse, os, json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d3_group", required=True, help="Path to artifacts/D3_TinyLlama_group_s1 (folder)")
    ap.add_argument("--out", default="fig_star_internals_panel.pdf")
    ap.add_argument("--topk", type=int, default=2, help="STAR router top-k used in runs (for histogram/line)")
    args = ap.parse_args()

    g = Path(args.d3_group)
    ci = g / "ci_by_day.csv"
    if not ci.exists():
        raise SystemExit(f"Missing {ci}")

    df = pd.read_csv(ci)

    # Expect columns like: day, buf_size (buffer size), maybe others
    # Bank size per day ≈ day index (one adapter/day). If you logged buf_size, we’ll show that too.
    if "day" not in df.columns:
        raise SystemExit("ci_by_day.csv must contain a 'day' column.")
    df = df.copy()
    # integer day for plotting
    df["day_i"] = pd.to_numeric(df["day"], errors="coerce")
    df = df.dropna(subset=["day_i"]).sort_values("day_i")

    # Build the 2×1 (since we don’t have per-query router logs here)
    fig = plt.figure(figsize=(7.2, 2.6), dpi=300)

    # (a) Router top-k histogram (single bar at configured top-k)
    ax1 = fig.add_subplot(2,1,1)
    ax1.bar([args.topk],[1.0], width=0.6)
    ax1.set_xlabel("top-k")
    ax1.set_ylabel("proportion")
    ax1.set_title("(a) Router top-k histogram")
    ax1.set_xticks([args.topk])

    # (b) Adapter-bank size vs. day (≈ day index). If buf_size exists, show as dashed overlay (secondary axis).
    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(df["day_i"], df["day_i"], marker="o", linewidth=1.5, label="bank size (≈ day)")
    if "buf_size" in df.columns:
        ax2.plot(df["day_i"], df["buf_size"], linestyle="--", marker="s", linewidth=1.2, label="replay buffer size")
    ax2.set_xlabel("day")
    ax2.set_ylabel("size")
    ax2.set_title("(b) Adapter-bank size vs. day (D3 TinyLlama, STAR)")
    ax2.grid(alpha=0.2, linewidth=0.6)
    ax2.legend(frameon=False, fontsize=8, ncol=2)

    plt.tight_layout()
    out = Path(args.out)
    plt.savefig(out, bbox_inches="tight")
    print(f"Wrote {out.resolve()}")

if __name__ == "__main__":
    main()
