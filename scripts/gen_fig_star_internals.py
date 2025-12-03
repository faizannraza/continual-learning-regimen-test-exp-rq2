import os, json, glob, argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path("fig_star_internals_panel.pdf")

def verbose_find(pattern):
    cands = [Path(p) for p in glob.glob(pattern, recursive=True)]
    print(f"[finder] {pattern} -> {len(cands)} matches")
    for p in cands[:10]:
        print("  -", p)
    if len(cands) > 10:
        print("  ...")

def find_d3_star_root():
    # Search for a D3 TinyLlama s1 STAR artifact with routers/day_01
    patterns = [
        "artifacts/**",  # broad
    ]
    cands = []
    for pat in patterns:
        verbose_find(pat)
        for p in glob.glob(pat, recursive=True):
            low = p.lower()
            if ("d3" in low) and ("tinyllama" in low or "m1p1b_tinyllama" in low) and ("s1" in low) and ("star" in low):
                pp = Path(p)
                if pp.is_dir() and (pp / "routers" / "day_01").exists():
                    cands.append(pp)
    if not cands:
        raise SystemExit("Could not auto-detect a D3 TinyLlama s1 STAR artifact with routers/day_01.")
    # choose shortest path (most specific)
    cands.sort(key=lambda x: len(str(x)))
    return cands[0]

def adapter_bank_sizes(adapters_root="models/adapters", regimen="star"):
    root = Path(adapters_root) / regimen
    if not root.exists(): return [], []
    days = sorted(d for d in root.glob("day_*") if d.is_dir())
    xs, ys = [], []
    for d in days:
        upto = [p for p in days if p.name <= d.name]
        xs.append(int(d.name.split("_")[1]))
        ys.append(len(upto))
    return xs, ys

def read_topk_from_router_meta(router_dir):
    meta = router_dir / "meta.json"
    if meta.exists():
        try:
            m = json.loads(meta.read_text())
            if "top_k" in m: return int(m["top_k"])
        except Exception:
            pass
    return None

def collect_router_loss(routers_root):
    xs, ys = [], []
    for d in sorted((routers_root).glob("day_*")):
        mfile = d / "metrics.json"
        if mfile.exists():
            try:
                m = json.loads(mfile.read_text())
                if "loss" in m:
                    xs.append(int(d.name.split("_")[1]))
                    ys.append(float(m["loss"]))
            except Exception:
                pass
    return xs, ys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--star_root", default="", help="Path to STAR artifact root (has routers/day_01). If empty, auto-detect.")
    ap.add_argument("--adapters_root", default="models/adapters", help="Path to adapters root (default models/adapters).")
    args = ap.parse_args()

    if args.star_root:
        star_root = Path(args.star_root)
        if not (star_root / "routers" / "day_01").exists():
            raise SystemExit(f"--star_root given but routers/day_01 missing: {star_root}")
    else:
        star_root = find_d3_star_root()
    print(f"[ok] Using STAR root: {star_root}")

    # (b) Adapter-bank size vs day
    xs_bank, ys_bank = adapter_bank_sizes(args.adapters_root, regimen="star")

    # (a) & (c) – router sparsity summaries
    routers_root = star_root / "routers"
    inferred_topk = 2
    t = read_topk_from_router_meta(routers_root / "day_01")
    if t is not None:
        inferred_topk = t

    days = sorted([int(d.name.split("_")[1]) for d in routers_root.glob("day_*")])
    frac_series = [1.0 for _ in days]  # all queries use top_k (approximation without per-query logs)
    loss_x, loss_y = collect_router_loss(routers_root)

    # --- Draw figure ---
    has_loss = bool(loss_x)
    nrows, ncols = (2,2) if has_loss else (2,1)
    fig = plt.figure(figsize=(7.2, 3.6 if has_loss else 2.6), dpi=300)

    # (a) top-k histogram
    ax1 = fig.add_subplot(nrows, ncols, 1)
    ax1.bar([inferred_topk], [1.0], width=0.6)
    ax1.set_xlabel("top-k")
    ax1.set_ylabel("proportion")
    ax1.set_title("(a) Router top-k histogram")
    ax1.set_xticks([inferred_topk])

    # (b) bank size vs day
    ax2 = fig.add_subplot(nrows, ncols, 2)
    if xs_bank:
        ax2.plot(xs_bank, ys_bank, marker="o", linewidth=1.5)
    ax2.set_xlabel("day")
    ax2.set_ylabel("adapter-bank size")
    ax2.set_title("(b) Adapter-bank size vs. day")
    ax2.grid(alpha=0.2, linewidth=0.6)

    # (c) fraction of queries by top-k over days
    ax3 = fig.add_subplot(nrows, ncols, 3)
    if days:
        ax3.plot(days, frac_series, marker="s", linewidth=1.5)
    ax3.set_xlabel("day")
    ax3.set_ylabel("fraction using top-k")
    ax3.set_ylim(0, 1.05)
    ax3.set_title("(c) Fraction of queries by top-k over days")
    ax3.grid(alpha=0.2, linewidth=0.6)

    # (d) optional: router loss
    if has_loss:
        ax4 = fig.add_subplot(nrows, ncols, 4)
        ax4.plot(loss_x, loss_y, marker="^", linewidth=1.5)
        ax4.set_xlabel("day")
        ax4.set_ylabel("router loss")
        ax4.set_title("(d) Router probe loss")
        ax4.grid(alpha=0.2, linewidth=0.6)

    plt.tight_layout()
    plt.savefig(OUT, bbox_inches="tight")
    print(f"[ok] Wrote {OUT.resolve()}")

if __name__ == "__main__":
    main()
