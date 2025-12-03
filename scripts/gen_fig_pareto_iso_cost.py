import argparse, math
from pathlib import Path
from collections import OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.ticker import FormatStrFormatter

# --- Colorblind-safe mapping (as requested) ---
# LoRA = near-black, Replay = blue, Hybrid = orange, STAR = vermillion/red
REG_COL = OrderedDict([
  ("lora",   "#000000"),
  ("replay", "#0072B2"),
  ("hybrid", "#E69F00"),
  ("star",   "#D55E00"),
])
DS_MARK = {"D1":"o","D2":"s","D3":"^"}  # ● ■ ▲

def read_group_csv(group_path: Path) -> pd.DataFrame:
    if group_path.is_file():
        df = pd.read_csv(group_path)
        group = group_path.parent.name
    else:
        csv_path = group_path / "all_runs_concat.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing {csv_path}")
        df = pd.read_csv(csv_path)
        group = group_path.name
    df["group"] = group
    if "regimen" in df.columns:
        df["regimen"] = (
            df["regimen"].astype(str)
                          .str.replace("_only","", regex=False)
                          .str.lower()
        )
    return df

def dataset_from_group(gname: str) -> str:
    G = gname.upper()
    for d in ("D1","D2","D3"):
        if d in G: return d
    return "D?"

def coalesce(df, *names):
    for c in names:
        if c in df.columns: return c
    return None

def unique_in_order(seq):
    seen=set(); out=[]
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def compute_axes_limits(vals):
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size==0: return (0.00, 0.08)
    lo, hi = float(v.min()), float(v.max())
    if hi==lo:
        lo = max(0.0, lo-0.005); hi = lo+0.01
    pad = 0.06*(hi-lo)
    lo = max(0.0, lo-pad)
    hi = min(1.00, hi+pad)
    if hi <= 0.12:
        lo = 0.00
        hi = max(0.06, min(0.08, hi))
    return (lo, hi)

def to_decimals(x_raw, y_raw):
    mx = float(np.nanmax(np.r_[x_raw, y_raw]))
    if mx > 1.2:  # looks like percentages → convert to decimals
        return x_raw/100.0, y_raw/100.0
    return x_raw, y_raw

def nondominated_front(xs, ys):
    pts = [(i, xs[i], ys[i]) for i in range(len(xs))]
    keep = []
    for i, x, y in pts:
        dominated = False
        for j, xx, yy in pts:
            if j!=i and xx>=x and yy>=y and (xx>x or yy>y):
                dominated = True; break
        if not dominated:
            keep.append((i, x, y))
    keep.sort(key=lambda t: t[1])
    return [i for (i,_,_) in keep]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", required=True,
                    help="paths like artifacts/figures_star/D1_TinyLlama_group_s1 ...")
    ap.add_argument("--out", default="fig_pareto_iso_cost.pdf")
    ap.add_argument("--levels", nargs="*", type=float, default=[2,4,6,9,60,90])
    ap.add_argument("--seed", type=int, default=7, help="jitter seed")
    args = ap.parse_args()

    frames = [ read_group_csv(Path(g)) for g in args.groups ]
    for i in range(len(frames)):
        frames[i] = frames[i].reset_index(drop=True)
    df = pd.concat(frames, ignore_index=True)

    col_ffi = coalesce(df, "FFI_EM","FFI","ffi_em")
    col_leg = coalesce(df, "Legacy_EM","LegacyAcc","Legacy","legacy_em")
    if col_ffi is None or col_leg is None:
        raise SystemExit(f"Missing FFI/Legacy columns. Have: {list(df.columns)}")

    if "day" in df.columns:
        df["day_i"] = pd.to_numeric(df["day"], errors="coerce")
        idx = df.groupby(["group","regimen"])["day_i"].transform("max") == df["day_i"]
        agg = df[idx].copy()
    else:
        agg = df.copy()

    if "train_seconds" in df.columns:
        cost = df.groupby(["group","regimen"])["train_seconds"].mean().reset_index(name="sec_per_day")
    else:
        cost = agg[["group","regimen"]].drop_duplicates()
        cost["sec_per_day"] = np.nan

    merged = agg.merge(cost, on=["group","regimen"], how="left")
    merged["min_per_day"] = merged["sec_per_day"] / 60.0

    x_raw = merged[col_ffi].astype(float).to_numpy()
    y_raw = merged[col_leg].astype(float).to_numpy()
    x, y = to_decimals(x_raw, y_raw)

    z = merged["min_per_day"].astype(float).to_numpy()
    reg = merged["regimen"].astype(str).to_numpy()
    grp = merged["group"].astype(str).to_numpy()
    dset = np.array([dataset_from_group(g) for g in grp])

    # --- Aesthetics / typography ---
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 9,
        "legend.fontsize": 8
    })
    fig = plt.figure(figsize=(7.2, 3.55), dpi=300)
    ax = fig.add_subplot(1,1,1)

    # Axis titles (exact capitalization) + decimals + thin ticks
    ax.set_xlabel("Freshness EM (FFI)")
    ax.set_ylabel("Legacy EM")
    xlo,xhi = compute_axes_limits(x); ylo,yhi = compute_axes_limits(y)
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', which='major', length=3.5, width=0.6)
    # major grid only (no minor)
    ax.grid(True, which="major", alpha=0.22, linewidth=0.55)

    # Optional short title (or remove if you prefer)
    ax.set_title("Pareto frontier with iso-cost overlays (TinyLlama, D1–D3)", pad=6)

    # Scatter (bigger markers + thin black edge)
    rng = np.random.RandomState(args.seed)
    jx = (xhi-xlo) * 0.0025
    jy = (yhi-ylo) * 0.0025

    xs, ys, zs = [], [], []
    for i in range(len(merged)):
        xi = float(x[i]); yi = float(y[i]); zi = float(z[i]) if i < len(z) else float("nan")
        ri = reg[i].lower(); di = dset[i]
        c = REG_COL.get(ri, "#666666"); m = DS_MARK.get(di, "o")
        xi += rng.uniform(-jx, jx); yi += rng.uniform(-jy, jy)
        ax.scatter([xi],[yi], s=90, c=[c], marker=m,
                   edgecolors="black", linewidths=0.6, zorder=3)
        xs.append(xi); ys.append(yi); zs.append(zi)

    # Pareto envelope (thin, semi-transparent, behind points)
    front_idx = nondominated_front(xs, ys)
    if len(front_idx) >= 2:
        fx = [xs[i] for i in front_idx]
        fy = [ys[i] for i in front_idx]
        ax.plot(fx, fy, color="#D55E00", linewidth=1.2, alpha=0.35, zorder=1)

    # Iso-cost contours: thin light gray, behind points; one label per level, horizontal
    xsA = np.asarray(xs); ysA = np.asarray(ys); zsA = np.asarray(zs, dtype=float)
    mask = np.isfinite(zsA)
    if mask.sum() >= 3:
        tri = mtri.Triangulation(xsA[mask], ysA[mask])
        levels = sorted(set(args.levels))
        cs = ax.tricontour(tri, zsA[mask], levels=levels,
                           linewidths=0.55, linestyles="dashed",
                           colors="#9e9e9e", alpha=0.95, zorder=0)

        # remove repeated labels and force horizontal style
        # first place labels (matplotlib decides positions); then prune dups & flatten rotation
        lbls = ax.clabel(cs, inline=False, fontsize=7,
                         fmt=lambda v: f"{int(v)} m/day", colors="#7f7f7f")
        seen = set()
        for t in lbls:
            txt = t.get_text()
            if txt in seen:
                t.set_visible(False)
            else:
                seen.add(txt)
            t.set_rotation(0)   # horizontal
            t.set_bbox(dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))

    # Legends (compact, single line titles)
    reg_items = [r for r in REG_COL.keys() if r in set(reg)]
    color_handles = [plt.Line2D([0],[0], marker="o", linestyle="",
                                markersize=7, markerfacecolor=REG_COL[r],
                                markeredgecolor="black", markeredgewidth=0.6, label=r.capitalize())
                     for r in reg_items]
    leg1 = ax.legend(color_handles, [h.get_label() for h in color_handles],
                     title="Regimen (color): LoRA, Replay, Hybrid, STAR",
                     frameon=False, ncol=min(4,len(color_handles)),
                     loc="lower right", bbox_to_anchor=(1.0, 0.02), borderaxespad=0.3)
    ax.add_artist(leg1)

    ds_items = unique_in_order([d for d in ["D1","D2","D3"] if d in set(dset)])
    mark_handles = [plt.Line2D([0],[0], marker=DS_MARK[d], linestyle="",
                               markersize=7, markerfacecolor="#7A7A7A",
                               markeredgecolor="black", markeredgewidth=0.6, label=d)
                    for d in ds_items]
    ax.legend(mark_handles, [h.get_label() for h in mark_handles],
              title="Dataset (marker): D1, D2, D3",
              frameon=False, ncol=len(mark_handles),
              loc="upper left", bbox_to_anchor=(0.01, 0.99), borderaxespad=0.3)

    # pad slightly so labels never clip
    plt.subplots_adjust(right=0.985, top=0.965)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"Wrote {out.resolve()}")

if __name__ == "__main__":
    main()
