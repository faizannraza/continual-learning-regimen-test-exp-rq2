# scripts/stats_star.py
import argparse, math, re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from scipy import stats as spstats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

RUN_RE = re.compile(r'^(?P<dataset>D\d+)_s(?P<stream>\d+)_seed(?P<seed>\d+)_(?P<regimen>[^/]+)$')

def parse_run_name(run: str):
    m = RUN_RE.match(run)
    if not m:
        # fall back: try to pull the tail folder if a longer path leaked in
        tail = Path(run).name
        m = RUN_RE.match(tail)
    if not m:
        return dict(dataset="UNK", stream=-1, seed=-1, regimen=run)
    d = m.groupdict()
    d["stream"] = int(d["stream"])
    d["seed"]   = int(d["seed"])
    return d

def t_crit_95(n):
    # two-sided 95% CI critical value; small-n uses t, else ~1.96
    if n <= 1:
        return float("nan")
    if HAVE_SCIPY:
        return float(spstats.t.ppf(0.975, df=n-1))
    # rough approx if SciPy missing
    lookup = {2:12.7,3:4.30,4:3.18,5:2.78,6:2.57,7:2.45,8:2.36,9:2.31,10:2.26,15:2.13,20:2.09,30:2.04}
    return lookup.get(n, 1.96)

def paired_ttest(x, y):
    # returns (t, p)
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    d = x - y
    n = d.size
    if n < 2:
        return (float("nan"), float("nan"))
    mean = d.mean()
    sd = d.std(ddof=1)
    if sd == 0:
        return (0.0, 1.0)
    t = mean / (sd / math.sqrt(n))
    if HAVE_SCIPY:
        p = 2 * (1 - spstats.t.cdf(abs(t), df=n-1))
    else:
        # normal approx, OK for n>=20
        p = 2 * (1 - 0.5*(1+math.erf(abs(t)/math.sqrt(2))))
    return (float(t), float(p))

def non_dominated(points):
    # points: list of (x=FFI, y=Legacy, idx)
    # return indices of Pareto frontier (maximize both)
    pts = sorted(points, key=lambda z: (-z[0], -z[1]))
    front = []
    best_y = -1
    for x,y,i in pts:
        if y > best_y:
            front.append(i)
            best_y = y
    return set(front)

def make_pareto(df, outdir, title_suffix="all"):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    # last day per run
    last = df.sort_values(["Run","day"]).groupby("Run", as_index=False).tail(1).copy()
    # compute total training seconds per run
    cost = df.groupby("Run")["train_seconds"].sum().rename("total_train_seconds")
    last = last.merge(cost, on="Run", how="left")

    # Ensure meta columns present exactly once (avoid duplicates)
    meta = last["Run"].apply(parse_run_name).apply(pd.Series)
    for c in ["dataset","stream","seed","regimen"]:
        if c not in last.columns:
            last[c] = meta[c]
    last = last.loc[:, ~last.columns.duplicated()]

    # color map per regimen
    regimens = sorted(last["regimen"].astype(str).unique())
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = {r: color_cycle[i % len(color_cycle)] for i, r in enumerate(regimens)}

    # global pareto
    pts = [(row["FFI_EM"], row["Legacy_EM"], i) for i, row in last.iterrows()]
    frontier = non_dominated(pts)

    plt.figure(figsize=(6.2,5.2))
    # remember first index per regimen for legend
    first_idx = {r: last.index[last["regimen"] == r][0] for r in regimens if (last["regimen"] == r).any()}
    for i, row in last.iterrows():
        s = 15 + 85 * (row["total_train_seconds"] / (last["total_train_seconds"].max()+1e-9))
        label = row["regimen"] if i == first_idx.get(row["regimen"], None) else ""
        plt.scatter(row["FFI_EM"], row["Legacy_EM"], s=s, alpha=0.7,
                    label=label, c=colors[row["regimen"]])
    # draw frontier
    fpts = last.loc[list(frontier)][["FFI_EM","Legacy_EM"]].sort_values(["FFI_EM","Legacy_EM"])
    if len(fpts) >= 2:
        plt.plot(fpts["FFI_EM"], fpts["Legacy_EM"], lw=2, linestyle="--", color="k", label="Pareto frontier")

    plt.xlabel("FFI (EM) – Fresh Facts")
    plt.ylabel("Legacy (EM) – Retention")
    plt.title(f"Pareto (size ∝ total train seconds) – {title_suffix}")
    handles, labels = plt.gca().get_legend_handles_labels()
    # dedupe legend entries and keep regimens + frontier
    lab_seen, handles2, labels2 = set(), [], []
    for h,lab in zip(handles, labels):
        if lab not in lab_seen and lab != "":
            lab_seen.add(lab); handles2.append(h); labels2.append(lab)
    plt.legend(handles2, labels2, fontsize=8, loc="lower left", frameon=True)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / f"pareto_{title_suffix}.png", dpi=180)

    # per-dataset paretos
    for dset, g in last.groupby("dataset"):
        pts = [(row["FFI_EM"], row["Legacy_EM"], i) for i, row in g.iterrows()]
        frontier = non_dominated(pts)
        plt.figure(figsize=(6.2,5.2))
        first_idx_g = {r: g.index[g["regimen"] == r][0] for r in g["regimen"].unique() if (g["regimen"] == r).any()}
        for i, row in g.iterrows():
            s = 15 + 85 * (row["total_train_seconds"] / (g["total_train_seconds"].max()+1e-9))
            label = row["regimen"] if i == first_idx_g.get(row["regimen"], None) else ""
            plt.scatter(row["FFI_EM"], row["Legacy_EM"], s=s, alpha=0.7,
                        label=label, c=colors[row["regimen"]])
        fpts = g.loc[list(frontier)][["FFI_EM","Legacy_EM"]].sort_values(["FFI_EM","Legacy_EM"])
        if len(fpts) >= 2:
            plt.plot(fpts["FFI_EM"], fpts["Legacy_EM"], lw=2, linestyle="--", color="k", label="Pareto frontier")
        plt.xlabel("FFI (EM)")
        plt.ylabel("Legacy (EM)")
        plt.title(f"Pareto – {dset}")
        handles, labels = plt.gca().get_legend_handles_labels()
        lab_seen, handles2, labels2 = set(), [], []
        for h,lab in zip(handles, labels):
            if lab not in lab_seen and lab != "":
                lab_seen.add(lab); handles2.append(h); labels2.append(lab)
        plt.legend(handles2, labels2, fontsize=8, loc="lower left", frameon=True)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / f"pareto_{dset}.png", dpi=180)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concat_csv", default="artifacts/figures_star/all_runs_concat.csv")
    ap.add_argument("--out", default="artifacts/figures_star")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.concat_csv)

    # Harmonize column names if needed
    if "FFI_EM" not in df.columns and "FFI" in df.columns:
        df["FFI_EM"] = df["FFI"]
    if "Legacy_EM" not in df.columns and "Legacy" in df.columns:
        df["Legacy_EM"] = df["Legacy"]

    # parse run metadata from folder names (once)
    meta = df["Run"].apply(parse_run_name).apply(pd.Series)  # dataset, stream, seed, regimen
    df = pd.concat([df, meta], axis=1)
    df = df.loc[:, ~df.columns.duplicated()]   # <-- prevent duplicate columns later

    # ============ 95% CIs per day/regimen/dataset ============
    group_cols = ["dataset", "regimen", "day"]
    agg_rows = []
    for (dset, reg, day), g in df.groupby(group_cols):
        n = g["FFI_EM"].count()
        for metric in ["FFI_EM", "Legacy_EM"] + (["FFI_F1"] if "FFI_F1" in df.columns else []):
            m = g[metric].mean()
            s = g[metric].std(ddof=1)
            tcrit = t_crit_95(n)
            ci = tcrit * (s / math.sqrt(n)) if n>1 and not math.isnan(tcrit) else float("nan")
            agg_rows.append(dict(dataset=dset, regimen=reg, day=day, metric=metric, n=n, mean=m, sd=s, ci95=ci))
    ci_df = pd.DataFrame(agg_rows)
    ci_df.sort_values(["dataset","regimen","metric","day"]).to_csv(out/"ci_by_day.csv", index=False)

    # ============ Paired tests: STAR vs baselines ============
    # Build unit of pairing: (dataset, stream, seed, day)
    key_cols = ["dataset","stream","seed","day"]
    wide = df.pivot_table(index=key_cols, columns="regimen", values=["FFI_EM","Legacy_EM"], aggfunc="mean")
    # Only keep rows where STAR and the baseline both exist
    paired_rows = []
    reg_cols = [c for c in wide["FFI_EM"].columns if isinstance(c, str)]
    baselines = [c for c in reg_cols if c.lower() != "star"]
    for b in baselines:
        # dropna on both STAR and baseline
        if ("star" not in wide["FFI_EM"].columns) or (b not in wide["FFI_EM"].columns):
            continue
        needed = [("FFI_EM","star"), ("FFI_EM",b), ("Legacy_EM","star"), ("Legacy_EM",b)]
        w = wide.dropna(subset=needed, how="any")
        if w.empty: continue
        ffi_star = w[("FFI_EM","star")].values
        ffi_base = w[("FFI_EM", b)].values
        leg_star = w[("Legacy_EM","star")].values
        leg_base = w[("Legacy_EM", b)].values
        t_ffi, p_ffi = paired_ttest(ffi_star, ffi_base)
        t_leg, p_leg = paired_ttest(leg_star, leg_base)
        paired_rows.append(dict(baseline=b, n=w.shape[0],
                                t_FFI=t_ffi, p_FFI=p_ffi,
                                delta_FFI=float(np.mean(ffi_star-ffi_base)),
                                t_Leg=t_leg, p_Leg=p_leg,
                                delta_Leg=float(np.mean(leg_star-leg_base))))
    paired_df = pd.DataFrame(paired_rows).sort_values("baseline")
    paired_df.to_csv(out/"paired_tests_star_vs_baselines.csv", index=False)

    # ============ Pareto plots ============
    make_pareto(df, outdir=out, title_suffix="all")

    print(f"Wrote:\n  {out/'ci_by_day.csv'}\n  {out/'paired_tests_star_vs_baselines.csv'}\n  {out/'pareto_all.png'} (plus per-dataset paretos)")

if __name__ == "__main__":
    main()
