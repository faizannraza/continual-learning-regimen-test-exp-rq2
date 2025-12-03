import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re

ORDER = ["star","hybrid","lora_only","replay_only"]

def _infer_regimen_from_run(run: str) -> str:
    if not isinstance(run, str):
        return "unknown"
    # take the suffix after the last underscore
    tail = run.split("_")[-1].lower()
    if tail == "lora":   return "lora_only"
    if tail == "replay": return "replay_only"
    if tail in {"star","hybrid","lora_only","replay_only"}:
        return tail
    return tail or "unknown"

def _infer_dataset_from_run(run: str) -> str:
    if not isinstance(run, str):
        return "D?"
    m = re.match(r'^(D\d+)_', run)
    return m.group(1) if m else "D?"

def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    # metric name harmonization
    if "FFI_EM" not in df.columns and "FFI" in df.columns:       df["FFI_EM"] = df["FFI"]
    if "Legacy_EM" not in df.columns and "Legacy" in df.columns: df["Legacy_EM"] = df["Legacy"]

    # ensure Run exists (some concat files already have it)
    if "Run" not in df.columns or df["Run"].isnull().all():
        # fallback: synthesize from regimen if present
        if "regimen" in df.columns and df["regimen"].notna().any():
            df["Run"] = df["regimen"].astype(str).apply(lambda r: f"D?_s?_seed?_{r}")
        else:
            df["Run"] = "D?_s?_seed?_unknown"

    # infer regimen if missing
    if "regimen" not in df.columns or df["regimen"].isnull().all():
        df["regimen"] = df["Run"].astype(str).apply(_infer_regimen_from_run)
    else:
        # fill nulls from Run, then coerce lora/replay names
        df["regimen"] = df["regimen"].astype(str)
        mask = df["regimen"].isin(["None","nan","NaN","", "unknown"])
        df.loc[mask, "regimen"] = df.loc[mask, "Run"].astype(str).apply(_infer_regimen_from_run)
        df["regimen"] = df["regimen"].replace({"lora":"lora_only","replay":"replay_only"})

    # dataset tag
    if "dataset" not in df.columns or df["dataset"].isnull().all():
        df["dataset"] = df["Run"].astype(str).apply(_infer_dataset_from_run)
    else:
        m = df["dataset"].astype(str).isin(["None","nan","NaN",""])
        df.loc[m, "dataset"] = df.loc[m, "Run"].astype(str).apply(_infer_dataset_from_run)

    # numeric types
    for c in ["FFI_EM","Legacy_EM","train_seconds","day"]:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce")

    return df

def _last_day_rows(df: pd.DataFrame) -> pd.DataFrame:
    last = df.sort_values(["Run","day"]).groupby("Run", as_index=False).tail(1).copy()
    if "train_seconds" in df.columns:
        cost = df.groupby("Run")["train_seconds"].sum().rename("total_train_seconds")
        last = last.merge(cost, on="Run", how="left")
    else:
        last["total_train_seconds"] = 0.0
    return last

def _size_from_cost(cost: pd.Series) -> pd.Series:
    c = cost.fillna(0.0).clip(lower=0)
    if c.max() <= 0:
        return pd.Series(np.full(len(c), 120.0), index=c.index)
    return 120.0 + 1680.0 * np.sqrt(c / (c.max() + 1e-9))

def _color_map(categories):
    base = plt.rcParams['axes.prop_cycle'].by_key()['color']
    cmap = {}
    for i, r in enumerate(ORDER):
        cmap[r] = base[i % len(base)]
    for r in categories:
        if r not in cmap:
            cmap[r] = base[len(cmap) % len(base)]
    return cmap

def _plot_group(df, outpath, title):
    if df.empty: return
    colors = _color_map(df["regimen"].astype(str).unique())
    sizes  = _size_from_cost(df["total_train_seconds"]) if "total_train_seconds" in df.columns else pd.Series(600, index=df.index)

    plt.figure(figsize=(9.4,7.2))
    for reg in ORDER:
        g = df[df["regimen"]==reg]
        if g.empty: continue
        plt.scatter(
            g["FFI_EM"], g["Legacy_EM"],
            s=sizes.loc[g.index],
            c=colors[reg],
            edgecolors="black", linewidths=0.7, alpha=0.95,
            label=reg
        )

    plt.xlabel("FFI (EM) – Fresh Facts")
    plt.ylabel("Legacy (EM) – Retention")
    plt.title(title)

    # Legend outside (right), so no point is obscured
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)

    # Pad axes a bit
    x = df["FFI_EM"].dropna(); y = df["Legacy_EM"].dropna()
    if len(x) and len(y):
        xpad = (x.max()-x.min())*0.1 + 1e-3
        ypad = (y.max()-y.min())*0.1 + 1e-3
        plt.xlim(x.min()-xpad, x.max()+xpad)
        plt.ylim(y.min()-ypad, y.max()+ypad)

    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close()

def make_pareto(csv_path: str, outdir: str, title_suffix: str = "all"):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(csv_path)
    df = _normalize_cols(df)
    last = _last_day_rows(df)

    _plot_group(last, out / f"pareto_{title_suffix}.png",
                f"Pareto (size ∝ total train seconds) – {title_suffix}")

    for dset, g in last.groupby("dataset"):
        _plot_group(g, out / f"pareto_{dset}.png", f"Pareto – {dset}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--title", default="all")
    args = ap.parse_args()
    make_pareto(args.csv, args.outdir, args.title)

if __name__ == "__main__":
    main()
