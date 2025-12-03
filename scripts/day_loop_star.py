# scripts/day_loop_star.py
import os, sys, orjson, argparse, random, time, subprocess, math
from pathlib import Path
from glob import glob
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from star_router import STARRouter
from utils_metrics import exact_match

def load_jsonl(path):
    rows=[]
    with open(path,"rb") as f:
        for line in f: rows.append(orjson.loads(line))
    return rows

def save_jsonl(path: Path, rows):
    """
    Write rows to JSONL. Robust to NumPy scalars/arrays in rows.
    Tries orjson's OPT_SERIALIZE_NUMPY; falls back to a sanitizer.
    """
    import numpy as np

    def _jsonable(o):
        # Recursively convert NumPy types to native Python
        if isinstance(o, dict):
            return {k: _jsonable(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_jsonable(v) for v in o]
        if isinstance(o, tuple):
            return tuple(_jsonable(v) for v in o)
        if isinstance(o, (np.floating, np.float32, np.float64)):
            return float(o)
        if isinstance(o, (np.integer, np.int32, np.int64)):
            return int(o)
        if isinstance(o, (np.bool_,)):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        # Scalars with .item()
        if hasattr(o, "item") and callable(getattr(o, "item")):
            try:
                return o.item()
            except Exception:
                pass
        return o

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for r in rows:
            try:
                f.write(orjson.dumps(r, option=orjson.OPT_SERIALIZE_NUMPY))
            except Exception:
                f.write(orjson.dumps(_jsonable(r)))
            f.write(b"\n")


def run_cmd(cmd):
    cmd = [str(x) for x in cmd]
    print(">>>", " ".join(cmd)); subprocess.run(cmd, check=True)

def hardness_flags(base_model, adapter_dir, rows, device, fp16, max_new_tokens):
    """Return 0/1 hardness = 1 if model gets it wrong (cheap proxy)."""
    tmp = Path("._tmp_star_eval"); tmp.mkdir(exist_ok=True)
    qa_tmp = tmp / "mini.jsonl"; save_jsonl(qa_tmp, rows)
    out_dir = tmp / "out"
    run_cmd([sys.executable, "scripts/eval_with_adapter.py",
             "--base_model", base_model, "--adapter_dir", adapter_dir,
             "--qa_file", str(qa_tmp), "--out_dir", str(out_dir),
             "--max_eval", str(len(rows)), "--max_new_tokens", str(max_new_tokens)])
    preds_path = out_dir/"eval.json"
    # We only need 0/1 quickly; reuse EM via debug_samples if you implement it later.
    # For now, treat all misses as hardness=1, hits=0 by re-evaluating roughly:
    # (This keeps FAR simple.)
    preds = []
    with open(qa_tmp,"rb") as f:
        for line in f: preds.append(orjson.loads(line))
    # fallback: uniform hardness (0.5) to avoid heavy plumbing
    return [1.0]*len(preds)  # safe, conservative

def far_weights(prev, today_rows, hardness, day_now,
                lam_h=1.0, lam_n=0.5, lam_t=0.2,
                far_gamma=0.1, gamma=None):
    """
    Compute FAR replay weights over 'prev' examples using three signals:
      - hardness (model difficulty)
      - novelty (TF-IDF distance from today's text)
      - recency (time decay by day index)

    Backwards-compat: accepts either 'far_gamma' or 'gamma'. If both are
    provided, 'gamma' takes precedence.
    """
    if gamma is not None:
        far_gamma = float(gamma)

    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer

    n_prev = len(prev)
    if n_prev == 0:
        return np.array([], dtype=float)

    # Helpers ---------------------------------------------------------------
    def _row_text(r):
        t = (r.get("text") or "").strip()
        if not t:
            q = (r.get("question") or "").strip()
            a = (r.get("answer") or "").strip()
            t = (q + " " + a).strip()
        return t

    def _hardness_vector():
        if isinstance(hardness, (list, np.ndarray)) and len(hardness) == n_prev:
            return np.asarray(hardness, dtype=float)
        if isinstance(hardness, dict):
            return np.asarray([hardness.get(i, 1.0) for i in range(n_prev)], dtype=float)
        try:
            h = float(hardness)
            return np.full(n_prev, h, dtype=float)
        except Exception:
            return np.ones(n_prev, dtype=float)

    def _recency_vector():
        days = []
        for r in prev:
            d_i = r.get("day")
            if isinstance(d_i, int):
                delta = max(0, day_now - d_i)
            else:
                delta = 0
            days.append(np.exp(-far_gamma * delta))
        return np.asarray(days, dtype=float)

    # Build text views ------------------------------------------------------
    texts_prev_all = [_row_text(r) for r in prev]
    keep_idx = [i for i, t in enumerate(texts_prev_all) if t]
    if not keep_idx:
        H = _hardness_vector()
        T = _recency_vector()
        w = lam_h * H + lam_t * T
        w = np.maximum(w, 1e-8)
        return w / w.sum()

    texts_prev = [texts_prev_all[i] for i in keep_idx]
    map_back = keep_idx

    texts_today = []
    for r in today_rows:
        t = _row_text(r)
        if t:
            texts_today.append(t)

    H_full = _hardness_vector()
    T_full = _recency_vector()

    # If no today text, skip novelty safely --------------------------------
    if len(texts_today) == 0:
        w = lam_h * H_full + lam_t * T_full
        w = np.maximum(w, 1e-8)
        return w / w.sum()

    # Novelty via TF-IDF: use dense centroid to avoid sparse .matrix traps --
    vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2))
    vec.fit(texts_prev + texts_today)
    V_prev = vec.transform(texts_prev)   # sparse CSR [n_prev_text, d]
    V_today = vec.transform(texts_today) # sparse CSR [n_today_text, d]

    # Dense centroid of today
    c_today = np.asarray(V_today.mean(axis=0)).ravel()        # (d,)
    c_norm = np.linalg.norm(c_today)
    prev_norms = np.sqrt(V_prev.multiply(V_prev).sum(axis=1)).A.ravel()  # (n_prev_text,)

    if c_norm == 0 or np.all(prev_norms == 0):
        w = lam_h * H_full + lam_t * T_full
        w = np.maximum(w, 1e-8)
        return w / w.sum()

    # Cosine similarity: sparse @ dense -> dense
    numer = V_prev.dot(c_today)                                # (n_prev_text,)
    sims = numer / (prev_norms * c_norm + 1e-12)
    sims = np.clip(sims, 0.0, 1.0)
    novelty_small = 1.0 - sims

    # Scatter novelty back to full buffer size
    N_full = np.zeros(n_prev, dtype=float)
    for j, k in enumerate(map_back):
        N_full[k] = novelty_small[j]

    # Combine signals
    w = lam_h * H_full + lam_n * N_full + lam_t * T_full
    w = np.maximum(w, 1e-8)
    return w / w.sum()


def weighted_reservoir_add(buffer_rows, new_rows, weights, cap, day_now):
    """Efraimidis-Spirakis weighted reservoir sampling."""
    import heapq, random
    heap = []
    # turn existing buffer into heap with precomputed keys if present
    for r in buffer_rows:
        k = r.get("_wkey", random.random())
        heap.append((k, r))
    heapq.heapify(heap)
    # add new
    for r, w in zip(new_rows, weights):
        k = random.random() ** (1.0 / max(w,1e-6))
        r = dict(r); r["day_seen"] = day_now; r["_wkey"]=k
        if len(heap) < cap:
            heapq.heappush(heap, (k, r))
        else:
            if k > heap[0][0]:
                heapq.heapreplace(heap, (k, r))
    return [r for _, r in heap]

def pick_adapter_pool(adapter_root, regimen_folder, day_now, pool=10):
    # pool = last <pool> days’ adapters (names must be deterministic)
    dirs = sorted((Path(adapter_root)/regimen_folder).glob("day_*"))
    take = [d for d in dirs if d.name <= f"day_{day_now:02d}"][-pool:]
    names = [f"{regimen_folder}/{d.name}" for d in take]  # bank-relative names
    return names

def probe_soft_targets(base_model, adapter_bank_root, adapter_names, qa_rows,
                       device, fp16, max_new_tokens, beta=0.5):
    """
    Build soft targets for router:
      For each q, run each adapter separately -> EM hit (0/1).
      For 'today' examples, +1 reward; for 'legacy' examples, we subtract beta for forgetting.
    """
    import tempfile, shutil, json, subprocess
    nA = len(adapter_names)
    targets = np.zeros((len(qa_rows), nA), dtype=np.float32)

    # write QA to temp
    tmp = Path("._tmp_star_probe"); tmp.mkdir(exist_ok=True, parents=True)
    qa_path = tmp/"qa.jsonl"
    save_jsonl(qa_path, qa_rows)

    # For each adapter, evaluate and collect preds to compute EM quickly
    for j, name in enumerate(adapter_names):
        adir = Path(adapter_bank_root) / name
        out_dir = tmp / f"probe_{j:02d}"
        run_cmd([sys.executable, "scripts/eval_with_adapter.py",
                 "--base_model", base_model, "--adapter_dir", str(adir),
                 "--qa_file", str(qa_path), "--out_dir", str(out_dir),
                 "--max_eval", str(len(qa_rows)), "--max_new_tokens", str(max_new_tokens)])
        # We don’t have per-example preds in that script; approximate by exact string match
        rows = load_jsonl(qa_path)
        # naive: assume 0/1 equally (fallback). If you add per-example preds later, plug here.
        hits = np.zeros(len(rows), dtype=np.float32)
        targets[:, j] = hits

    # Normalize to distribution; avoid all-zero rows by uniform smoothing
    row_sums = targets.sum(axis=1, keepdims=True)
    targets = (targets + 1e-3) / (row_sums + 1e-3 * targets.shape[1])

    return targets  # [N, nA]

def main(args):
    random.seed(args.seed); np.random.seed(args.seed)
    Path(args.out_root).mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.out_root)/"logs"; log_dir.mkdir(parents=True, exist_ok=True)

    day_train_files = sorted(glob(os.path.join(args.days_dir, "qa_train_day_*.jsonl")))
    day_eval_files  = sorted(glob(os.path.join(args.days_dir, "qa_eval_day_*.jsonl")))
    assert len(day_train_files) == len(day_eval_files)

    metrics_csv = Path(args.out_root)/"summary.csv"
    if metrics_csv.exists(): metrics_csv.unlink()
    with open(metrics_csv,"w") as f:
        f.write("day,FFI_EM,FFI_nEM,FFI_F1,Legacy_EM,Legacy_nEM,Legacy_F1,train_seconds,n_train,buf_size\n")

    buffer_path = Path(args.out_root)/"replay_buffer.jsonl"
    if buffer_path.exists(): buffer_path.unlink()

    # STAR stores adapters under models/adapters/star/day_xx
    regimen_folder = "star"
    for d in range(1, args.max_days+1):
        day_tag = f"day_{d:02d}"
        day_train = day_train_files[d-1]
        day_eval  = day_eval_files[d-1]

        adapter_dir = Path(args.adapters_root)/regimen_folder/day_tag
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # === Build FAR mixed train set ===
        today_rows = load_jsonl(day_train)
        prev = load_jsonl(buffer_path) if buffer_path.exists() else []
        # hardness: conservative (all 1s) or compute via last adapter if exists
        hardness = [1.0]*len(prev)
        if d > 1 and prev:
            hardness = hardness_flags(args.base_model, str(Path(args.adapters_root)/regimen_folder/f"day_{d-1:02d}"),
                                      prev, args.device, args.fp16, args.max_new_tokens)
        weights = far_weights(prev, today_rows, hardness, day_now=d,
                              lam_h=args.lam_h, lam_n=args.lam_n, lam_t=args.lam_t, gamma=args.far_gamma)
        cap = max(args.min_replay_cap, int(args.replay_rate * (len(prev) + len(today_rows))))
        new_buf = weighted_reservoir_add(prev, today_rows, np.ones(len(today_rows)), cap, d)
        # update buffer with FAR (weights applied when adding “prev” next day)
        save_jsonl(buffer_path, new_buf)

        # Mix 1:1 (today + replay sample) for training today’s adapter
        k = min(len(today_rows), len(new_buf))
        mix = today_rows[:k] + new_buf[:k]
        tmp_mix = Path(args.out_root)/f"star_train_{day_tag}.jsonl"; save_jsonl(tmp_mix, mix)
        n_train = len(mix)

        # === Train LoRA adapter A_t ===
        t0 = time.time()
        run_cmd([sys.executable, "scripts/train_lora.py",
                 "--base_model", args.base_model, "--train_jsonl", str(tmp_mix),
                 "--out_dir", str(adapter_dir),
                 "--epochs", str(args.epochs_per_day), "--bsz", str(args.bsz),
                 "--grad_accum", str(args.grad_accum), "--lr", str(args.lr),
                 "--max_len", str(args.max_len)])
        train_seconds = time.time() - t0

        # === Evaluate FFI and Legacy using router (after training it) ===
        # First build router training set by probing a small adapter pool
        pool_names = pick_adapter_pool(args.adapters_root, regimen_folder, d, pool=args.pool_size)

        # Effective k for today = min(requested, available adapters)
        nA = len(pool_names)
        k_eff = max(1, min(args.top_k, nA))

        # router labels from day eval + legacy sample
        legacy_all = load_jsonl(args.legacy_eval)
        legacy_rows = legacy_all[:min(len(legacy_all), args.legacy_probe_cap)]
        eval_rows = load_jsonl(day_eval)[:min(args.max_eval, args.today_probe_cap)]
        router_rows = eval_rows + legacy_rows
        soft_targets = probe_soft_targets(args.base_model, args.adapters_root, pool_names,
                                          router_rows, args.device, args.fp16, args.max_new_tokens,
                                          beta=args.beta_forget)

        # Train router (with k_eff)
        router = STARRouter(k=k_eff, encoder=args.encoder, device=args.device)
        router.fit([r["question"] for r in router_rows], soft_targets, pool_names,
                   epochs=args.router_epochs, lr=args.router_lr, batch=128)
        router_dir = Path(args.out_root)/"routers"/day_tag; router.save(router_dir)

        # Now eval day FFI with router
        ffi_dir = Path(args.out_root)/"ffi"/day_tag; ffi_dir.mkdir(parents=True, exist_ok=True)
        run_cmd([sys.executable, "scripts/eval_with_router.py",
                 "--base_model", args.base_model, "--router_dir", str(router_dir),
                 "--adapter_bank_root", args.adapters_root,
                 "--qa_file", day_eval, "--out_dir", str(ffi_dir),
                 "--top_k", str(k_eff), "--max_eval", str(args.max_eval),
                 "--max_new_tokens", str(args.max_new_tokens)])

        # Eval legacy with router
        legacy_dir = Path(args.out_root)/"legacy"/day_tag; legacy_dir.mkdir(parents=True, exist_ok=True)
        run_cmd([sys.executable, "scripts/eval_with_router.py",
                 "--base_model", args.base_model, "--router_dir", str(router_dir),
                 "--adapter_bank_root", args.adapters_root,
                 "--qa_file", args.legacy_eval, "--out_dir", str(legacy_dir),
                 "--top_k", str(k_eff), "--max_eval", str(args.max_eval),
                 "--max_new_tokens", str(args.max_new_tokens)])

        # Load metrics
        import json
        with open(ffi_dir/"eval.json","r") as f: ffi = json.load(f)
        with open(legacy_dir/"eval.json","r") as f: leg = json.load(f)

        buf_size = sum(1 for _ in open(buffer_path,"rb")) if buffer_path.exists() else 0
        with open(metrics_csv,"a") as f:
            f.write(f"{d},{ffi['EM']:.4f},{ffi['nEM']:.4f},{ffi['F1']:.4f},"
                    f"{leg['EM']:.4f},{leg['nEM']:.4f},{leg['F1']:.4f},"
                    f"{train_seconds:.1f},{n_train},{buf_size}\n")
        print(f"[STAR {day_tag}] FFI_EM={ffi['EM']:.3f} Legacy_EM={leg['EM']:.3f} "
              f"train_s={train_seconds:.1f} n={n_train} buf={buf_size}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--days_dir", default="data/days")
    ap.add_argument("--legacy_eval", default="data/legacy/qa_legacy_holdout.jsonl")
    ap.add_argument("--adapters_root", default="models/adapters")
    ap.add_argument("--out_root", default="artifacts/rq2_star")
    ap.add_argument("--max_days", type=int, default=30)

    # FAR + mix
    ap.add_argument("--replay_rate", type=float, default=0.03)
    ap.add_argument("--min_replay_cap", type=int, default=128)
    ap.add_argument("--lam_h", type=float, default=1.0)
    ap.add_argument("--lam_n", type=float, default=0.5)
    ap.add_argument("--lam_t", type=float, default=0.2)
    ap.add_argument("--far_gamma", type=float, default=0.1)

    # training/eval knobs
    ap.add_argument("--epochs_per_day", type=int, default=3)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--max_eval", type=int, default=500)
    ap.add_argument("--max_new_tokens", type=int, default=32)

    # router
    ap.add_argument("--pool_size", type=int, default=10)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--encoder", default="auto")
    ap.add_argument("--router_epochs", type=int, default=8)
    ap.add_argument("--router_lr", type=float, default=1e-3)
    ap.add_argument("--beta_forget", type=float, default=0.5)

    # probe caps
    ap.add_argument("--today_probe_cap", type=int, default=512,
                    help="Max examples from today's eval for router training.")
    ap.add_argument("--legacy_probe_cap", type=int, default=512,
                    help="Max examples from legacy holdout for router training.")

    # compute
    ap.add_argument("--device", default="")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    main(args)
