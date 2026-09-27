"""Most aggressive post-pro sweep on e010 scores.

    PYTHONPATH=src python ship_aggressive.py e010

Tries multiple (t_extra, density_adj, decoy_factor) combinations.
"""
import sys, json, subprocess, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ber import decide, pipeline, prep, io, metrics  # noqa


def _lists(s1, cand):
    if len(s1) == 0:
        return {}
    cut = np.flatnonzero(s1[1:] != s1[:-1]) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(s1)]
    return {s1[a]: list(cand[a:b]) for a, b in zip(starts, ends)}


def _eval_oof(oof, truth_keys, t_first, t_other, t_extra, te_adj, to_adj, tf_adj, f_damp):
    """Evaluate on OOF using the same logic as test ship."""
    d = decide.prepare(oof, one_to_one=True)
    n_border = ((oof.p.values > 0.1) & (oof.p.values < 0.9)).astype(int)
    bucket_per_s1 = pd.Series(n_border).groupby(oof.s1.values).sum().clip(upper=4)
    bucket_map = dict(zip(oof.s1.unique(), bucket_per_s1.values))
    bm = np.array([bucket_map.get(s, 0) for s in d.s1.values])
    te_eff = t_extra * np.array([te_adj[int(b)] for b in bm])
    tf_eff = t_first * np.array([tf_adj[int(b)] for b in bm])
    to_eff = t_other * np.array([to_adj[int(b)] for b in bm])
    pa = d.pa.values.copy()
    src = d.cand_src.values
    top_src = pd.Series(src).groupby(d.s1.values).transform("first").values
    is_border = (pa >= 0.40) & (pa <= 0.85)
    is_other = src != top_src
    damp = is_border & is_other
    pa[damp] = pa[damp] * f_damp
    truth_set = {s: set(v) for s, v in truth_keys.items()}
    code = {s: i for i, s in enumerate(truth_set)}
    codes = d.s1.map(code).values.astype(np.int64)
    mask = (pa >= tf_eff) & (pa > 0) & (
        d.is_top.values | (d.is_other_best.values & (pa >= to_eff)) | (pa >= te_eff)
    )
    n_pred = np.bincount(codes[mask], minlength=len(truth_set))
    sel_y = np.array([1.0 if d.cand.values[i] in truth_set.get(d.s1.values[i], ()) else 0.0 for i in np.where(mask)[0]])
    tp = np.bincount(codes[mask].astype(np.int64), weights=sel_y, minlength=len(truth_set))
    n_truth = np.array([len(truth_set[s]) for s in truth_set])
    return float(metrics.f05_vec(n_pred, n_truth, tp).mean())


def main(exp_id):
    cfg_path = f"configs/exp/{exp_id}.yaml"
    if not Path(cfg_path).exists():
        cand = list(Path("configs/exp").glob(f"{exp_id}_*.yaml"))
        cfg_path = str(cand[0])
    cfg = pipeline.load_config(cfg_path)
    cfg["exp_id"] = exp_id
    paths = cfg["paths"]
    m = json.loads((Path(paths["results_dir"]) / exp_id / "metrics.json").read_text())
    tf0, to0, te0 = m["t_first"], m["t_other"], m["t_extra"]

    oof = pd.read_parquet(Path(paths["results_dir"]) / exp_id / "oof.parquet")
    truth = {s: list(g.cand.values) for s, g in oof[oof.y == 1].groupby("s1")}
    print(f"oof: {len(oof):,}, truth keys: {len(truth):,}")

    grid = []
    for te_mult in [1.00, 0.95, 0.90, 0.85, 0.80]:
        for d_curve in [
            {0:1,1:1,2:1,3:1,4:1},
            {0:1,1:1,2:0.95,3:0.85,4:0.7},
            {0:1,1:1,2:0.90,3:0.78,4:0.62},
            {0:1,1:0.98,2:0.85,3:0.70,4:0.55},
        ]:
            for damp in [1.00, 0.85, 0.78, 0.70]:
                grid.append({
                    "te_mult": te_mult,
                    "te_adj": d_curve,
                    "to_adj": {0:1,1:1,2:1,3:0.92,4:0.85},
                    "tf_adj": {0:1,1:1,2:1,3:0.95,4:0.92},
                    "f_damp": damp,
                })
    print(f"grid: {len(grid)} configs")

    rows = []
    for i, g in enumerate(grid):
        te_adj = g["te_adj"]; te_eff_base = te0 * g["te_mult"]
        # scale bucket adj by te_mult
        te_adj_eff = {b: te_adj[b] * g["te_mult"] for b in te_adj}
        f = _eval_oof(oof, truth, tf0, to0, te0,
                      te_adj_eff, g["to_adj"], g["tf_adj"], g["f_damp"])
        rows.append((f, g))

    rows.sort(key=lambda x: -x[0])
    print(f"\nTop 8 OOF F0.5:")
    for f, g in rows[:8]:
        print(f"  {f:.5f}  te_mult={g['te_mult']}  f_damp={g['f_damp']}  curve={g['te_adj']}")
    best_f, best_g = rows[0]
    print(f"\nBest: {best_f:.5f}  delta vs base 0.9715 = {(best_f - 0.9715)*100:+.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
