"""Fast OOF threshold sweep using macro F0.5 vectorisation (no python loops).

    PYTHONPATH=src python oof_swift_check.py <exp_id>

Uses decide.Scorer with vectorised f05_vec — evaluates one threshold triplet per call.
Caches all entity-level n_pred/n_truth/tp so sweeps take seconds.
"""
import sys, json, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ber import decide, pipeline  # noqa


def _prep(oof, truth_keys):
    """Prepare (n_pred, n_truth, tp) per code-index s1 from OOF (already 0/1 y)."""
    d = decide.prepare(oof, one_to_one=True)
    # build code map for ALL s1 entities (including ones without candidates)
    code = {s: i for i, s in enumerate(truth_keys)}
    y = np.fromiter((c in truth_keys[s] for s, c in zip(d.s1, d.cand)), float, len(d))
    codes = d.s1.map(code).values
    n_truth = np.array([len(truth_keys[s]) for s in truth_keys])
    E = len(truth_keys)
    return d, codes, y, n_truth, E


def _apply_decide(d, codes, E, t_first, t_other, t_extra):
    mask = decide.select(d, t_first, t_other, t_extra)
    c = codes[mask]
    n_pred = np.bincount(c, minlength=E)
    tp = np.bincount(c, weights=(mask & (d.y.values == 1)).astype(float) if hasattr(d, "y") else np.zeros(len(mask))[mask],
                     minlength=E)
    return n_pred, tp


def main():
    exp_id = sys.argv[1]
    cfg_path = f"configs/exp/{exp_id}.yaml"
    if not Path(cfg_path).exists():
        cand = list(Path("configs/exp").glob(f"{exp_id}_*.yaml"))
        cfg_path = str(cand[0])
    cfg = pipeline.load_config(cfg_path)
    cfg["exp_id"] = exp_id
    paths = cfg["paths"]

    oof = pd.read_parquet(Path(paths["results_dir"]) / exp_id / "oof.parquet")
    print(f"oof: {len(oof):,}")
    truth_list = pd.read_csv(Path(paths["results_dir"]) / exp_id / "entity_scores.csv")
    # build truth_map (s1 -> set of cands) using OOF positives
    truth = {s: list(g.cand.values) for s, g in oof[oof.y == 1].groupby("s1")}
    for s in truth_list.s1.values:
        truth.setdefault(s, [])
    # 'truth' is list; convert on the fly
    truth_set = {s: set(v) for s, v in truth.items()}
    truth_keys = list(truth_set.keys())
    print(f"entities: {len(truth_keys):,}")

    d = decide.prepare(oof, one_to_one=True)
    code = {s: i for i, s in enumerate(truth_keys)}
    y = np.fromiter((c in truth_set[s] for s, c in zip(d.s1, d.cand)), float, len(d))
    codes = d.s1.map(code).values
    n_truth = np.array([len(truth_set[s]) for s in truth_keys])
    E = len(truth_keys)

    # vectorised F0.5
    from ber.metrics import f05_vec

    def F(t_first, t_other, t_extra):
        mask = decide.select(d, t_first, t_other, t_extra)
        c = codes[mask]
        n_pred = np.bincount(c, minlength=E)
        # tp = sum of y on selected rows (y is already 0/1)
        sel_y = y[mask]
        tp = np.bincount(c, weights=sel_y, minlength=E)
        f = f05_vec(n_pred, n_truth, tp)
        return float(f.mean())

    # report several thresholds
    cases = [
        (0.5, 0.9, 0.7),
        (0.5, 0.8, 0.75),
        (0.45, 0.8, 0.7),
        (0.4, 0.7, 0.6),
        (0.45, 0.7, 0.55),
        (0.5, 0.7, 0.55),
        (0.5, 0.7, 0.5),
        (0.5, 0.6, 0.5),
        (0.45, 0.6, 0.45),
        (0.4, 0.5, 0.4),
    ]
    base = F(*cases[0])
    print(f"baseline (0.5, 0.9, 0.7)  F0.5 = {base:.5f}")
    for tf, to, te in cases[1:]:
        v = F(tf, to, te)
        print(f"  ({tf:.2f}, {to:.2f}, {te:.2f})  F0.5 = {v:.5f}  delta={(v-base)*100:+.3f}")


if __name__ == "__main__":
    main()
