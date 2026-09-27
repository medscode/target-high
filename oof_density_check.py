"""Validate the per-bucket density threshold adjustment on OOF labels.

    PYTHONPATH=src python oof_density_check.py <exp_id>
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ber import decide, pipeline, prep, metrics  # noqa


def _bucket(s, p):
    """bucket per s1 = count of candidates with 0.1<p<0.9 clipped at 4."""
    df = pd.DataFrame({"s1": s, "n": ((p > 0.1) & (p < 0.9)).astype(int)})
    return df.groupby("s1").n.sum().clip(upper=4).to_dict()


def _score(d, truth_map, t_first, t_other, t_extra):
    mask = decide.select(d, t_first, t_other, t_extra)
    sel = d[mask]
    pred = sel.groupby("s1", sort=False).cand.apply(list).to_dict()
    for s in truth_map:
        pred.setdefault(s, [])
    return metrics.entity_scores(pred, truth_map).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_id")
    args = ap.parse_args()

    cfg_path = f"configs/exp/{args.exp_id}.yaml"
    if not Path(cfg_path).exists():
        cand = list(Path("configs/exp").glob(f"{args.exp_id}_*.yaml"))
        cfg_path = str(cand[0])
    cfg = pipeline.load_config(cfg_path)
    cfg["exp_id"] = args.exp_id
    paths = cfg["paths"]

    oof = pd.read_parquet(Path(paths["results_dir"]) / args.exp_id / "oof.parquet")
    truth = {s: set(g.cand.values) for s, g in oof[oof.y == 1].groupby("s1")}
    ents = pd.read_csv(Path(paths["results_dir"]) / args.exp_id / "entity_scores.csv")
    for s in ents.s1.values:
        truth.setdefault(s, set())

    m = json.loads((Path(paths["results_dir"]) / args.exp_id / "metrics.json").read_text())
    tf0, to0, te0 = m["t_first"], m["t_other"], m["t_extra"]
    print(f"Base thresholds: t_first={tf0}, t_other={to0}, t_extra={te0}", flush=True)

    d = decide.prepare(oof, one_to_one=True)

    # baseline
    base = _score(d, truth, tf0, to0, te0)
    print(f"Baseline macroF05 (OOF) = {base:.5f}", flush=True)

    # bucket per s1
    bm = np.array([_bucket(d.s1.values, d.p.values).get(s, 0) for s in d.s1.values])

    # Search per-bucket multiplier for t_extra
    te_mults = {0: 1.00, 1: 1.00, 2: 0.95, 3: 0.85, 4: 0.70}
    te_eff = te0 * np.array([te_mults[int(b)] for b in bm])
    pa = d.pa.values
    mask = (pa >= tf0) & (pa > 0) & (
        d.is_top.values | (d.is_other_best.values & (pa >= to0)) | (pa >= te_eff)
    )
    sel = d[mask]
    pred = sel.groupby("s1", sort=False).cand.apply(list).to_dict()
    for s in truth:
        pred.setdefault(s, [])
    d_score = metrics.entity_scores(pred, truth).mean()
    print(f"Density-aware t_extra={te_mults} → macroF05={d_score:.5f}  delta={(d_score-base)*100:+.3f}", flush=True)

    # try the other way too: a more aggressive t_extra drop
    for te_mults in [
        {0:1.0,1:1.0,2:0.95,3:0.85,4:0.70},
        {0:1.0,1:0.95,2:0.90,3:0.80,4:0.65},
        {0:1.0,1:0.95,2:0.85,3:0.75,4:0.55},
        {0:1.0,1:1.0,2:1.0,3:0.85,4:0.65},
        {0:1.0,1:1.0,2:0.90,3:0.75,4:0.50},
        {0:1.0,1:0.98,2:0.92,3:0.80,4:0.60},
    ]:
        te_eff = te0 * np.array([te_mults[int(b)] for b in bm])
        mask = (pa >= tf0) & (pa > 0) & (
            d.is_top.values | (d.is_other_best.values & (pa >= to0)) | (pa >= te_eff)
        )
        sel = d[mask]
        pred = sel.groupby("s1", sort=False).cand.apply(list).to_dict()
        for s in truth: pred.setdefault(s, [])
        sc = metrics.entity_scores(pred, truth).mean()
        print(f"  {te_mults} → {sc:.5f} ({(sc-base)*100:+.3f})", flush=True)


if __name__ == "__main__":
    main()
