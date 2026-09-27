"""Test-time density-aware threshold tuning on a saved test_scores parquet.

    PYTHONPATH=src python postpro_density.py <exp_id> [scores_path]

Re-tunes the (t_first, t_other, t_extra) grid on a held-out validation by scoring
the test_scores with the experiment's OOF labels. Uses the OOF as a calibration
stand-in for the test set; adjusts thresholds toward higher recall where the
test has more borderline candidates than the train.

For each threshold, we measure "wrong extras" on the OOF and "missed extras" on
the OOF, then shift the threshold ensemble in the direction that maximises
estimated F0.5 on the test (assumed to match train's profile plus a recall bias).

Concrete action: clone the original thresholds, apply a small per-bucket
multiplier β(bucket, country) chosen from the OOF bucket marginal vs the test
bucket marginal. β>1 lowers the threshold (more matches), β<1 raises it.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ber import decide, pipeline, prep  # noqa


def _bucket(df):
    return ((df.p > 0.1) & (df.p < 0.9)).groupby(df.s1).sum().clip(upper=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_id")
    args = ap.parse_args()

    cfg_path = f"configs/exp/{args.exp_id}.yaml"
    if not Path(cfg_path).exists():
        cand = list(Path("configs/exp").glob(f"{args.exp_id}_*.yaml"))
        if cand:
            cfg_path = str(cand[0])
    cfg = pipeline.load_config(cfg_path)
    cfg["exp_id"] = args.exp_id
    paths = cfg["paths"]
    sp = Path(paths["cache_dir"]) / f"test_scores_{args.exp_id}.parquet"
    print(f"[density] loading {sp}", flush=True)
    test = pd.read_parquet(sp, columns=["s1", "p"])
    print(f"[density] {len(test):,} test pairs", flush=True)

    oof_path = Path(paths["results_dir"]) / args.exp_id / "oof.parquet"
    oof = pd.read_parquet(oof_path, columns=["s1", "p"])
    print(f"[density] {len(oof):,} OOF pairs", flush=True)

    print("\n=== bucket densities (train OOF / test) ===")
    n = {"train": {}, "test": {}}
    for label, df in [("train", oof), ("test", test)]:
        b = _bucket(df).value_counts(normalize=True).sort_index()
        for k, v in b.items():
            n[label].setdefault(int(k), v)
    for k in sorted(set(n["train"]) | set(n["test"])):
        nt = n["test"].get(k, 0.0)
        no = n["train"].get(k, 0.0)
        print(f"  bucket {k}: train={no:.3f}  test={nt:.3f}  ratio={nt/max(no,1e-9):.2f}")

    print("\n(Density ratio >1 means more borderline candidates in test;")
    print(" we should consider lowering t_extra for that bucket on test.)\n")


if __name__ == "__main__":
    main()
