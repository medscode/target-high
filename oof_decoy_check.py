"""Evaluate the decoy-house-number prior on saved train features + e010 OOF.

    PYTHONPATH=src python oof_decoy_check.py

The prior flags pairs whose num_signed_diff ∈ {1,2,3,4,5,7,9,11,13,21} (a known
decoy house-number offset set). We learn what fraction are TRUE matches
according to OOF labels, then estimate the gain from suppressing them.
"""
import sys, json
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ber import decide, pipeline, prep, metrics  # noqa


DECOY = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}


def main():
    train_dir = "/Users/ravipandey/Dev/Personal/target-high/cache/train_9161417760a4/"
    feats = pd.read_parquet(f"{train_dir}feats.parquet", columns=[
        "s1", "cand", "cand_src", "num_signed_diff", "b_num1", "a_num1",
    ])
    print(f"feats: {len(feats):,}")

    # ----- frequency analysis: among pairs with num_signed_diff in DECOY set,
    # what's the positive rate (y=1)?
    mask = feats.num_signed_diff.isin(DECOY)
    print(f"pairs with num_signed_diff ∈ DECOY: {int(mask.sum()):,}  ({mask.mean()*100:.2f}%)")
    feats_pos = feats[mask]
    print(f"  positive rate at num_signed_diff ≈ decoy: {len(feats_pos):,}")

    # ----- bucket by score (we don't have OOF p in feats)
    # We need to load oof to see if these decoy-flagged pairs are positive.
    oof = pd.read_parquet('/Users/ravipandey/Dev/Personal/target-high/results/e010/oof.parquet',
                          columns=["s1", "cand", "y"])
    print(f"oof positives: {(oof.y==1).sum():,}")

    # join
    j = feats_pos.merge(oof, on=["s1", "cand"], how="left")
    j["y"] = j.y.fillna(0).astype(int)
    print(f"decoy-flagged pairs joined: {len(j):,}, positives: {(j.y==1).sum():,}")
    print(f"  positive rate among decoy-flagged: {(j.y==1).mean():.4f}")

    # What about the population baseline positive rate?
    print(f"Baseline positive rate: {(oof.y==1).mean():.4f}")
    print(f"=> decoy-flagged pairs are {(j.y==1).mean() / max(1e-9, (oof.y==1).mean()):.2f}x baseline "
          f"({'lower = suppress' if (j.y==1).mean() < (oof.y==1).mean() else 'higher = keep'})")


if __name__ == "__main__":
    main()
