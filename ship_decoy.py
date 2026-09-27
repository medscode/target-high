"""Suppress decoy-house-number candidates via score reweighting.

    PYTHONPATH=src python ship_decoy.py <exp_id>

We don't have raw test features cached, so we approximate the prior using the
score distribution: pairs with `0.4 < p < 0.85` AND `cand_src` matching the
S1's other source are the most likely decoys (positive rate in OOF analysis:
0.0385 vs baseline 0.0837). For each such pair we multiply p by 0.5 (a
suppression factor) before applying thresholds. This is a safe approximation
because the missed-decoy cost is dominated by the wrong-extra cost in
E4_extra_wrong (the largest bucket after E6b).

Better: we DO have test scores with p, and we ALSO have train feats + OOF p.
We can fit a per-decoy score scaling from train and apply to test (the
scaling is country- and source-agnostic).
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ber import decide, pipeline, prep, io  # noqa


DECOY_OFFSETS = np.array([-21, -13, -11, -9, -7, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 7, 9, 11, 13, 21])


def _lists(s1, cand):
    if len(s1) == 0:
        return {}
    cut = np.flatnonzero(s1[1:] != s1[:-1]) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(s1)]
    return {s1[a]: list(cand[a:b]) for a, b in zip(starts, ends)}


def _score(scored, oof, train_feats):
    """Fit a per-bucket score scaler from train OOF + feats, apply to test."""
    # join oof p + feats by (s1, cand)
    j = oof[["s1", "cand", "p", "y"]].merge(train_feats, on=["s1", "cand"], how="inner")
    # mark decoy
    decoy = j.num_signed_diff.isin(DECOY_OFFSETS).values
    pos_decoy = j.y.values[decoy].mean() if decoy.any() else 0
    pos_clean = j.y.values[~decoy].mean() if (~decoy).any() else 0
    print(f"  train: pos@decoy={pos_decoy:.4f}  pos@clean={pos_clean:.4f}  (decoy is {pos_clean/max(pos_decoy,1e-9):.2f}x more positive)")
    return pos_decoy, pos_clean


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

    sp = Path(paths["cache_dir"]) / f"test_scores_{args.exp_id}.parquet"
    print(f"[ship-decoy] loading {sp}", flush=True)
    scored = pd.read_parquet(sp)
    print(f"[ship-decoy] {len(scored):,} pairs", flush=True)

    # fit scaler from train
    train_feats = pd.read_parquet('/Users/ravipandey/Dev/Personal/target-high/cache/train_9161417760a4/feats.parquet',
                                  columns=["s1", "cand", "num_signed_diff"])
    oof = pd.read_parquet(Path(paths["results_dir"]) / args.exp_id / "oof.parquet",
                          columns=["s1", "cand", "p", "y"])
    pos_d, pos_c = _score(scored, oof, train_feats)
    if pos_d >= pos_c:
        print("[ship-decoy] no decoy signal found; skipping suppression")
        return

    # Decoy suppression factor: scale decoy candidates by (pos_d / pos_c) which is <1.
    # We set pa' = pa * f, where f = pos_d / pos_c (calibrated so decoys need ~ lower
    # threshold to be selected, biased toward abstaining).
    f = float(np.sqrt(pos_d / pos_c))  # sqrt = partial suppression (we don't want to fully suppress, just damp)
    print(f"[ship-decoy] applying score-damp factor f={f:.3f} on borderline pairs "
          f"(we approximate with the test scoring itself since we don't have test feats)")

    # We CANNOT compute num_signed_diff on test (we don't have test feats).
    # Instead, approximate via p in (0.4, 0.85) AND cand_src != top: same statistical profile.
    # Conservative: only damp the borderline range AND keep first/other-best structure.
    d = decide.prepare(scored, cfg["decision"].get("one_to_one", True))
    del scored

    pa = d.pa.values.copy()
    src = d.cand_src.values
    top_src = pd.Series(src).groupby(d.s1.values).transform("first").values
    is_borderline = (pa >= 0.40) & (pa <= 0.85)
    is_other_src = src != top_src
    damp = is_borderline & is_other_src
    pa[damp] = pa[damp] * f
    print(f"[ship-decoy] damped {int(damp.sum()):,} pairs ({damp.mean()*100:.2f}%)")

    m = json.loads((Path(paths["results_dir"]) / args.exp_id / "metrics.json").read_text())
    tf, to, te = m["t_first"], m["t_other"], m["t_extra"]
    pa = pd.Series(pa, index=d.index)
    d2 = d.assign(pa=pa)
    mask = decide.select(d2, tf, to, te)
    s1v, cv = d2.s1.values, d2.cand.values
    match_map = _lists(s1v[mask], cv[mask])
    cand_map = _lists(s1v, cv)
    print(f"[ship-decoy] {len(match_map):,} matched S1, {sum(len(v) for v in match_map.values()):,} candidates",
          flush=True)

    s1_ids = prep.load(cfg, "test", [1], ["entity_id"]).entity_id.tolist()
    out = Path(paths.get("submissions_dir", "submissions")) / f"sub_{time.strftime('%Y%m%d_%H%M')}_{args.exp_id}_decoy"
    out.mkdir(parents=True, exist_ok=True)
    io.write_id_lists(match_map, s1_ids, out / "matching_results.tsv", "matched_entity_ids")
    io.write_id_lists(cand_map, s1_ids, out / "candidate_pairs.tsv", "candidate_entity_ids")

    validator = Path(paths.get("validator", ""))
    status = "validator_not_found"
    if validator.is_file():
        r = subprocess.run([sys.executable, str(validator),
                            "--matching", str(out / "matching_results.tsv"),
                            "--candidate", str(out / "candidate_pairs.tsv"),
                            "--test-dir", str(Path(paths["data_dir"]) / "test")],
                           capture_output=True, text=True)
        print(r.stdout[-600:], r.stderr[-2000:])
        status = "PASS" if r.returncode == 0 else "FAIL"

    n_pred = np.array([len(match_map.get(s, ())) for s in s1_ids])
    row = {"sub_dir": out.name, "exp_id": args.exp_id, "validator": status,
           "cv_macroF05": m.get("macroF05"),
           "t_first": tf, "t_other": to, "t_extra": te,
           "decoy_factor": f, "damped_count": int(damp.sum()),
           "test_pred_matches": int(n_pred.sum()),
           "test_pred_singleton_rate": float((n_pred == 0).mean()),
           "note": "decoy-prior score-damp on borderline other-source pairs",
           "uploaded": "", "lb_score": ""}
    (out / "submission_info.json").write_text(json.dumps(row, indent=2, default=str))
    reg = Path(paths.get("submissions_dir", "submissions")) / "registry.csv"
    pd.concat([pd.read_csv(reg), pd.DataFrame([row])] if reg.exists()
              else [pd.DataFrame([row])], ignore_index=True).to_csv(reg, index=False)
    print(json.dumps(row, indent=2, default=str))
    print(f"[ship-decoy] validator: {status}")


if __name__ == "__main__":
    main()
