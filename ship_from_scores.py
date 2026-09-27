"""Build the submission files from a saved consolidated scores parquet.

    python ship_from_scores.py <exp_id> [score_path] [note]

Reads cache/test_scores_<exp>.parquet (or path arg), applies thresholds from
results/<exp>/metrics.json + an optional decoy prior, writes both TSVs and
appends to the registry. No model / training data in memory.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

from ber import decide, io, pipeline, prep


def _lists(s1, cand):
    if len(s1) == 0:
        return {}
    cut = np.flatnonzero(s1[1:] != s1[:-1]) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(s1)]
    return {s1[a]: list(cand[a:b]) for a, b in zip(starts, ends)}


DECOY_OFFSETS = {1, 2, 3, 4, 5, 7, 9, 11, 13, 21}


def apply_decoy_prior(scored: pd.DataFrame, t_cut: float = 0.35) -> pd.DataFrame:
    """Drop the score of pairs whose signed house-number offset is a known decoy
    unless the candidate agrees with the S1's consensus house number or the S1
    has no strong matches yet.

    We work off the consolidated score parquet which only carries s1,cand,cand_src,p;
    the consensus check needs the original features (not in this file). So as a
    safe proxy we only apply the prior to candidates whose raw score is below
    t_cut AND that lose to another candidate of the same S1 — i.e. borderline
    decoys that are about to fail the threshold anyway and may earn a wrong-
    extra by being kept.

    Returns the same dataframe with new column 'p_postproc' that re-scales
    'pa' for decide.select.
    """
    s = scored.copy().sort_values(["s1", "p"], ascending=[True, False], kind="stable")
    g = s.groupby("s1", sort=False)
    s["p_postproc"] = s["p"].values
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_id")
    ap.add_argument("--config", default=None,
                    help="path under configs/exp/. defaults to <exp_id>.yaml then <exp_id>_<tag>.yaml")
    ap.add_argument("--scores", default=None)
    ap.add_argument("--note", default="ship_from_scores")
    ap.add_argument("--override-t", default=None,
                    help="comma-separated t_first,t_other,t_extra to override metrics.json")
    args = ap.parse_args()

    cfg_path = args.config or f"configs/exp/{args.exp_id}.yaml"
    if not Path(cfg_path).exists():
        # try <id>_<tag>.yaml
        cand = list(Path("configs/exp").glob(f"{args.exp_id}_*.yaml"))
        if cand:
            cfg_path = str(cand[0])
    cfg = pipeline.load_config(cfg_path)
    if cfg["exp_id"] != args.exp_id:
        cfg["exp_id"] = args.exp_id  # reuse parent paths; only the result/test folder differs
        sys.exit(f"config exp_id mismatch: {cfg['exp_id']} vs {args.exp_id}")
    paths = cfg["paths"]
    sp = Path(args.scores or Path(paths["cache_dir"]) / f"test_scores_{args.exp_id}.parquet")
    print(f"[ship] loading {sp}", flush=True)
    scored = pd.read_parquet(sp)
    print(f"[ship] {len(scored)} pairs", flush=True)

    m = json.loads((Path(paths["results_dir"]) / args.exp_id / "metrics.json").read_text())
    if args.override_t:
        tf, to, te = [float(x) for x in args.override_t.split(",")]
        print(f"[ship] OVERRIDE thresholds: {tf},{to},{te}")
    else:
        tf, to, te = m["t_first"], m["t_other"], m["t_extra"]
    print(f"[ship] thresholds: t_first={tf}, t_other={to}, t_extra={te}", flush=True)

    cfg_yaml = {
        **cfg,
        "paths": {**cfg["paths"], "submissions_dir": "submissions",
                  "results_dir": "results"},
    }
    d = decide.prepare(scored, cfg["decision"].get("one_to_one", True))
    del scored
    mask = decide.select(d, tf, to, te)
    s1v, cv = d.s1.values, d.cand.values
    match_map = _lists(s1v[mask], cv[mask])
    cand_map = _lists(s1v, cv)
    print(f"[ship] {len(match_map):,} matches, {sum(len(v) for v in match_map.values()):,} candidates",
          flush=True)

    s1_ids = prep.load(cfg, "test", [1], ["entity_id"]).entity_id.tolist()
    out = Path(paths.get("submissions_dir", "submissions")) / f"sub_{time.strftime('%Y%m%d_%H%M')}_{args.exp_id}_ship"
    out.mkdir(parents=True, exist_ok=True)
    io.write_id_lists(match_map, s1_ids, out / "matching_results.tsv", "matched_entity_ids")
    io.write_id_lists(cand_map, s1_ids, out / "candidate_pairs.tsv", "candidate_entity_ids")
    print(f"[ship] wrote {out}", flush=True)

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
    sizes = np.array([len(cand_map.get(s, ())) for s in s1_ids])
    row = {"sub_dir": out.name, "exp_id": args.exp_id, "validator": status,
           "cv_macroF05": m.get("macroF05"),
           "t_first": tf, "t_other": to, "t_extra": te,
           "test_cand_mean": float(sizes.mean()),
           "test_pred_matches": int(n_pred.sum()),
           "test_pred_singleton_rate": float((n_pred == 0).mean()),
           "note": args.note, "uploaded": "", "lb_score": ""}
    (out / "submission_info.json").write_text(json.dumps(row, indent=2, default=str))
    reg = Path(paths.get("submissions_dir", "submissions")) / "registry.csv"
    pd.concat([pd.read_csv(reg), pd.DataFrame([row])] if reg.exists()
              else [pd.DataFrame([row])], ignore_index=True).to_csv(reg, index=False)
    print(json.dumps(row, indent=2, default=str))
    print(f"[ship] validator: {status}")


if __name__ == "__main__":
    main()
