"""Build a submission using per-(country, bucket) thresholds on a saved scores parquet.

    PYTHONPATH=src python ship_density.py <exp_id> [scores_path]

Decisions:
- bucket = count of candidates with 0.1 < p < 0.9, clipped at 4 (test-like).
- Per-country base thresholds from results/<exp>/metrics.json.
- For high-bucket S1s (more borderline cands), lower t_extra proportionally
  to density ratio, and raise t_other slightly to keep one-to-one sanity.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ber import decide, pipeline, prep, io  # noqa


def _bucket(p_per_s1: np.ndarray) -> np.ndarray:
    return np.clip(((p_per_s1 > 0.1) & (p_per_s1 < 0.9)).astype(int).cumsum(), 0, 4)


def _per_s1_bucket(scored: pd.DataFrame) -> dict:
    """bucket index per s1 (the max bucket observed = 'how borderline is this s1')."""
    grp = scored.groupby("s1", sort=False)
    n_border = ((scored.p.values > 0.1) & (scored.p.values < 0.9)).astype(int)
    df = pd.DataFrame({"s1": scored.s1.values, "n": n_border})
    n = df.groupby("s1").n.sum().clip(upper=4)
    return n.to_dict()


def _lists(s1, cand):
    if len(s1) == 0:
        return {}
    cut = np.flatnonzero(s1[1:] != s1[:-1]) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(s1)]
    return {s1[a]: list(cand[a:b]) for a, b in zip(starts, ends)}


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
    print(f"[ship-density] loading {sp}", flush=True)
    scored = pd.read_parquet(sp)
    print(f"[ship-density] {len(scored):,} pairs", flush=True)

    m = json.loads((Path(paths["results_dir"]) / args.exp_id / "metrics.json").read_text())
    tf0, to0, te0 = m["t_first"], m["t_other"], m["t_extra"]
    print(f"[ship-density] base thresholds: t_first={tf0}, t_other={to0}, t_extra={te0}", flush=True)

    s1_ids = prep.load(cfg, "test", [1], ["entity_id"]).entity_id.tolist()
    bc = prep.load(cfg, "test", [1], ["entity_id", "bc"]).set_index("entity_id").bc.reindex(s1_ids).values

    # bucket per S1 (we only need s1, p columns)
    n_border = ((scored.p.values > 0.1) & (scored.p.values < 0.9)).astype(int)
    bucket_per_s1 = pd.Series(n_border).groupby(scored.s1.values).sum().clip(upper=4)
    bucket_map = dict(zip(scored.s1.unique(), bucket_per_s1.values))
    print(f"[ship-density] bucket distribution: {pd.Series(bucket_map).value_counts().sort_index().to_dict()}", flush=True)

    # per-bucket threshold adjustments
    # bucket 0/1: keep base. bucket 2: t_extra *= 0.85. bucket 3: t_extra *= 0.70. bucket 4: t_extra *= 0.55.
    # Apply for both US and IN. France too (we don't have separate test scores, but
    # apply the same logic to be consistent — bucket count is country-agnostic).
    te_adj = {0: 1.00, 1: 1.00, 2: 0.85, 3: 0.70, 4: 0.55}
    to_adj = {0: 1.00, 1: 1.00, 2: 1.00, 3: 0.92, 4: 0.85}
    tf_adj = {0: 1.00, 1: 1.00, 2: 1.00, 3: 0.95, 4: 0.92}

    d = decide.prepare(scored, cfg["decision"].get("one_to_one", True))
    del scored

    # apply per-s1 thresholds
    pa = d.pa.values
    bm = np.array([bucket_map.get(s, 0) for s in d.s1.values])
    te_eff = te0 * np.array([te_adj[int(b)] for b in bm])
    tf_eff = tf0 * np.array([tf_adj[int(b)] for b in bm])
    to_eff = to0 * np.array([to_adj[int(b)] for b in bm])

    mask = (pa >= tf_eff) & (pa > 0) & (
        d.is_top.values | (d.is_other_best.values & (pa >= to_eff)) | (pa >= te_eff)
    )
    print(f"[ship-density] matches: {int(mask.sum()):,}", flush=True)

    s1v, cv = d.s1.values, d.cand.values
    match_map = _lists(s1v[mask], cv[mask])
    cand_map = _lists(s1v, cv)
    print(f"[ship-density] {len(match_map):,} matched S1, {sum(len(v) for v in cand_map.values()):,} candidates", flush=True)

    out = Path(paths.get("submissions_dir", "submissions")) / f"sub_{time.strftime('%Y%m%d_%H%M')}_{args.exp_id}_density"
    out.mkdir(parents=True, exist_ok=True)
    io.write_id_lists(match_map, s1_ids, out / "matching_results.tsv", "matched_entity_ids")
    io.write_id_lists(cand_map, s1_ids, out / "candidate_pairs.tsv", "candidate_entity_ids")
    print(f"[ship-density] wrote {out}", flush=True)

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
           "t_first_base": tf0, "t_other_base": to0, "t_extra_base": te0,
           "bucket_adj": str(te_adj),
           "test_pred_matches": int(n_pred.sum()),
           "test_pred_singleton_rate": float((n_pred == 0).mean()),
           "note": "density-aware thresholds (per-bucket)", "uploaded": "", "lb_score": ""}
    (out / "submission_info.json").write_text(json.dumps(row, indent=2, default=str))
    reg = Path(paths.get("submissions_dir", "submissions")) / "registry.csv"
    pd.concat([pd.read_csv(reg), pd.DataFrame([row])] if reg.exists()
              else [pd.DataFrame([row])], ignore_index=True).to_csv(reg, index=False)
    print(json.dumps(row, indent=2, default=str))
    print(f"[ship-density] validator: {status}")


if __name__ == "__main__":
    main()
