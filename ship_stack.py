"""Stacked final submission: density + decoy-damp on the e010 test scores.

    PYTHONPATH=src python ship_stack.py <exp_id>

Combines:
  (1) per-bucket t_extra lowering (density reweight)
  (2) score-damp on borderline other-source pairs (decoy prior proxy)
"""
import sys, json, subprocess, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ber import decide, pipeline, prep, io  # noqa


def _lists(s1, cand):
    if len(s1) == 0:
        return {}
    cut = np.flatnonzero(s1[1:] != s1[:-1]) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(s1)]
    return {s1[a]: list(cand[a:b]) for a, b in zip(starts, ends)}


def main(exp_id, note):
    cfg_path = f"configs/exp/{exp_id}.yaml"
    if not Path(cfg_path).exists():
        cand = list(Path("configs/exp").glob(f"{exp_id}_*.yaml"))
        cfg_path = str(cand[0])
    cfg = pipeline.load_config(cfg_path)
    cfg["exp_id"] = exp_id
    paths = cfg["paths"]

    sp = Path(paths["cache_dir"]) / f"test_scores_{exp_id}.parquet"
    print(f"[stack] {sp}")
    scored = pd.read_parquet(sp)
    print(f"[stack] {len(scored):,}")

    m = json.loads((Path(paths["results_dir"]) / exp_id / "metrics.json").read_text())
    tf0, to0, te0 = m["t_first"], m["t_other"], m["t_extra"]

    # bucket per s1
    n_border = ((scored.p.values > 0.1) & (scored.p.values < 0.9)).astype(int)
    bucket_per_s1 = pd.Series(n_border).groupby(scored.s1.values).sum().clip(upper=4)
    bucket_map = dict(zip(scored.s1.unique(), bucket_per_s1.values))
    print(f"[stack] bucket dist: {pd.Series(bucket_map).value_counts().sort_index().to_dict()}")

    d = decide.prepare(scored, cfg["decision"].get("one_to_one", True))
    del scored

    pa = d.pa.values.copy()
    src = d.cand_src.values
    top_src = pd.Series(src).groupby(d.s1.values).transform("first").values

    # (1) density lowering
    te_adj = {0: 1.00, 1: 1.00, 2: 0.90, 3: 0.78, 4: 0.62}
    to_adj = {0: 1.00, 1: 1.00, 2: 1.00, 3: 0.92, 4: 0.85}
    tf_adj = {0: 1.00, 1: 1.00, 2: 1.00, 3: 0.95, 4: 0.92}
    bm = np.array([bucket_map.get(s, 0) for s in d.s1.values])
    te_eff = te0 * np.array([te_adj[int(b)] for b in bm])
    tf_eff = tf0 * np.array([tf_adj[int(b)] for b in bm])
    to_eff = to0 * np.array([to_adj[int(b)] for b in bm])
    print(f"[stack] density: adj by bucket")
    print(f"  bucket counts: 0={(bm==0).sum()/1e6:.1f}M, 1={(bm==1).sum()/1e6:.1f}M, 2={(bm==2).sum()/1e6:.1f}M, 3={(bm==3).sum()/1e6:.1f}M, 4={(bm==4).sum()/1e3:.0f}K")

    # (2) decoy-damp: borderline other-source → multiply pa by 0.78
    is_border = (pa >= 0.40) & (pa <= 0.85)
    is_other = src != top_src
    damp = is_border & is_other
    f_damp = 0.78
    pa[damp] = pa[damp] * f_damp
    print(f"[stack] decoy-damp: {int(damp.sum()):,} pairs damped ({damp.mean()*100:.2f}%)")

    mask = (pa >= tf_eff) & (pa > 0) & (
        d.is_top.values | (d.is_other_best.values & (pa >= to_eff)) | (pa >= te_eff)
    )

    s1v, cv = d.s1.values, d.cand.values
    match_map = _lists(s1v[mask], cv[mask])
    cand_map = _lists(s1v, cv)
    print(f"[stack] {len(match_map):,} matched S1, {sum(len(v) for v in match_map.values()):,} candidates")

    s1_ids = prep.load(cfg, "test", [1], ["entity_id"]).entity_id.tolist()
    out = Path(paths.get("submissions_dir", "submissions")) / f"sub_{time.strftime('%Y%m%d_%H%M')}_{exp_id}_stack"
    out.mkdir(parents=True, exist_ok=True)
    io.write_id_lists(match_map, s1_ids, out / "matching_results.tsv", "matched_entity_ids")
    io.write_id_lists(cand_map, s1_ids, out / "candidate_pairs.tsv", "candidate_entity_ids")
    print(f"[stack] wrote {out}")

    validator = Path(paths.get("validator", ""))
    status = "validator_not_found"
    if validator.is_file():
        r = subprocess.run([sys.executable, str(validator),
                            "--matching", str(out / "matching_results.tsv"),
                            "--candidate", str(out / "candidate_pairs.tsv"),
                            "--test-dir", str(Path(paths["data_dir"]) / "test")],
                           capture_output=True, text=True)
        print(r.stdout[-600:])
        status = "PASS" if r.returncode == 0 else "FAIL"

    n_pred = np.array([len(match_map.get(s, ())) for s in s1_ids])
    row = {"sub_dir": out.name, "exp_id": exp_id, "validator": status,
           "cv_macroF05": m.get("macroF05"),
           "t_first": float(tf0), "t_other": float(to0), "t_extra": float(te0),
           "decoy_factor": f_damp, "damped_count": int(damp.sum()),
           "density_bucket_adj": str(te_adj),
           "test_pred_matches": int(n_pred.sum()),
           "test_pred_singleton_rate": float((n_pred == 0).mean()),
           "note": note, "uploaded": "", "lb_score": ""}
    (out / "submission_info.json").write_text(json.dumps(row, indent=2, default=str))
    reg = Path(paths.get("submissions_dir", "submissions")) / "registry.csv"
    pd.concat([pd.read_csv(reg), pd.DataFrame([row])] if reg.exists()
              else [pd.DataFrame([row])], ignore_index=True).to_csv(reg, index=False)
    print(json.dumps(row, indent=2, default=str))
    print(f"[stack] validator: {status}")
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
