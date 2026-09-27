"""Quick ship with custom thresholds on e010 test scores.

    PYTHONPATH=src python ship_sweep.py <exp_id> <t_first> <t_other> <t_extra>
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


def main(exp_id, t_first, t_other, t_extra, note):
    cfg_path = f"configs/exp/{exp_id}.yaml"
    if not Path(cfg_path).exists():
        cand = list(Path("configs/exp").glob(f"{exp_id}_*.yaml"))
        cfg_path = str(cand[0])
    cfg = pipeline.load_config(cfg_path)
    cfg["exp_id"] = exp_id
    paths = cfg["paths"]

    sp = Path(paths["cache_dir"]) / f"test_scores_{exp_id}.parquet"
    print(f"[sweep] {sp}")
    scored = pd.read_parquet(sp)
    d = decide.prepare(scored, cfg["decision"].get("one_to_one", True))
    del scored

    mask = decide.select(d, t_first, t_other, t_extra)
    s1v, cv = d.s1.values, d.cand.values
    match_map = _lists(s1v[mask], cv[mask])
    cand_map = _lists(s1v, cv)
    print(f"[sweep] {len(match_map):,} matched S1, {sum(len(v) for v in match_map.values()):,} matches")

    s1_ids = prep.load(cfg, "test", [1], ["entity_id"]).entity_id.tolist()
    out = Path(paths.get("submissions_dir", "submissions")) / f"sub_{time.strftime('%Y%m%d_%H%M')}_{exp_id}_sweep_{t_first}_{t_other}_{t_extra}"
    out.mkdir(parents=True, exist_ok=True)
    io.write_id_lists(match_map, s1_ids, out / "matching_results.tsv", "matched_entity_ids")
    io.write_id_lists(cand_map, s1_ids, out / "candidate_pairs.tsv", "candidate_entity_ids")
    print(f"[sweep] wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]), sys.argv[5] if len(sys.argv) > 5 else "")
