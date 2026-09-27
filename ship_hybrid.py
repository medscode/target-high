"""Hybrid submission: e011 chunks (where scored) + e010 scores for the rest.

    PYTHONPATH=src python ship_hybrid.py
"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ber import decide, pipeline, prep, io


def _lists(s1, cand):
    if len(s1) == 0:
        return {}
    cut = np.flatnonzero(s1[1:] != s1[:-1]) + 1
    starts = np.r_[0, cut]
    ends = np.r_[cut, len(s1)]
    return {s1[a]: list(cand[a:b]) for a, b in zip(starts, ends)}


def main():
    paths = pipeline.load_config("configs/exp/e010.yaml" if Path("configs/exp/e010.yaml").exists()
                                  else list(Path("configs/exp").glob("e010_*.yaml"))[0])["paths"]
    # load e011 chunks (the strong model, partial)
    e011_chunks = sorted(Path("/Users/ravipandey/Dev/Personal/target-high/cache/test_chunks_e011_31146d55ef7b").glob("*.parquet"))
    print(f"[hybrid] e011 chunks: {len(e011_chunks)}")
    e011_df = pd.concat([pd.read_parquet(p) for p in e011_chunks], ignore_index=True)
    e011_s1 = set(e011_df.s1.unique())
    print(f"[hybrid] e011 s1 covered: {len(e011_s1):,}")

    # load e010 scores for the rest
    e010 = pd.read_parquet(Path(paths["cache_dir"]) / "test_scores_e010.parquet")
    print(f"[hybrid] e010: {len(e010):,} pairs")
    e010_missing = e010[~e010.s1.isin(e011_s1)].copy()
    print(f"[hybrid] e010 missing: {len(e010_missing):,} pairs")

    # combine
    e011_keep = e011_df[["s1", "cand", "cand_src", "p"]].copy()
    e010_keep = e010_missing[["s1", "cand", "cand_src", "p"]].copy()
    combined = pd.concat([e011_keep, e010_keep], ignore_index=True)
    print(f"[hybrid] combined: {len(combined):,} pairs, {combined.s1.nunique():,} s1")

    # apply e011 thresholds (better OOF than e010)
    m = json.loads((Path(paths["results_dir"]) / "e011_fast" / "metrics.json").read_text())
    tf, to, te = m["t_first"], m["t_other"], m["t_extra"]
    print(f"[hybrid] thresholds: tf={tf}, to={to}, te={te}")

    d = decide.prepare(combined, one_to_one=True)
    del combined
    mask = decide.select(d, tf, to, te)
    s1v, cv = d.s1.values, d.cand.values
    match_map = _lists(s1v[mask], cv[mask])
    cand_map = _lists(s1v, cv)
    print(f"[hybrid] matched: {len(match_map):,} S1, {sum(len(v) for v in match_map.values()):,} matches")

    s1_ids = prep.load(pipeline.load_config("configs/exp/e011_hop.yaml"), "test", [1], ["entity_id"]).entity_id.tolist()
    out = Path(paths.get("submissions_dir", "submissions")) / f"sub_{time.strftime('%Y%m%d_%H%M')}_hybrid_e011_e010"
    out.mkdir(parents=True, exist_ok=True)
    io.write_id_lists(match_map, s1_ids, out / "matching_results.tsv", "matched_entity_ids")
    io.write_id_lists(cand_map, s1_ids, out / "candidate_pairs.tsv", "candidate_entity_ids")
    print(f"[hybrid] wrote {out}")

    # copy to output/ as latest
    od = Path(paths.get("output_dir", "output"))
    od.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(out / "matching_results.tsv", od / "matching_results.tsv")
    shutil.copy(out / "candidate_pairs.tsv", od / "candidate_pairs.tsv")
    print(f"[hybrid] mirrored to {od}")


if __name__ == "__main__":
    main()
