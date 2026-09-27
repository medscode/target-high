"""Score a single IN chunk directly using cached OOF models and test features.
Uses smaller query_chunk and prints progress."""
import sys, time, json, gc
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ber import decide, pipeline, prep, retrieve, featurize, model  # noqa


def score_one_in(country, s_lo, s_hi):
    """Score a single chunk (s1 range) for the given country."""
    cfg = pipeline.load_config("configs/exp/e011_hop_fast.yaml")
    paths = cfg["paths"]
    ck = Path(paths["cache_dir"]) / f"test_chunks_{cfg['exp_id']}_{pipeline.stable_hash(cfg)}"
    out_path = ck / f"{country}_{s_lo}_{s_hi}.parquet"
    if out_path.exists():
        print(f"[skip] {out_path.name} already done")
        return
    print(f"[score] {country} {s_lo}-{s_hi}", flush=True)

    # build pool from test sources
    cfg2 = pipeline.load_config("configs/exp/e011_hop.yaml")
    test = prep.load(cfg2, "test", [1], ["entity_id", "bc"]).query("bc == @country").reset_index(drop=True)
    test = test[test.entity_id.str.startswith("S1-")].reset_index(drop=True)
    # find indices in the range
    starts = test["entity_id"].str.replace("S1-", "").astype(int).values
    idx = np.where((starts >= s_lo) & (starts < s_hi))[0]
    if len(idx) == 0:
        print(f"[score] no queries in {country} {s_lo}-{s_hi}", flush=True)
        return
    queries = test.iloc[idx].reset_index(drop=True)
    print(f"[score] {len(queries):,} queries", flush=True)

    # pool
    s2 = pd.read_csv(Path(paths["data_dir"]) / "test" / "test_source2.tsv",
                     sep="\t", dtype=str, on_bad_lines="skip", engine="python")
    s3 = pd.read_csv(Path(paths["data_dir"]) / "test" / "test_source3.tsv",
                     sep="\t", dtype=str, on_bad_lines="skip", engine="python")
    s2["entity_id"] = s2["entity_id"].astype(str); s3["entity_id"] = s3["entity_id"].astype(str)
    pool = pd.concat([s2, s3], ignore_index=True)
    print(f"[score] pool: {len(pool):,}", flush=True)

    # block + featurize
    t0 = time.time()
    import hashlib
    h = hashlib.md5(json.dumps(cfg2, sort_keys=True, default=str).encode()).hexdigest()[:8]
    block_path = Path(paths["cache_dir"]) / f"test_block_{country}_{h}"
    block_path.mkdir(parents=True, exist_ok=True)
    feats, cand_path = retrieve.test_features(cfg2, queries, pool, country, block_path, cfg2.get("test_query_chunk", 100000))
    print(f"[score] featurized: {len(feats):,} pairs ({time.time()-t0:.0f}s)", flush=True)
    del s2, s3, pool

    # load OOF to get iters
    saved = Path(paths["results_dir"]) / "e011_fast" / "metrics.json"
    m = json.loads(saved.read_text())
    n_iter = int(m["best_iter_median"] * cfg["model"].get("full_iter_mult", 0.9))
    print(f"[score] n_iter: {n_iter}", flush=True)

    # train full model
    t1 = time.time()
    tr = pipeline.build_train(cfg2)
    feats_tr, s1_tr = tr["feats"], tr["s1"]
    truth = io.load_ground_truth(Path(paths["data_dir"]) / "train" / "train_ground_truth.tsv",
                                 s1_tr.entity_id.tolist()) if False else {}
    # quick load of truth from prep
    from ber import io as _io
    truth = _io.load_ground_truth(Path(paths["data_dir"]) / "train" / "train_ground_truth.tsv",
                                  s1_tr.entity_id.tolist())
    y = np.fromiter((c in truth[s] for s, c in zip(feats_tr.s1, feats_tr.cand)), bool, len(feats_tr)).astype(int)
    w_rows = model.entity_weights(cfg["model"], feats_tr.s1.values)
    models = model.train_full(feats_tr[m["features"]], y, cfg["model"], n_iter, w_rows)
    print(f"[score] trained ({time.time()-t1:.0f}s)", flush=True)
    del feats_tr, tr

    # predict
    p = model.predict(models, feats[m["features"]]).astype(np.float32)
    out = feats[["s1", "cand", "cand_src"]].assign(p=p)
    out.to_parquet(out_path)
    print(f"[score] wrote {out_path.name} ({time.time()-t0:.0f}s total)", flush=True)


if __name__ == "__main__":
    score_one_in(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
