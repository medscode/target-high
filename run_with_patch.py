"""Wrapper to run ber.* scripts with the sandbox ProcessPoolExecutor patch applied.

Usage: run_with_patch.py <ber.module> [args...]

Patches:
1. concurrent.futures.process._check_system_limits → no-op (sandbox blocks os.sysconf)
2. os.sysconf('SC_SEM_NSEMS_MAX') → 256
3. ber.retrieve.hash_views → sequential (sandbox segfaults inside ProcessPoolExecutor workers
   when sharing large CSR matrices / TF-IDF state)
"""
import os
import sys
import concurrent.futures.process

concurrent.futures.process._check_system_limits = lambda: None
os.sysconf_orig = os.sysconf
os.sysconf = lambda name: 256 if name == "SC_SEM_NSEMS_MAX" else os.sysconf_orig(name)


def _patch_hash_views():
    from ber import retrieve as _retr
    import numpy as _np
    import scipy.sparse as _sp

    def _seq(df, views, workers=7, chunk=100_000):
        cols = [c for c in _retr._DOC_COLS if c in df]
        starts = list(range(0, len(df), chunk))
        res = [_retr._hash_chunk(df[cols].iloc[s:s + chunk], views) for s in starts]
        X = {v: _sp.vstack([r[0][v] for r in res]).tocsr() for v in views}
        return X, _np.concatenate([r[1] for r in res])

    _retr.hash_views = _seq


def _patch_prep():
    """prep.prep_split also uses ProcessPoolExecutor. Patch to use threads sequentially
    (ProcessPoolExecutor inside the sandboxed shell reliably segfaults)."""
    from ber import prep as _prep

    def _seq(cfg, split, workers=None, chunk=100_000):
        from concurrent.futures import ThreadPoolExecutor
        from pathlib import Path
        out = _prep.norm_dir(cfg, split)
        if (out / "_DONE").exists():
            return out
        out.mkdir(parents=True, exist_ok=True)
        workers = workers or cfg.get("workers", 7)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for s in (1, 2, 3):
                path = Path(cfg["paths"]["data_dir"]) / split / f"{split}_source{s}.tsv"
                reader = _pd_read_tsv(path)
                pending, n = {}, 0
                for i, df in enumerate(reader):
                    if len(pending) >= workers * 2:
                        done = [f for f in pending if f.done()]
                        for f in done:
                            f.result().to_parquet(pending.pop(f), index=False)
                    pending[ex.submit(_prep._norm_chunk, df, s)] = out / f"s{s}_{i:05d}.parquet"
                    n += len(df)
                for f in list(pending):
                    f.result().to_parquet(pending[f], index=False)
                print(f"[prep] {split} source{s}: {n} rows", flush=True)
        (out / "_DONE").write_text("ok")
        return out

    _prep.prep_split = _seq


def _pd_read_tsv(path):
    import pandas as _pd
    import csv as _csv
    return _pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False,
                        quoting=_csv.QUOTE_NONE, chunksize=100_000, encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: run_with_patch.py <ber.module> [args...]")
        sys.exit(1)

    mod_name = sys.argv[1]
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    if mod_name in ("ber.train_pruner", "ber.run_experiment", "ber.make_submission"):
        _patch_hash_views()
    if mod_name == "ber.train_pruner":
        from ber import train_pruner
        train_pruner.main()
    elif mod_name == "ber.run_experiment":
        from ber import run_experiment
        run_experiment.main()
    elif mod_name == "ber.make_submission":
        from ber import make_submission
        make_submission.main()
    elif mod_name == "ber.compare":
        from ber import compare
        compare.main()
    else:
        print(f"Unknown module: {mod_name}")
        sys.exit(1)