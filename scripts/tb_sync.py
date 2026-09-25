"""Mirror every runs/**/progress.csv into TensorBoard event files, without touching the training processes.

Training keeps writing its CSV (every 10 updates); this polls those files and appends only the new rows.
It needs just the ``tensorboard`` package, so it can run from any env that has it:

    python scripts/tb_sync.py &                                  # polls runs/ every 30 s
    tensorboard --logdir runs/_tensorboard                       # http://localhost:6006

Tags are grouped: eval/, loss/, judge/, rollout/, time/. J_T also gets a log10 copy.
It also renders the toy benchmarks, which write whole files rather than a CSV: the i05 toy grid (toy/),
fox (fox/, fox_model/) and hippogriff (hippo/); a benchmark directory holding INVALID.txt is skipped. Re-running from scratch
is safe: the output directory is rebuilt, and each run's events go to runs/_tensorboard/<run path>.
"""
import argparse
import csv
import glob
import math
import os
import shutil
import time

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter

GROUPS = {"eval_": "eval", "judge_": "judge", "help_": "judge", "guide_": "judge"}
LOSSES = {"pg_loss", "v_loss", "approx_kl", "clip_frac", "kl", "log_std", "entropy"}
ROLLOUT = {"mean_reward", "min_JT", "frac_done"}


def tag_for(key):
    for prefix, group in GROUPS.items():
        if key.startswith(prefix):
            return f"{group}/{key[len('eval_'):] if group == 'eval' else key}"
    if key in LOSSES:
        return f"loss/{key}"
    if key in ROLLOUT:
        return f"rollout/{key}"
    return f"other/{key}"


def scalars(row):
    step = int(float(row["step"]))
    out = {"time/wall_hours": float(row["wall"]) / 3600} if row.get("wall") else {}
    for k, v in row.items():
        if k in ("step", "wall") or v in ("", None):
            continue
        try:
            x = float(v)
        except ValueError:
            continue
        if not math.isfinite(x):
            continue
        out[tag_for(k)] = x
        if "JT" in k and "t_JT" not in k and x > 0:
            out[tag_for(k) + "_log10"] = math.log10(x)
    return step, out


def sync(root, out, writers, seen):
    for path in sorted(glob.glob(os.path.join(root, "**", "progress.csv"), recursive=True)):
        run = os.path.relpath(os.path.dirname(path), root)
        if run.startswith("_tensorboard"):
            continue
        try:
            with open(path) as f:
                rows = list(csv.DictReader(f))
        except (OSError, csv.Error):
            continue        # caught mid-write; next poll
        new = rows[seen.get(run, 0):]
        if not new:
            continue
        if run not in writers:
            writers[run] = EventFileWriter(os.path.join(out, run))
        w = writers[run]
        for row in new:
            if not row.get("step"):
                continue
            step, vals = scalars(row)
            summary = Summary(value=[Summary.Value(tag=t, simple_value=v) for t, v in vals.items()])
            w.add_event(Event(wall_time=time.time(), step=step, summary=summary))
        w.flush()
        seen[run] = len(rows)


def _write(out, run, series):
    """series: {tag: [(step, value), ...]} -> a fresh event file for ``run`` (replacing any earlier one)."""
    d = os.path.join(out, run)
    shutil.rmtree(d, ignore_errors=True)
    w = EventFileWriter(d)
    for tag, pts in series.items():
        for step, v in pts:
            if v is not None and math.isfinite(v):
                w.add_event(Event(wall_time=time.time(), step=int(step),
                                  summary=Summary(value=[Summary.Value(tag=tag, simple_value=float(v))])))
    w.close()


def _toybench(path, root):
    """i05 toy grid: one TB run per env/variant; eval return against env steps (mean and per seed)."""
    import numpy as np
    c = np.load(path)
    name = os.path.basename(os.path.dirname(path))            # <env>_<variant>_<timestamp>
    env, variant = name.split("_")[0], name.split("_")[1]
    s = {"toy/eval_return_mean": list(zip(c["step"], c["ret"].mean(1))),
         "toy/veto_rate": list(zip(c["step"], c["veto_rate"].mean(1)))}
    for i in range(c["ret"].shape[1]):
        s[f"toy/eval_return_seed{i}"] = list(zip(c["step"], c["ret"][:, i]))
    base = os.path.relpath(os.path.dirname(os.path.dirname(path)), root)
    return {f"{base}/{env}/{variant}": s}


def _foxbench(path, root):
    """i06 fox: per results file, ppo and fox curves over adaptation updates (means over devices and seeds)."""
    import json as _json
    import numpy as np
    rows = _json.load(open(path))
    if not rows:
        return {}
    base = os.path.relpath(os.path.dirname(path), root)
    ev = {m: np.array([r[m]["evals"] for r in rows]) for m in ("ppo", "fox")}          # (runs, B+1)
    start = ev["ppo"][:, :1]
    runs = {}
    for m in ("ppo", "fox"):
        runs[f"{base}/{m}"] = {"fox/eval_return": list(enumerate(ev[m].mean(0))),
                               "fox/gain_over_start": list(enumerate((ev[m] - start).mean(0)))}
    diff = ev["fox"] - ev["ppo"]
    runs[f"{base}/fox_minus_ppo"] = {"fox/eval_diff_mean": list(enumerate(diff.mean(0))),
                                     "fox/eval_diff_sem": list(enumerate(diff.std(0) / np.sqrt(len(diff)))),
                                     "fox/frac_devices_fox_better": list(enumerate((diff > 0).mean(0)))}
    logs = [r["fox"]["log"] for r in rows]
    B = len(logs[0])
    mean_of = lambda key: [(k, float(np.nanmean([l[k].get(key, np.nan) for l in logs]))) for k in range(B)]
    stop = np.array([r["fox"]["stopped_at"] or np.inf for r in rows])
    runs[f"{base}/fox"].update({"fox_model/confidence_z": mean_of("z"), "fox_model/gain_per_unit_step": mean_of("gain"),
                                "fox_model/pred_next_optimistic": mean_of("pred_next"),
                                "fox_model/exploit_frac": mean_of("exploit"), "fox_model/cache_size": mean_of("n_cache"),
                                "fox_model/frac_stopped": [(k, float((stop <= k).mean())) for k in range(B + 1)]})
    return runs


def _hippogriff(path, root):
    """i08: per mode, return per deployment episode (mean over devices) and, within episode 1, the belief's
    z error and uncertainty ratio rho against the step."""
    import json as _json
    import numpy as np
    res = _json.load(open(path))
    base = os.path.relpath(os.path.dirname(path), root)
    runs = {}
    for mode, r in res.get("modes", {}).items():
        ret = np.array(r["returns"])                          # (episodes, devices)
        z = np.array(r["zerr"])[0].mean(0)                    # episode 1, mean over devices, per step
        rho = np.array(r["rho"])[0].mean(0)
        runs[f"{base}/{mode}"] = {"hippo/return_per_episode": list(enumerate(ret.mean(1), 1)),
                                  "hippo/ep1_z_error": list(enumerate(z)), "hippo/ep1_rho": list(enumerate(rho)),
                                  "hippo/probing_steps_per_episode": list(enumerate(np.array(r["probing_steps"]).mean(1), 1))}
    return runs


BENCH_SOURCES = [("**/toybench/*/curves.npz", _toybench), ("**/foxbench/*/results.json", _foxbench),
                 ("i08_hippogriff/*/results.json", _hippogriff)]


def sync_benchmarks(root, out, mtimes):
    for pattern, render in BENCH_SOURCES:
        for path in sorted(glob.glob(os.path.join(root, pattern), recursive=True)):
            if os.path.exists(os.path.join(os.path.dirname(path), "INVALID.txt")):
                continue
            mt = os.path.getmtime(path)
            if mtimes.get(path) == mt:
                continue
            try:
                runs = render(path, root)
            except Exception:
                continue        # caught mid-write, or an older format; next poll
            for run, series in runs.items():
                _write(out, run, series)
            mtimes[path] = mt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="runs")
    p.add_argument("--out", default=None, help="default: <root>/_tensorboard")
    p.add_argument("--every", type=float, default=30.0, help="seconds between polls")
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    out = args.out or os.path.join(args.root, "_tensorboard")
    shutil.rmtree(out, ignore_errors=True)      # rebuilt from the CSVs, which are the source of truth
    writers, seen, mtimes = {}, {}, {}
    while True:
        sync(args.root, out, writers, seen)
        sync_benchmarks(args.root, out, mtimes)
        if args.once:
            break
        time.sleep(args.every)
    for w in writers.values():
        w.close()


if __name__ == "__main__":
    main()
