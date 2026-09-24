"""Mirror every runs/**/progress.csv into TensorBoard event files, without touching the training processes.

Training keeps writing its CSV (every 10 updates); this polls those files and appends only the new rows.
It needs just the ``tensorboard`` package, so it can run from any env that has it:

    python scripts/tb_sync.py &                                  # polls runs/ every 30 s
    tensorboard --logdir runs/_tensorboard                       # http://localhost:6006

Tags are grouped: eval/, loss/, judge/, rollout/, time/. J_T also gets a log10 copy. Re-running from scratch
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="runs")
    p.add_argument("--out", default=None, help="default: <root>/_tensorboard")
    p.add_argument("--every", type=float, default=30.0, help="seconds between polls")
    p.add_argument("--once", action="store_true")
    args = p.parse_args()
    out = args.out or os.path.join(args.root, "_tensorboard")
    shutil.rmtree(out, ignore_errors=True)      # rebuilt from the CSVs, which are the source of truth
    writers, seen = {}, {}
    while True:
        sync(args.root, out, writers, seen)
        if args.once:
            break
        time.sleep(args.every)
    for w in writers.values():
        w.close()


if __name__ == "__main__":
    main()
