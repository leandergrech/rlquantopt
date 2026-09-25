"""Compact records of the idea iterations (the "working ideas" pages cite these), built from runs/.

    python scripts/idea_records.py                     # all ideas it knows
    python scripts/idea_records.py --scratch <dir>     # also copy smoke-test runs kept outside runs/

Writes results/iNN_<codename>/: summary.json (the numbers the docs quote) and small source files
(progress.csv of smoke tests, toy-grid curves). The raw runs stay in runs/ (gitignored).
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rlquantopt.jx.ideas import IDEAS  # noqa: E402

BASELINES = {f"ppo_s{s}": f"results/jax_ppo_s{s}" for s in ("123", "1", "2", "3")}


def gate_run(d):
    """Headline numbers of one train.py run on the gate env."""
    p = pd.read_csv(os.path.join(d, "progress.csv"))
    e = p.dropna(subset=["eval_JT_min"])
    r = dict(steps=int(p.step.iloc[-1]), wall_h=round(float(p.wall.iloc[-1]) / 3600, 2),
             best_JT=float(e.eval_JT_min.min()), median_JT_last30=float(e.eval_JT_min.iloc[-30:].median()),
             reaches_PE_last30=float(e.eval_t_first_PE.iloc[-30:].notna().mean()),
             approx_kl_last300=float(p.approx_kl.iloc[-300:].mean()), clip_frac_last300=float(p.clip_frac.iloc[-300:].mean()))
    for thr in (1e-1, 1e-2, 1e-3):
        hit = e[e.eval_JT_min < thr]
        r[f"steps_to_JT<{thr:g}"] = int(hit.step.iloc[0]) if len(hit) else None
    for col in ("judge_corr", "frac_steps_taken", "sil_active"):
        if col in p:
            v = p[col].dropna()
            q = max(len(v) // 4, 1)
            r[col] = dict(first_quarter=float(v.iloc[:q].mean()), last_quarter=float(v.iloc[-q:].mean()))
    cfg = json.load(open(os.path.join(d, "config.json")))
    r["args"] = {k: v for k, v in cfg["args"].items() if v not in (None, False, "", 0.0) or k in ("seed",)}
    return r


def gate_idea(codename):
    n = IDEAS[codename].number
    runs = sorted(glob.glob(f"runs/i{n:02d}_{codename}/*_s*/progress.csv"))
    return {os.path.basename(os.path.dirname(f)).split("_", 3)[-1]: gate_run(os.path.dirname(f)) for f in runs}


def write(codename, summary, extra_files=()):
    n = IDEAS[codename].number
    out = f"results/i{n:02d}_{codename}"
    os.makedirs(out, exist_ok=True)
    summary = dict(idea=codename, number=n, summary=IDEAS[codename].summary, **summary)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    for src, name in extra_files:
        shutil.copy(src, os.path.join(out, name))
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", default=None, help="a runs directory holding the chameleon/dragonfly smoke tests")
    args = ap.parse_args()

    write("axolotl", dict(gate_runs=gate_idea("axolotl"), baselines={k: gate_run(v) for k, v in BASELINES.items()}))
    write("badger", dict(gate_runs=gate_idea("badger")))

    for codename in ("chameleon", "dragonfly"):
        src = glob.glob(os.path.join(args.scratch or "runs", f"i0{IDEAS[codename].number}_{codename}/*smoke/progress.csv"))
        if src:
            p = pd.read_csv(src[0])
            write(codename, dict(smoke=dict(updates=len(p), steps=int(p.step.iloc[-1]),
                                            s_per_update=float(p.wall.diff().iloc[5:].median()),
                                            eval_return_last=float(p.eval_return.dropna().iloc[-1]),
                                            mean_reward_first=float(p.mean_reward.iloc[0]),
                                            mean_reward_best=float(p.mean_reward.max()),
                                            mean_reward_last=float(p.mean_reward.iloc[-1]))),
                  [(src[0], "smoke_progress.csv")])

    toy = {}
    extra = []
    for f in sorted(glob.glob("runs/i05_echidna/toybench/*/summary.json")):
        s = json.load(open(f))
        toy[f"{s['env']}/{s['variant']}"] = {k: s[k] for k in ("final_mean", "final_std", "auc_mean", "auc_std",
                                                               "final_per_seed", "seeds")}
        extra.append((os.path.join(os.path.dirname(f), "curves.npz"), f"toy_{s['env']}_{s['variant']}.npz"))
    write("echidna", dict(toy_grid=toy, gate_runs=gate_idea("echidna")), extra)

    big = sorted(d for d in glob.glob("runs/i06_fox/foxbench/tracker_*") if not os.path.exists(os.path.join(d, "INVALID.txt")))
    groups = {"first_24": [d for d in big if d.endswith("115532")], "main_72": [d for d in big if "-1259" in d]}
    fox = {}
    for name, dirs in groups.items():
        if dirs:
            res = subprocess.run([sys.executable, "-m", "rlquantopt.jx.foxbench", "--summarise", *dirs, "--budget", "60"],
                                 capture_output=True, text=True, check=True)
            fox[name] = json.loads(res.stdout)
    write("fox", dict(benchmarks=fox, invalid=[d for d in glob.glob("runs/i06_fox/foxbench/*/INVALID.txt")]))

    hip = {}
    for f in sorted(glob.glob("runs/i08_hippogriff/seed*/results.json")):
        r = json.load(open(f))
        tag = "noisy" if f.split("/")[-2].endswith("noisy") else "quiet"
        run = dict(seed=r["args"]["seed"], method=r.get("method"), calib=r["calib"], calib_blind=r["calib_blind"],
                   model_b_fit=r["model_b_fit"], modes={})
        for mode, m in r["modes"].items():
            ret = np.array(m["returns"])
            run["modes"][mode] = dict(return_per_episode=ret.mean(1).tolist(),
                                      zerr_end_ep1=float(np.array(m["zerr"])[0, :, -1].mean()),
                                      rho_end_ep1=float(np.array(m["rho"])[0, :, -1].mean()),
                                      probing_steps_ep1=float(np.mean(m["probing_steps"][0])))
        hip.setdefault(tag, []).append(run)
    if hip:
        write("hippogriff", dict(runs=hip))


if __name__ == "__main__":
    main()
