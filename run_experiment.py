"""
Main experiment: compare local, central, FedAvg and personalised FL (FedPer)
for detecting unusual days, and test the explanations.

Run:   python run_experiment.py            (full run, 3 seeds, a few minutes on a laptop)
       python run_experiment.py --quick    (fast check that everything works)

Results are written to results/.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from src.explain import block_errors, deletion_test, explain_day, explanation_hit
from src.features import cell_mask, day_counts, room_mask, to_input
from src.federated import (shared_param_count, train_central, train_federated,
                           train_local)
from src.model import masked_mse
from src.simulate import ANOMALY_TYPES, make_personas, simulate_home

CFG = {
    "n_homes": 10, "n_train": 70, "n_val": 14, "n_test": 40, "n_anom_per_type": 2,
    "lr": 1e-3, "batch": 16,
    "local_only_epochs": 150, "central_epochs": 60,
    "rounds": 50, "local_epochs": 3,
    "threshold_pct": 97, "dp_clip": 1.0, "dp_sigmas": [0.5, 1.0],
    "seeds": [0, 1, 2],
}
OUT = Path(__file__).parent / "results"


def build_world(seed):
    rng = np.random.default_rng(seed)
    personas = make_personas(CFG["n_homes"], rng)
    world = {}
    for p in personas:
        train, val, test = simulate_home(p, rng, CFG["n_train"], CFG["n_val"],
                                         CFG["n_test"], CFG["n_anom_per_type"])
        mask = room_mask(p.sensors)

        def pack(days):
            counts = [day_counts(d.times, d.sensor_room) for d in days]
            X = torch.tensor(np.stack([to_input(c, mask) for c in counts]))
            M = torch.tensor(np.tile(cell_mask(mask), (len(days), 1)))
            return X, M, counts

        world[p.home_id] = {"persona": p, "train": pack(train), "val": pack(val),
                            "test": pack(test), "test_days": test}
    return world


@torch.no_grad()
def block_stats(model, X, M):
    """Per-block mean and std of the error on a home's own normal training days."""
    model.eval()
    r = model(X).numpy()
    be = np.stack([block_errors(X[i].numpy(), r[i], M[i].numpy())
                   for i in range(len(X))])
    return be.mean(axis=0), be.std(axis=0) + 1e-4


@torch.no_grad()
def scores_and_recon(model, X, M, scoring="day_mean", stats=None):
    """day_mean: average error over the whole day (version 1).
    block_max: error of the single most unusual (room x 2 h) block.
    block_z: like block_max, but each block's error is compared with how much
             that block normally varies for this person (z-score).
    Version 2 was added after version 1 missed short events such as a skipped
    meal, whose error gets diluted when averaged over the whole day."""
    model.eval()
    r = model(X)
    if scoring == "day_mean":
        return masked_mse(r, X, M).numpy(), r.numpy()
    x, rn, m = X.numpy(), r.numpy(), M.numpy()
    be = np.stack([block_errors(x[i], rn[i], m[i]) for i in range(len(x))])
    if scoring == "block_max":
        return be.reshape(len(x), -1).max(axis=1), rn
    # block_z: how unusual each block is compared with its own normal spread
    mu, sd = stats
    z = (be - mu) / sd
    return z.reshape(len(x), -1).max(axis=1), rn


def evaluate(models, world, method, seed, scoring):
    rows, type_rows = [], []
    for hid, h in world.items():
        m = models[hid]
        st = block_stats(m, h["train"][0], h["train"][1]) if scoring == "block_z" else None
        val_s, _ = scores_and_recon(m, h["val"][0], h["val"][1], scoring, st)
        thr = np.percentile(val_s, CFG["threshold_pct"])
        test_s, _ = scores_and_recon(m, h["test"][0], h["test"][1], scoring, st)
        y = np.array([d.anomaly is not None for d in h["test_days"]])
        pred = test_s > thr
        n_norm = (~y).sum()
        rows.append({
            "method": method, "scoring": scoring, "seed": seed, "home": hid,
            "auroc": roc_auc_score(y, test_s),
            "f1": f1_score(y, pred, zero_division=0),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "false_alarms_per_week": (pred & ~y).sum() / n_norm * 7,
        })
        for t in ANOMALY_TYPES:
            idx = [i for i, d in enumerate(h["test_days"]) if d.anomaly == t]
            type_rows.append({"method": method, "scoring": scoring,
                              "seed": seed, "home": hid,
                              "type": t, "detected": pred[idx].mean()})
    return rows, type_rows


def explain_all(models, world, seed):
    rng = np.random.default_rng(seed)
    rows, examples = [], []
    for hid, h in world.items():
        m = models[hid]
        Xtr, _, ctr = h["train"]
        baseline = np.mean(ctr, axis=0)
        st = block_stats(m, h["train"][0], h["train"][1])
        val_s, _ = scores_and_recon(m, h["val"][0], h["val"][1], "block_z", st)
        thr = np.percentile(val_s, CFG["threshold_pct"])
        X, M, counts = h["test"]
        s, R = scores_and_recon(m, X, M, "block_z", st)
        for i, d in enumerate(h["test_days"]):
            if not d.anomaly or s[i] <= thr:
                continue
            x, r, cm = X[i].numpy(), R[i], M[i].numpy()
            ex = explain_day(x, r, cm, counts[i], baseline)
            top_drop, rand_drop = deletion_test(x, r, cm, rng)
            rows.append({"seed": seed, "home": hid, "type": d.anomaly,
                         "hit": explanation_hit(ex, d), "top_share": ex[0]["share"],
                         "deletion_drop_top": top_drop,
                         "deletion_drop_random": rand_drop})
            examples.append({"seed": seed, "home": hid, "type": d.anomaly,
                             "score": float(s[i]), "threshold": float(thr),
                             "explanation": [e["text"] for e in ex],
                             "x": x.tolist(), "recon": r.tolist(),
                             "baseline": baseline.tolist(),
                             "counts": counts[i].tolist(), "mask": cm.tolist()})
    return rows, examples


def main(quick=False):
    if quick:
        CFG.update({"seeds": [0], "rounds": 10, "local_only_epochs": 30,
                    "central_epochs": 15, "dp_sigmas": [1.0]})
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    all_rows, all_types, all_expl, all_examples, logs, timing = [], [], [], [], {}, {}

    for seed in CFG["seeds"]:
        t0 = time.time()
        world = build_world(seed)
        clients = {hid: (h["train"][0], h["train"][1]) for hid, h in world.items()}
        print(f"[seed {seed}] simulated {len(world)} homes in {time.time() - t0:.1f}s")

        runs = {
            "local": lambda: train_local(clients, CFG, seed),
            "central": lambda: train_central(clients, CFG, seed),
            "fedavg": lambda: train_federated(clients, CFG, seed, personalised=False,
                                              log=logs.setdefault(f"fedavg_{seed}", [])),
            "fedper": lambda: train_federated(clients, CFG, seed, personalised=True,
                                              log=logs.setdefault(f"fedper_{seed}", [])),
        }
        for sigma in CFG["dp_sigmas"]:
            runs[f"fedper_dp{sigma}"] = (lambda s=sigma: train_federated(
                clients, CFG, seed, personalised=True, dp_sigma=s))

        for name, fn in runs.items():
            t1 = time.time()
            models = fn()
            timing[f"{name}_{seed}"] = time.time() - t1
            for scoring in ("day_mean", "block_max", "block_z"):
                rows, trows = evaluate(models, world, name, seed, scoring)
                all_rows += rows
                all_types += trows
            print(f"  {name:13s} [block_z] AUROC {np.mean([r['auroc'] for r in rows]):.3f}  "
                  f"F1 {np.mean([r['f1'] for r in rows]):.3f}  "
                  f"false alarms/wk {np.mean([r['false_alarms_per_week'] for r in rows]):.2f}"
                  f"  ({timing[f'{name}_{seed}']:.0f}s)")
            if name == "fedper":
                e_rows, ex = explain_all(models, world, seed)
                all_expl += e_rows
                all_examples += ex

    per_home = pd.DataFrame(all_rows)
    per_home.to_csv(OUT / "per_home.csv", index=False)

    # mean over homes per seed, then mean and std over seeds
    by_seed = per_home.groupby(["method", "scoring", "seed"]).mean(numeric_only=True)
    summary = by_seed.groupby(["method", "scoring"]).agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.drop(columns=[c for c in summary.columns if c.startswith("seed")],
                           errors="ignore")
    comm = {"local": 0, "central": None,
            "fedavg": shared_param_count(False), "fedper": shared_param_count(True)}
    for s in CFG["dp_sigmas"]:
        comm[f"fedper_dp{s}"] = shared_param_count(True)
    summary["shared_params"] = [comm.get(m) for m, _ in summary.index]
    summary["kb_per_home_per_round"] = [
        (c * 4 * 2 / 1024) if c else None for c in summary["shared_params"]]
    summary.round(4).to_csv(OUT / "summary.csv")

    types = pd.DataFrame(all_types).groupby(["method", "scoring", "type"])["detected"].mean()
    types.unstack().round(3).to_csv(OUT / "detection_by_type.csv")

    expl = pd.DataFrame(all_expl)
    expl.to_csv(OUT / "explanations.csv", index=False)
    expl_summary = {
        "n_explained_alerts": int(len(expl)),
        "hit_rate_overall": float(expl["hit"].mean()),
        "hit_rate_by_type": expl.groupby("type")["hit"].mean().round(3).to_dict(),
        "deletion_drop_top_block": float(expl["deletion_drop_top"].mean()),
        "deletion_drop_random_block": float(expl["deletion_drop_random"].mean()),
    }
    (OUT / "explanation_summary.json").write_text(json.dumps(expl_summary, indent=2))
    (OUT / "examples.json").write_text(json.dumps(all_examples))
    (OUT / "training_log.json").write_text(json.dumps(logs))
    (OUT / "config.json").write_text(json.dumps(CFG, indent=2))

    with open(OUT / "example_explanations.txt", "w") as f:
        rng = np.random.default_rng(0)
        for t in ANOMALY_TYPES:
            ex = [e for e in all_examples if e["type"] == t]
            for e in (rng.choice(ex, size=min(2, len(ex)), replace=False) if ex else []):
                f.write(f"[{e['home']}, seed {e['seed']}] injected: {t}\n")
                for line in e["explanation"]:
                    f.write(f"  - {line}\n")
                f.write("\n")

    print("\nSummary (mean over homes, then mean ± std over seeds):")
    cols = ["auroc_mean", "auroc_std", "f1_mean", "false_alarms_per_week_mean"]
    print(summary[cols].round(3).to_string())
    print("\nExplanations:", json.dumps(expl_summary, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
