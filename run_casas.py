"""
Run the same method on real CASAS homes (next step of the project).

Each CASAS file is treated as one home (one federated client). Days are split
in time order: first 70% train, next 15% validation (to set each home's alert
threshold), last 15% test. CASAS has no labelled health events, so this script
reports which test days the model finds unusual and why; checking those days
by hand against the activity labels is part of the research.

Usage:
    python run_casas.py data/aruba/data data/milan/data ...            # rooms inferred
    python run_casas.py data/aruba/data --room-map maps/aruba.json      # your own map
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from run_experiment import CFG, block_stats, scores_and_recon
from src.casas_loader import infer_room_map, load_days, load_room_map
from src.explain import explain_day
from src.features import cell_mask, to_input
from src.federated import train_federated


def main(files, room_maps, out):
    clients, homes = {}, {}
    for i, path in enumerate(files):
        rmap = load_room_map(room_maps[i]) if room_maps else infer_room_map(path)
        dates, counts, mask = load_days(path, rmap)
        if len(dates) < 20:
            print(f"skip {path}: only {len(dates)} days")
            continue
        n = len(dates)
        a, b = int(n * 0.7), int(n * 0.85)
        X = torch.tensor(np.stack([to_input(c, mask) for c in counts]))
        M = torch.tensor(np.tile(cell_mask(mask), (n, 1)))
        hid = f"home{i + 1}_{Path(path).parent.name or Path(path).stem}"
        homes[hid] = dict(dates=dates, counts=counts, X=X, M=M, split=(a, b),
                          room_map=rmap)
        clients[hid] = (X[:a], M[:a])
        print(f"{hid}: {n} days, rooms with sensors: "
              f"{sorted(set(rmap.values()))}")

    models = train_federated(clients, CFG, seed=0, personalised=True)
    report = {}
    for hid, h in homes.items():
        a, b = h["split"]
        m = models[hid]
        st = block_stats(m, h["X"][:a], h["M"][:a])
        val_s, _ = scores_and_recon(m, h["X"][a:b], h["M"][a:b], "block_z", st)
        thr = float(np.percentile(val_s, CFG["threshold_pct"]))
        s, R = scores_and_recon(m, h["X"][b:], h["M"][b:], "block_z", st)
        baseline = np.mean(h["counts"][:a], axis=0)
        flagged = []
        for j in np.argsort(-s):
            if s[j] <= thr:
                break
            k = b + j
            ex = explain_day(h["X"][k].numpy(), R[j], h["M"][k].numpy(),
                             h["counts"][k], baseline)
            flagged.append({"date": str(h["dates"][k]), "score": float(s[j]),
                            "explanation": [e["text"] for e in ex]})
        report[hid] = {"threshold": thr, "test_days": len(s), "flagged": flagged}
        print(f"\n{hid}: {len(flagged)} of {len(s)} test days flagged")
        for f in flagged[:5]:
            print(f"  {f['date']}  {f['explanation'][0]}")
    Path(out).write_text(json.dumps(report, indent=2))
    print(f"\nFull report: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--room-map", nargs="*", default=None,
                    help="one JSON room map per file, in the same order")
    ap.add_argument("--out", default="results/casas_report.json")
    args = ap.parse_args()
    main(args.files, args.room_map, args.out)
