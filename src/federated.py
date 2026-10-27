"""
Four ways to train the routine model, compared in the experiment:

  local    each home trains its own model on its own data only
           (fully private, but each home has little data)
  central  all homes send their raw data to one server, one model
           (NOT private; used as a reference point)
  fedavg   standard federated learning (McMahan et al., 2017): every home
           trains locally, the server averages the whole model
  fedper   personalised federated learning: only the encoder is averaged,
           each home keeps its own decoder (Arivazhagan et al., 2019)

All federated training here is simulated on one computer: each "client" is a
home's data, and averaging is done in memory. The Flower framework can replace
this loop later without changing the model or the evaluation.

Optional differential privacy (dp_sigma > 0): each home's update is clipped to
a fixed size and Gaussian noise is added before averaging (the idea behind
DP-FedAvg, McMahan et al., 2018). This prototype does NOT compute a formal
privacy budget (epsilon) yet; that is planned future work.
"""

import copy

import numpy as np
import torch

from .model import RoutineAE, masked_mse


def _train_epochs(model, X, M, epochs, lr, batch, params=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(params if params is not None else model.parameters(), lr=lr)
    n = X.shape[0]
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = masked_mse(model(X[idx]), X[idx], M[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def train_local(clients, cfg, seed):
    models = {}
    for cid, (X, M) in clients.items():
        torch.manual_seed(seed)
        m = RoutineAE()
        models[cid] = _train_epochs(m, X, M, cfg["local_only_epochs"], cfg["lr"],
                                    cfg["batch"], seed=seed)
    return models


def train_central(clients, cfg, seed):
    torch.manual_seed(seed)
    X = torch.cat([c[0] for c in clients.values()])
    M = torch.cat([c[1] for c in clients.values()])
    m = _train_epochs(RoutineAE(), X, M, cfg["central_epochs"], cfg["lr"],
                      cfg["batch"], seed=seed)
    return {cid: m for cid in clients}


def _flat(module):
    return torch.cat([p.data.view(-1) for p in module.parameters()])


def _load_flat(module, vec):
    i = 0
    for p in module.parameters():
        n = p.numel()
        p.data.copy_(vec[i:i + n].view_as(p))
        i += n


def train_federated(clients, cfg, seed, personalised=False, dp_sigma=0.0,
                    log=None):
    """FedAvg (personalised=False) or FedPer (personalised=True)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    global_model = RoutineAE()
    local = {cid: copy.deepcopy(global_model) for cid in clients}
    sizes = {cid: X.shape[0] for cid, (X, _) in clients.items()}
    total = sum(sizes.values())
    shared = (lambda m: m.encoder) if personalised else (lambda m: m)

    for rnd in range(cfg["rounds"]):
        g_vec = _flat(shared(global_model))
        updates, weights = [], []
        for cid, (X, M) in clients.items():
            m = local[cid]
            _load_flat(shared(m), g_vec)                 # receive shared part
            _train_epochs(m, X, M, cfg["local_epochs"], cfg["lr"], cfg["batch"],
                          seed=seed * 1000 + rnd)
            delta = _flat(shared(m)) - g_vec            # what this home learned
            if dp_sigma > 0:
                norm = delta.norm().item()
                delta = delta * min(1.0, cfg["dp_clip"] / (norm + 1e-12))
            updates.append(delta)
            weights.append(sizes[cid] / total)
        agg = sum(w * u for w, u in zip(weights, updates))
        if dp_sigma > 0:
            noise_std = dp_sigma * cfg["dp_clip"] / len(clients)
            agg = agg + torch.from_numpy(
                rng.normal(0, noise_std, agg.numel()).astype(np.float32))
        _load_flat(shared(global_model), g_vec + agg)
        if log is not None and (rnd + 1) % 10 == 0:
            with torch.no_grad():
                losses = [masked_mse(local[c](X), X, M).mean().item()
                          for c, (X, M) in clients.items()]
            log.append((rnd + 1, float(np.mean(losses))))

    if personalised:
        # each home: latest shared encoder + its own decoder
        g_vec = _flat(global_model.encoder)
        for cid in clients:
            _load_flat(local[cid].encoder, g_vec)
        return local
    return {cid: global_model for cid in clients}


def shared_param_count(personalised):
    m = RoutineAE()
    return sum(p.numel() for p in (m.encoder if personalised else m).parameters())
