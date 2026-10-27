"""Run with:  python -m pytest -q   (or simply: python tests/test_pipeline.py)"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.casas_loader import infer_room_map, load_days  # noqa: E402
from src.features import N_SLOTS, cell_mask, day_counts, room_mask, to_input  # noqa: E402
from src.model import RoutineAE, masked_mse  # noqa: E402
from src.simulate import (ROOM_INDEX, export_casas, make_personas,  # noqa: E402
                          simulate_day)

HERE = Path(__file__).parent


def test_infer_room_map_from_labels():
    m = infer_room_map(HERE / "sample_casas.txt")
    assert m["M003"] == "bedroom"
    assert m["M004"] == "bathroom"
    assert m["M015"] == "kitchen" and m["M016"] == "kitchen"
    assert m["M020"] == "living"
    assert m["D001"] == "entrance"
    assert "T001" not in m


def test_load_days_counts_only_motion_and_door():
    m = infer_room_map(HERE / "sample_casas.txt")
    dates, counts, mask = load_days(HERE / "sample_casas.txt", m)
    assert len(dates) == 2
    day1 = counts[0]
    assert day1[ROOM_INDEX["bedroom"]].sum() == 3        # three M003 ON events
    assert day1[ROOM_INDEX["kitchen"]].sum() == 3        # M015 x2 + M016
    assert day1[ROOM_INDEX["entrance"]].sum() == 2       # OPEN + CLOSE
    assert mask.sum() == 5


def test_export_and_reload_matches_simulator(tmp_path=None):
    tmp = Path(tmp_path or HERE / "_tmp")
    tmp.mkdir(exist_ok=True)
    rng = np.random.default_rng(1)
    p = make_personas(3, rng)[0]
    days = [simulate_day(p, rng) for _ in range(3)]
    room_map = export_casas(p, days, "2024-01-01", tmp / "home.txt")
    _, counts, _ = load_days(tmp / "home.txt", room_map)
    for d, c in zip(days, counts):
        assert np.allclose(day_counts(d.times, d.sensor_room), c)


def test_masked_loss_ignores_missing_rooms():
    mask = room_mask({"bedroom": 1, "bathroom": 1, "kitchen": 1})
    x = torch.tensor(to_input(np.ones((5, N_SLOTS)), mask))[None]
    cm = torch.tensor(cell_mask(mask))[None]
    recon = x.clone()
    recon[0, ROOM_INDEX["living"] * N_SLOTS] = 99.0      # error in a missing room
    assert masked_mse(recon, x, cm).item() == 0.0
    assert RoutineAE()(x).shape == x.shape


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
