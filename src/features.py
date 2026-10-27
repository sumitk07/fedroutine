"""
Turn a day of raw sensor events into a fixed-size matrix the model can read.

Every home is mapped onto the same five room types (bedroom, bathroom, kitchen,
living, entrance) and the day is cut into 48 slots of 30 minutes. Each cell is
log(1 + number of sensor events in that room and slot).

Homes that have no sensor in a room get zeros there, plus a mask that tells
the model and the loss function to ignore that room. This is how one model
can be shared across homes with different layouts.
"""

import numpy as np

from .simulate import ROOMS

N_ROOMS = len(ROOMS)
SLOT_MIN = 30
N_SLOTS = 24 * 60 // SLOT_MIN
SCALE = 4.0          # log-counts are divided by this so values sit roughly in [0, 1]


def day_counts(times, sensor_room):
    """Raw event counts per (room, 30-minute slot)."""
    counts = np.zeros((N_ROOMS, N_SLOTS))
    slots = np.clip((times // (SLOT_MIN * 60)).astype(int), 0, N_SLOTS - 1)
    np.add.at(counts, (sensor_room, slots), 1)
    return counts


def room_mask(sensors: dict):
    return np.array([1.0 if sensors.get(r, 0) > 0 else 0.0 for r in ROOMS])


def to_input(counts, mask):
    """Counts (rooms x slots) -> flat model input with missing rooms zeroed."""
    x = np.log1p(counts) / SCALE
    x = x * mask[:, None]
    return x.reshape(-1).astype(np.float32)


def cell_mask(mask):
    return np.repeat(mask, N_SLOTS).astype(np.float32)
