"""
Read real CASAS smart-home files (https://casas.wsu.edu/datasets/) and turn
them into the same day matrices the simulator produces.

A CASAS line looks like:
    2010-11-04 00:03:50.209589 M003 ON Sleeping begin
    date       time            sensor value [activity begin|end]

We use motion sensors (M..., value ON) and door sensors (D..., OPEN/CLOSE).
Temperature and other sensors are ignored.

Each sensor must be assigned to one of our five room types. There are two ways:
  1. A JSON file you write by looking at the dataset's floor plan:
         {"M003": "bedroom", "M015": "kitchen", "D001": "entrance", ...}
  2. Automatically, from the activity labels in the file: a sensor that fires
     mostly during "Meal_Preparation" is probably in the kitchen, and so on.
     Always check the automatic result against the floor plan.
"""

import json
import re
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np

from .features import day_counts
from .simulate import ROOM_INDEX, ROOMS

LINE = re.compile(r"^(\d{4}-\d{2}-\d{2})\s+(\d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+(\S+)\s+(\S+)(?:\s+(\S+)\s+(begin|end))?")

ACTIVITY_ROOM = {        # CASAS activity label -> room type
    "meal_preparation": "kitchen", "wash_dishes": "kitchen", "eating": "kitchen",
    "sleeping": "bedroom", "bed_to_toilet": "bathroom", "relax": "living",
    "enter_home": "entrance", "leave_home": "entrance", "housekeeping": None,
    "work": None, "respirate": None,
}


def parse(path):
    """Yield (datetime, sensor, value, activity, begin/end) for each valid line."""
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE.match(line.strip())
            if not m:
                continue
            d, t, sensor, value, act, be = m.groups()
            fmt = "%Y-%m-%d %H:%M:%S.%f" if "." in t else "%Y-%m-%d %H:%M:%S"
            try:
                ts = datetime.strptime(f"{d} {t}", fmt)
            except ValueError:
                continue
            yield ts, sensor, value, act, be


def _is_event(sensor, value):
    return (sensor.startswith("M") and value == "ON") or \
           (sensor.startswith("D") and value in ("OPEN", "CLOSE"))


def infer_room_map(path, min_share=0.4):
    """Guess each sensor's room from the activity labels it fires during."""
    current = None
    votes = defaultdict(Counter)
    for ts, sensor, value, act, be in parse(path):
        if act and be == "begin":
            current = ACTIVITY_ROOM.get(act.lower())
        if _is_event(sensor, value) and current:
            votes[sensor][current] += 1
        if act and be == "end":
            current = None
    room_map = {}
    for sensor, c in votes.items():
        room, n = c.most_common(1)[0]
        if n / sum(c.values()) >= min_share:
            room_map[sensor] = room
    return room_map


def load_days(path, room_map):
    """Return (sorted dates, list of count matrices, room mask)."""
    by_day = defaultdict(lambda: ([], []))
    for ts, sensor, value, _, _ in parse(path):
        room = room_map.get(sensor)
        if room not in ROOM_INDEX or not _is_event(sensor, value):
            continue
        secs = ts.hour * 3600 + ts.minute * 60 + ts.second + ts.microsecond / 1e6
        by_day[ts.date()][0].append(secs)
        by_day[ts.date()][1].append(ROOM_INDEX[room])
    dates = sorted(by_day)
    counts = [day_counts(np.array(by_day[d][0]), np.array(by_day[d][1])) for d in dates]
    mask = np.array([1.0 if r in room_map.values() else 0.0 for r in ROOMS])
    return dates, counts, mask


def load_room_map(path):
    with open(path) as f:
        return json.load(f)
