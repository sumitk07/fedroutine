"""
Smart-home simulator for older adults living alone.

Why a simulator?
  Public datasets (CASAS, ARAS) have no labelled health events, and each one
  covers only a few homes. The simulator lets us create many homes that differ
  in layout, number of sensors and daily routine, and inject realistic unusual
  days whose type, room and time we know exactly. This is what we need to test
  whether the method works before moving to real data (see casas_loader.py).

How it works
  1. Each home gets a "persona" (wake time, meal times, outings, night toilet
     trips, ...) and a sensor layout (which rooms have sensors, and how many).
  2. For every day we paint a minute-by-minute timeline: which room the person
     is in and how active they are (0 = still, 1 = moving around).
  3. Each motion sensor in the active room fires as a Poisson process whose rate
     depends on the activity level. Sensors also fire falsely now and then.
  4. Unusual days are created by changing the timeline itself (for example,
     removing breakfast), not by editing the final numbers.

The output for each day is a list of sensor events (time in seconds, sensor id),
which is the same shape of data a real smart home produces.
"""

from dataclasses import dataclass, field

import numpy as np

ROOMS = ["bedroom", "bathroom", "kitchen", "living", "entrance"]
ROOM_INDEX = {r: i for i, r in enumerate(ROOMS)}
NONE = -1          # person not detectable (out of home, or completely still)
MIN_PER_DAY = 1440

ANOMALY_TYPES = ["skipped_meal", "long_inactivity", "night_wandering",
                 "disturbed_sleep", "late_wake"]


@dataclass
class Persona:
    home_id: str
    wake: float                 # minutes after midnight
    sleep: float                # minutes after midnight (can be > 1440)
    breakfast_delay: float
    lunch: float
    dinner: float
    p_out: float                # chance of going out on a given day
    p_nap: float
    night_toilet_rate: float    # expected night toilet trips
    activity_scale: float       # how much this person moves around
    sensors: dict = field(default_factory=dict)   # room -> number of sensors
    noise_rate: float = 0.002   # false triggers per sensor per minute


@dataclass
class Day:
    times: np.ndarray           # event times in seconds after midnight
    sensor_room: np.ndarray     # room index of the sensor that fired
    sensor_id: np.ndarray       # sensor number within the home
    anomaly: str | None = None
    anomaly_rooms: tuple = ()
    anomaly_window: tuple = ()  # (start_hour, end_hour)


def make_personas(n_homes: int, rng: np.random.Generator) -> list[Persona]:
    """Create homes that differ in routine and sensor layout.

    The last two homes are deliberately unusual, because real deployments
    always include people like this:
      - a late riser who goes to bed after midnight;
      - a home with only three sensors (no living room or door sensor).
    """
    personas = []
    for h in range(n_homes):
        wake = rng.normal(6.5, 0.6) * 60
        sensors = {
            "bedroom": int(rng.integers(1, 3)),
            "bathroom": 1,
            "kitchen": int(rng.integers(1, 4)),
            "living": int(rng.integers(1, 4)),
            "entrance": int(rng.integers(0, 2)),      # some homes have no door sensor
        }
        personas.append(Persona(
            home_id=f"H{h + 1:02d}",
            wake=wake,
            sleep=rng.normal(21.8, 0.6) * 60,
            breakfast_delay=rng.uniform(25, 60),
            lunch=rng.normal(12.2, 0.4) * 60,
            dinner=rng.normal(18.3, 0.5) * 60,
            p_out=rng.uniform(0.15, 0.5),
            p_nap=rng.uniform(0.1, 0.7),
            night_toilet_rate=rng.uniform(0.5, 2.0),
            activity_scale=rng.uniform(0.7, 1.3),
            sensors=sensors,
        ))

    if n_homes >= 2:
        late = personas[-2]
        late.wake, late.sleep = 9.8 * 60, 25.5 * 60         # sleeps at 01:30
        late.breakfast_delay, late.lunch, late.dinner = 40, 13.8 * 60, 20.0 * 60
        late.night_toilet_rate = 0.3

        sparse = personas[-1]
        sparse.sensors = {"bedroom": 1, "bathroom": 1, "kitchen": 1,
                          "living": 0, "entrance": 0}
    return personas


def _paint(timeline, intensity, start, end, room, level):
    s, e = int(max(0, start)), int(min(MIN_PER_DAY, end))
    if e > s:
        timeline[s:e] = ROOM_INDEX[room] if room is not None else NONE
        intensity[s:e] = level


def _day_timeline(p: Persona, rng, anomaly=None):
    """Return (room per minute, activity level per minute, anomaly info)."""
    room = np.full(MIN_PER_DAY, ROOM_INDEX["living"])
    act = np.full(MIN_PER_DAY, 0.45)
    info = {"rooms": (), "window": ()}

    wake = p.wake + rng.normal(0, 25)
    sleep = p.sleep + rng.normal(0, 25)
    if anomaly == "late_wake":
        wake += rng.uniform(150, 210)
        info = {"rooms": ("bedroom", "kitchen", "bathroom"),
                "window": (p.wake / 60, wake / 60 + 1)}

    # Night: asleep in bedroom (small movements only)
    _paint(room, act, 0, wake, "bedroom", 0.03)
    if sleep > MIN_PER_DAY:                       # late sleeper: awake after midnight
        _paint(room, act, 0, sleep - MIN_PER_DAY - 15, "living", 0.5)
        _paint(room, act, sleep - MIN_PER_DAY - 15, sleep - MIN_PER_DAY, "bedroom", 0.8)
        _paint(room, act, sleep - MIN_PER_DAY, wake, "bedroom", 0.03)

    # Night toilet trips
    n_trips = rng.poisson(p.night_toilet_rate)
    if anomaly == "disturbed_sleep":
        n_trips = int(rng.integers(6, 10))
        info = {"rooms": ("bathroom", "bedroom"), "window": (0, wake / 60)}
    night_start = max(60, (sleep - MIN_PER_DAY) + 60) if sleep > MIN_PER_DAY else 60
    for _ in range(n_trips):
        t = rng.uniform(night_start, max(night_start + 1, wake - 30))
        _paint(room, act, t, t + 3, "bedroom", 0.9)
        _paint(room, act, t + 3, t + rng.uniform(6, 11), "bathroom", 1.0)

    # Morning routine
    _paint(room, act, wake, wake + 15, "bedroom", 1.0)
    _paint(room, act, wake + 15, wake + 35, "bathroom", 1.0)
    b0 = wake + p.breakfast_delay + rng.normal(0, 8)
    if anomaly == "skipped_meal":
        info = {"rooms": ("kitchen",), "window": (b0 / 60 - 1, b0 / 60 + 2)}
    else:
        _paint(room, act, b0, b0 + rng.uniform(20, 40), "kitchen", 1.0)

    # Outing (person not in the home; only the door sensor sees it)
    went_out = rng.random() < p.p_out and anomaly is None
    if went_out:
        o0 = rng.uniform(9.5, 15) * 60
        o1 = o0 + rng.uniform(80, 180)
        _paint(room, act, o0, o0 + 2, "entrance", 1.0)
        _paint(room, act, o0 + 2, o1, None, 0.0)
        _paint(room, act, o1, o1 + 2, "entrance", 1.0)

    # Meals, nap, evening
    l0 = p.lunch + rng.normal(0, 20)
    if not (went_out and o0 - 30 < l0 < o1):
        _paint(room, act, l0, l0 + rng.uniform(25, 45), "kitchen", 0.9)
    if rng.random() < p.p_nap:
        n0 = rng.uniform(13.8, 15) * 60
        _paint(room, act, n0, n0 + rng.uniform(30, 60), "bedroom", 0.1)
    d0 = p.dinner + rng.normal(0, 20)
    _paint(room, act, d0, d0 + rng.uniform(35, 60), "kitchen", 1.0)
    if sleep < MIN_PER_DAY:
        _paint(room, act, sleep - 30, sleep - 15, "bathroom", 1.0)
        _paint(room, act, sleep - 15, sleep, "bedroom", 0.8)
        _paint(room, act, sleep, MIN_PER_DAY, "bedroom", 0.03)

    # Anomalies that overwrite part of the day
    if anomaly == "long_inactivity":
        s = rng.uniform(9, 14) * 60
        dur = rng.uniform(240, 360)
        _paint(room, act, s, s + dur, None, 0.0)   # at home but not moving
        info = {"rooms": tuple(ROOMS), "window": (s / 60, (s + dur) / 60)}
    if anomaly == "night_wandering":
        for _ in range(int(rng.integers(3, 6))):
            t = rng.uniform(1, 4) * 60
            r = rng.choice(["living", "kitchen", "entrance"])
            _paint(room, act, t, t + rng.uniform(10, 25), r, 1.0)
        info = {"rooms": ("living", "kitchen", "entrance"), "window": (1, 4.5)}

    act = act * p.activity_scale
    return room, act, info


def simulate_day(p: Persona, rng, anomaly=None) -> Day:
    room, act, info = _day_timeline(p, rng, anomaly)
    base_rate = 0.6                                    # firings/min when fully active
    times, rooms, ids = [], [], []
    sensor_no = 0
    for r_name, n_sensors in p.sensors.items():
        r_idx = ROOM_INDEX[r_name]
        here = (room == r_idx).astype(float)
        for _ in range(n_sensors):
            rate = base_rate * act * here + p.noise_rate
            counts = rng.poisson(rate)                 # firings in each minute
            minutes = np.repeat(np.arange(MIN_PER_DAY), counts)
            t = (minutes + rng.random(minutes.size)) * 60.0
            times.append(t)
            rooms.append(np.full(t.size, r_idx))
            ids.append(np.full(t.size, sensor_no))
            sensor_no += 1
    times = np.concatenate(times) if times else np.zeros(0)
    order = np.argsort(times)
    return Day(times=times[order],
               sensor_room=np.concatenate(rooms)[order].astype(int),
               sensor_id=np.concatenate(ids)[order].astype(int),
               anomaly=anomaly, anomaly_rooms=info["rooms"],
               anomaly_window=info["window"])


def simulate_home(p: Persona, rng, n_train, n_val, n_test, n_anom_per_type):
    """Train and validation days are all normal. Test days include unusual days."""
    train = [simulate_day(p, rng) for _ in range(n_train)]
    val = [simulate_day(p, rng) for _ in range(n_val)]
    test_labels = [None] * (n_test - n_anom_per_type * len(ANOMALY_TYPES))
    for a in ANOMALY_TYPES:
        test_labels += [a] * n_anom_per_type
    rng.shuffle(test_labels)
    test = [simulate_day(p, rng, a) for a in test_labels]
    return train, val, test


def sensor_names(p: Persona):
    """CASAS-style names: M001, M002, ... for motion, D001 for the door."""
    names, m = [], 1
    for r_name, n in p.sensors.items():
        for _ in range(n):
            if r_name == "entrance":
                names.append((f"D{m:03d}", r_name))
            else:
                names.append((f"M{m:03d}", r_name))
            m += 1
    return names


def export_casas(p: Persona, days: list, start_date: str, path: str):
    """Write simulated days in CASAS text format (used to test casas_loader.py)."""
    from datetime import datetime, timedelta
    names = sensor_names(p)
    d0 = datetime.strptime(start_date, "%Y-%m-%d")
    with open(path, "w") as f:
        for k, day in enumerate(days):
            base = d0 + timedelta(days=k)
            for t, sid in zip(day.times, day.sensor_id):
                ts = base + timedelta(seconds=float(t))
                name, room = names[sid]
                value = "OPEN" if name.startswith("D") else "ON"
                f.write(f"{ts:%Y-%m-%d %H:%M:%S.%f} {name} {value}\n")
    return {n: r for n, r in names}
