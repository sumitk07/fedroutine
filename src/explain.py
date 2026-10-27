"""
Explain an alert in plain words, based on the person's own normal days.

Steps:
  1. The model's rebuilt day is what it expected for this person.
  2. The squared difference between today and the expected day shows where the
     day is unusual. We add it up in blocks of (room x 2 hours).
  3. The block that contributes most is described in one sentence, using the
     person's own average from their normal days as "usually".

A language model can later rephrase this sentence (for example into Japanese),
but it must only use these facts; it is not allowed to add new ones.

We also test whether the explanation is faithful with a deletion test: if we
replace the highlighted block with the expected values, the anomaly score
should drop much more than if we replace a random block of the same size.
"""

import numpy as np

from .features import N_ROOMS, N_SLOTS, SLOT_MIN
from .simulate import ROOMS

BLOCK_SLOTS = 4                     # 4 x 30 min = 2-hour blocks
N_BLOCKS = N_SLOTS // BLOCK_SLOTS


def block_errors(x, recon, cmask):
    err = ((x - recon) ** 2 * cmask).reshape(N_ROOMS, N_BLOCKS, BLOCK_SLOTS)
    return err.sum(axis=2)          # rooms x blocks


def _hhmm(slot):
    m = slot * SLOT_MIN
    return f"{m // 60:02d}:{m % 60:02d}"


def explain_day(x, recon, cmask, counts_today, baseline_counts, top_k=2):
    """Return a list of explanation dicts, most important first."""
    be = block_errors(x, recon, cmask)
    total = be.sum() + 1e-12
    order = np.argsort(be, axis=None)[::-1][:top_k]
    out = []
    for flat in order:
        r, b = np.unravel_index(flat, be.shape)
        s0, s1 = b * BLOCK_SLOTS, (b + 1) * BLOCK_SLOTS
        today = counts_today[r, s0:s1].sum()
        usual = baseline_counts[r, s0:s1].sum()
        direction = "less" if today < usual else "more"
        out.append({
            "room": ROOMS[r], "start_hour": s0 * SLOT_MIN / 60,
            "end_hour": s1 * SLOT_MIN / 60, "share": be[r, b] / total,
            "today": int(today), "usual": float(usual), "direction": direction,
            "text": (f"{ROOMS[r].capitalize()}: {direction} activity than usual "
                     f"between {_hhmm(s0)} and {_hhmm(s1)} "
                     f"({int(today)} sensor event{'' if int(today) == 1 else 's'} today, "
                     f"usually about {usual:.0f})."),
        })
    return out


def explanation_hit(expl, day):
    """Does the top explanation point at the room and time of the injected event?"""
    if not day.anomaly:
        return None
    e = expl[0]
    a0, a1 = day.anomaly_window
    time_ok = e["start_hour"] < a1 + 1 and e["end_hour"] > a0 - 1
    room_ok = e["room"] in day.anomaly_rooms
    return bool(time_ok and room_ok)


def deletion_test(x, recon, cmask, rng, n_random=20):
    """Score drop when the top block is replaced vs. when a random block is."""
    def score(v):
        return float(((v - recon) ** 2 * cmask).sum() / max(cmask.sum(), 1))

    base = score(x)
    if base <= 0:
        return 0.0, 0.0
    be = block_errors(x, recon, cmask)
    r, b = np.unravel_index(np.argmax(be), be.shape)

    def replaced(r_, b_):
        v = x.copy().reshape(N_ROOMS, N_SLOTS)
        rc = recon.reshape(N_ROOMS, N_SLOTS)
        v[r_, b_ * BLOCK_SLOTS:(b_ + 1) * BLOCK_SLOTS] = rc[r_, b_ * BLOCK_SLOTS:(b_ + 1) * BLOCK_SLOTS]
        return v.reshape(-1)

    top_drop = 1 - score(replaced(r, b)) / base
    present = np.where(cmask.reshape(N_ROOMS, N_SLOTS)[:, 0] > 0)[0]
    rand = [1 - score(replaced(rng.choice(present), rng.integers(N_BLOCKS))) / base
            for _ in range(n_random)]
    return top_drop, float(np.mean(rand))
