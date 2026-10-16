"""Build live-shift tasks: real EIA-930 days + an RTS-GMLC fleet + seeded trips, scored by policy.

usage: uv run python -m grid_dispatch.build EIA_DIR PGLIB_DIR OUT_JSONL [--tasks 10] [--candidates 30] [--seed 0]

For each candidate it plays five policies through the same simulator:
  rolling   re-plans every step with what is known then           -> the reference
  hindsight one plan knowing the whole real day, trips included      -> the floor
  static    one plan at 00:00 on the day-ahead forecast, never changed
  myopic    re-plans every step but only looks one step ahead
  nothing   keeps the midnight units all day
and keeps the candidate when the static plan grades <= MAX_STATIC_REWARD (adapting matters).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np

from grid_dispatch.data import days, load_eia930, rts_gmlc_units
from grid_dispatch.planner import plan, run
from grid_dispatch.world import Battery, Task, World

BAS = ["PJM", "ISNE", "MISO", "ERCO", "SWPP"]   # CISO: midday forecast bias; NYIS: zero rows
FLEET = {"nuclear": 1, "coal": 2, "cc": 3, "steam": 4, "ct": 7}
PEAK_SHARE = 0.80          # peak net load (the larger of forecast and actual) as a share of fleet capacity
RENEW_FORECAST_ERROR = 0.5
MAX_STATIC_REWARD = 0.9


def grade(cost: float, ref: float, floor: float) -> float:
    g = max(0.0, cost - ref) / (max(0.0, ref - floor) + 0.01 * ref)
    return 0.2 + 0.8 * math.exp(-g / 0.5)


def make_task(day, units_all, rng: random.Random, idx: int) -> Task:
    by_kind = {}
    for u in units_all:
        by_kind.setdefault(u.kind, []).append(u)
    units = []
    for kind, n in FLEET.items():
        pool = by_kind[kind]
        if kind == "ct":   # a spread of peaker sizes
            pool = sorted(pool, key=lambda u: (-u.pmax, u.id))
            pick = rng.sample(pool[:20], 3) + rng.sample(pool[20:], n - 3)
        else:
            pick = rng.sample(pool, min(n, len(pool)))
        units += [copy.deepcopy(u) for u in pick]
    for i, u in enumerate(units):
        u.id = f"{u.kind.upper()}{i + 1:02d}"
        for k, v in u.__dict__.items():
            if isinstance(v, float):
                setattr(u, k, round(v, 3))
        u.curve = [(round(a, 3), round(c, 3)) for a, c in u.curve]
    cap = sum(u.pmax for u in units)
    renew = day.wind + day.solar
    # Renewables forecast (synthetic): the actual plus half the error of a persistence forecast
    # (yesterday's output). Pure persistence misses far more than real day-ahead wind forecasts.
    renew_da = renew + RENEW_FORECAST_ERROR * (day.wind_prev + day.solar_prev - renew)
    peak = max(float(np.max(day.forecast - renew_da)), float(np.max(day.demand - renew)))
    scale = PEAK_SHARE * cap / peak
    big = [u for u in units if u.kind != "ct"]
    n_trips = rng.choices([0, 1, 2], weights=[0.25, 0.55, 0.2])[0]
    trips = {}
    for u in rng.sample(big, n_trips):
        trips[u.id] = rng.randint(6, 19)
    r1 = lambda x: np.round(x * scale, 1)   # saved tasks hold exactly the numbers that were simulated
    task = Task(
        id=f"{day.ba}-{day.date}-s{idx}", ba=day.ba, date=day.date, units=units,
        battery=Battery(power=round(0.06 * cap), energy=round(0.24 * cap)),
        demand=r1(day.demand), demand_da=r1(day.forecast), renew=r1(renew), renew_da=r1(renew_da),
        trips=trips, init_on={u.id: False for u in units}, init_p={u.id: 0.0 for u in units})
    # midnight state: hour 0 of a day-ahead plan made without notice limits
    w = World(task)
    sched, _, _ = plan(w, task.demand_da, task.renew_da, ignore_lead=True)
    probe = World(task)
    for uid, x in sched.items():
        probe.sched[uid] = x
    probe.advance(1)
    task.init_on = {u.id: bool(sched[u.id][0]) for u in units}
    task.init_p = {u.id: round(float(probe.p[u.id][0]), 3) for u in units}
    return task


def to_row(task: Task, results: dict) -> dict:
    u2d = lambda u: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in u.__dict__.items()}
    # The floor is the hindsight plan's cost; if any policy came out cheaper (the simulator and the
    # solver round ramps slightly differently), that cheaper cost is the floor.
    ref, floor = results["rolling"]["cost"], min(v["cost"] for v in results.values())
    return {
        "id": task.id,
        "world": {
            "ba": task.ba, "date": task.date, "step": task.step,
            "units": [u2d(u) for u in task.units], "battery": task.battery.__dict__,
            "demand": [round(x, 1) for x in task.demand], "demand_da": [round(x, 1) for x in task.demand_da],
            "renew": [round(x, 1) for x in task.renew], "renew_da": [round(x, 1) for x in task.renew_da],
            "trips": task.trips, "init_on": task.init_on, "init_p": {k: round(v, 3) for k, v in task.init_p.items()},
        },
        "reference": {
            "cost": round(ref, 2), "floor": round(floor, 2),
            "baselines": {k: round(v["cost"], 2) for k, v in results.items()},
            "scores": {k: round(grade(v["cost"], ref, floor), 4) for k, v in results.items()},
            "shed_mwh": {k: round(v["shed_mwh"], 1) for k, v in results.items()},
            "starts": {k: v["starts"] for k, v in results.items()},
            "solve_s": {k: round(v["solve_s"], 2) for k, v in results.items()},
            "rolling_actions": results["rolling"]["steps"],   # hidden; the oracle replays these
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eia_dir", type=Path)
    ap.add_argument("pglib_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--candidates", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    df = load_eia930(sorted(args.eia_dir.glob("EIA930_BALANCE_*.csv")), BAS)
    all_days = {ba: days(df, ba) for ba in BAS}
    units_all = rts_gmlc_units(sorted((args.pglib_dir / "rts_gmlc").glob("*.json"))[0])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    with args.out.open("w") as out:
        for i in range(args.candidates):
            if kept >= args.tasks:
                break
            ba = BAS[i % len(BAS)]
            day = rng.choice(all_days[ba])
            t0 = time.perf_counter()
            task = make_task(day, units_all, rng, i)
            results = {p: run(World(task), p) for p in ["rolling", "hindsight", "static", "myopic", "nothing"]}
            row = to_row(task, results)
            sc, ref = row["reference"]["scores"], row["reference"]
            keep = sc["static"] <= MAX_STATIC_REWARD
            print(f"{task.id:24} trips={len(task.trips)} cost: rolling {ref['cost']:>10,.0f} hindsight {ref['floor']:>10,.0f} "
                  f"| score static {sc['static']:.2f} myopic {sc['myopic']:.2f} nothing {sc['nothing']:.2f} "
                  f"| shed MWh {ref['shed_mwh']} | {time.perf_counter() - t0:5.1f}s {'KEEP' if keep else 'drop'}", flush=True)
            if keep:
                out.write(json.dumps(row) + "\n")
                out.flush()
                kept += 1
    print(f"kept {kept} tasks in {args.out}")


if __name__ == "__main__":
    main()
