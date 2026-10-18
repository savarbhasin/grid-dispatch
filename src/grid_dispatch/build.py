"""Build live-shift tasks: real EIA-930 days + an RTS-GMLC fleet + seeded trips, scored by policy.

usage: uv run python -m grid_dispatch.build EIA_DIR PGLIB_DIR OUT_DIR [--tasks 50] [--candidates 400]
           [--jobs 3] [--seed 0]

Each candidate is one real day with a sampled fleet and seeded trips, played by six policies through
the same simulator:
  rolling   re-plans every step with what is known then, keeping N-1 reserve
  rolling2  the same with N-2 reserve (enough for any two large units to trip)
  hindsight one plan knowing the whole real day, trips included (for reference only)
  static    one plan at 00:00 on the day-ahead forecast, never changed
  myopic    re-plans every step but only looks one step ahead
  nothing   keeps the midnight units all day
The reference is whichever of rolling and rolling2 shed less on the day (then the cheaper). Every
policy is scored against it with grading.score. A candidate is kept when the naive policies (static
and nothing) both score <= MAX_NAIVE_SCORE, so adapting matters, and the reference shed at most
MAX_REF_SHED of the day's demand, so the day is winnable.

Days are chosen to be hard: windier grid areas are drawn more often, days are drawn from those
where actual net load ran furthest above the forecast, the fleet is tight, and trips favour the hours
before the evening peak.

Stop any time: `touch OUT_DIR/STOP` lets running candidates finish; a re-run resumes. Kept tasks go
to OUT_DIR/tasks.jsonl and every decision to OUT_DIR/candidates.jsonl the moment it is made.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

from grid_dispatch.data import days, load_eia930, rts_gmlc_units
from grid_dispatch.grading import score
from grid_dispatch.planner import plan, run
from grid_dispatch.world import Battery, Task, World

BAS = {"SWPP": 3, "ERCO": 3, "MISO": 2, "PJM": 2, "ISNE": 1}   # draw weights; CISO and NYIS excluded
FLEET = {"nuclear": 1, "coal": 2, "cc": 3, "steam": 4, "ct": 7}
PEAK_SHARE = 0.85          # peak net load (the larger of forecast and actual) as a share of fleet capacity
RENEW_FORECAST_ERROR = 0.5
SURPRISE_SHARE = 0.4       # days are drawn from the top 40% by how far actual net load beat the forecast
TRIPS = ([0, 1, 2], [0.1, 0.5, 0.4])
PRE_PEAK_TRIP = 0.7        # chance a trip lands in the 6 hours before the forecast peak
MAX_NAIVE_SCORE = 0.6
MAX_REF_SHED = 0.01
POLICIES = ["rolling", "rolling2", "hindsight", "static", "myopic", "nothing"]


def renew_forecast(day) -> np.ndarray:
    """Synthetic: the actual plus half the error of a persistence forecast (yesterday's output);
    pure persistence misses far more than real day-ahead wind forecasts."""
    renew = day.wind + day.solar
    return renew + RENEW_FORECAST_ERROR * (day.wind_prev + day.solar_prev - renew)


def surprise(day) -> float:
    """How far actual net load ran above the day-ahead forecast at worst, as a share of its peak."""
    net, net_fc = day.demand - (day.wind + day.solar), day.forecast - renew_forecast(day)
    return float(np.max(net - net_fc) / np.max(net))


def make_task(day, units_all, rng: random.Random, tid: str) -> Task:
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
    renew, renew_da = day.wind + day.solar, renew_forecast(day)
    peak = max(float(np.max(day.forecast - renew_da)), float(np.max(day.demand - renew)))
    scale = PEAK_SHARE * cap / peak
    peak_hour = int(np.argmax(day.forecast - renew_da))
    big = [u for u in units if u.kind != "ct"]
    trips = {}
    for u in rng.sample(big, rng.choices(*TRIPS)[0]):
        lo = max(3, peak_hour - 6)
        pre = rng.random() < PRE_PEAK_TRIP and peak_hour - 1 >= lo
        trips[u.id] = rng.randint(lo, peak_hour - 1) if pre else rng.randint(4, 20)
    r1 = lambda x: np.round(x * scale, 1)   # saved tasks hold exactly the numbers that were simulated
    task = Task(
        id=tid, ba=day.ba, date=day.date, units=units,
        battery=Battery(power=round(0.06 * cap), energy=round(0.24 * cap)),
        demand=r1(day.demand), demand_da=r1(day.forecast), renew=r1(renew), renew_da=r1(renew_da),
        trips=trips, init_on={u.id: False for u in units}, init_p={u.id: 0.0 for u in units})
    # midnight state: hour 0 of a day-ahead plan made without notice limits
    sched, _, _ = plan(World(task), task.demand_da, task.renew_da, ignore_lead=True)
    probe = World(task)
    for uid, x in sched.items():
        probe.sched[uid] = x
    probe.advance(1)
    task.init_on = {u.id: bool(sched[u.id][0]) for u in units}
    task.init_p = {u.id: round(float(probe.p[u.id][0]), 3) for u in units}
    return task


def evaluate(task: Task) -> dict:
    """Play every policy; pick the reference; score everything against it."""
    results = {p: run(World(task), p) for p in POLICIES}
    ref_name = min(["rolling", "rolling2"], key=lambda p: (round(results[p]["shed_mwh"]), results[p]["cost"]))
    ref = results[ref_name]
    demand = float(np.sum(task.demand))
    baselines = {}
    for p, r in results.items():
        sc, rel, eco = score(r["cost"], r["shed_mwh"], ref["cost"], ref["shed_mwh"], demand)
        baselines[p] = {"cost": round(r["cost"], 2), "shed_mwh": round(r["shed_mwh"], 1), "starts": r["starts"],
                        "score": round(sc, 4), "R": round(rel, 4), "E": round(eco, 4), "solve_s": round(r["solve_s"], 2)}
    return {"policy": ref_name, "cost": round(ref["cost"], 2), "shed_mwh": round(ref["shed_mwh"], 1),
            "demand_mwh": round(demand, 1), "baselines": baselines,
            "actions": ref["steps"]}   # hidden; the oracle replays these


def to_world(task: Task) -> dict:
    u2d = lambda u: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in u.__dict__.items()}
    return {"ba": task.ba, "date": task.date, "step": task.step,
            "units": [u2d(u) for u in task.units], "battery": task.battery.__dict__,
            "demand": [round(x, 1) for x in task.demand], "demand_da": [round(x, 1) for x in task.demand_da],
            "renew": [round(x, 1) for x in task.renew], "renew_da": [round(x, 1) for x in task.renew_da],
            "trips": task.trips, "init_on": task.init_on, "init_p": {k: round(v, 3) for k, v in task.init_p.items()}}


# ------------------------------------------------------------------ worker side
_DAYS: dict = {}
_UNITS: list = []


def _init(eia_dir: str, pglib_dir: str, seed: int):
    global _DAYS, _UNITS
    df = load_eia930(sorted(Path(eia_dir).glob("EIA930_BALANCE_*.csv")), list(BAS))
    for ba in BAS:
        ds = sorted(days(df, ba), key=surprise, reverse=True)
        _DAYS[ba] = ds[: max(1, int(len(ds) * SURPRISE_SHARE))]
    _UNITS = rts_gmlc_units(sorted((Path(pglib_dir) / "rts_gmlc").glob("*.json"))[0])


def candidate(i: int, seed: int) -> dict:
    t0 = time.perf_counter()
    rng = random.Random(f"{seed}:{i}")
    ba = rng.choices(list(BAS), weights=list(BAS.values()))[0]
    day = rng.choice(_DAYS[ba])
    task = make_task(day, _UNITS, rng, f"{ba}-{day.date}-c{i}")
    ref = evaluate(task)
    b = ref["baselines"]
    naive = max(b["static"]["score"], b["nothing"]["score"])
    winnable = ref["shed_mwh"] <= MAX_REF_SHED * ref["demand_mwh"]
    keep = naive <= MAX_NAIVE_SCORE and winnable
    reason = "kept" if keep else ("naive policy scores " + f"{naive:.2f}" if winnable else
                                  f"reference sheds {ref['shed_mwh']:.0f} MWh")
    row = {"id": task.id, "world": to_world(task), "reference": ref}
    summary = {"i": i, "id": task.id, "keep": keep, "reason": reason, "trips": task.trips,
               "reference": ref["policy"], "ref_shed": ref["shed_mwh"],
               "scores": {p: v["score"] for p, v in b.items()}, "seconds": round(time.perf_counter() - t0, 1)}
    return {"row": row if keep else None, "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("eia_dir", type=Path)
    ap.add_argument("pglib_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--candidates", type=int, default=400)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    tasks_f, cands_f, stop = out / "tasks.jsonl", out / "candidates.jsonl", out / "STOP"
    done, kept = set(), 0
    if cands_f.exists():
        for line in cands_f.read_text().splitlines():
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add(s["i"])
            kept += s["keep"]
    todo = [i for i in range(args.candidates) if i not in done]
    print(f"resume: {len(done)} candidates done, {kept} kept; {len(todo)} left; stop with: touch {stop}", flush=True)
    t_start, inflight = time.perf_counter(), {}
    with ProcessPoolExecutor(args.jobs, initializer=_init, initargs=(str(args.eia_dir), str(args.pglib_dir), args.seed)) as pool:
        def submit() -> bool:
            if stop.exists() or not todo or kept >= args.tasks:
                return False
            i = todo.pop(0)
            inflight[pool.submit(candidate, i, args.seed)] = i
            return True

        while len(inflight) < args.jobs and submit():
            pass
        while inflight:
            finished, _ = wait(inflight, return_when=FIRST_COMPLETED)
            for fut in finished:
                i = inflight.pop(fut)
                try:
                    res = fut.result()
                except Exception as e:   # one broken candidate must not stop the build
                    res = {"row": None, "summary": {"i": i, "keep": False, "reason": f"error: {e!r}"[:300]}}
                s = res["summary"]
                if res["row"] is not None and kept < args.tasks:
                    with tasks_f.open("a") as f:
                        f.write(json.dumps(res["row"]) + "\n")
                    kept += 1
                elif s["keep"]:
                    s |= {"keep": False, "reason": "enough tasks already"}
                with cands_f.open("a") as f:
                    f.write(json.dumps(s) + "\n")
                sc = s.get("scores", {})
                print(f"[{(time.perf_counter() - t_start) / 60:5.1f} min] #{i:<4} {s.get('id', ''):24} trips {len(s.get('trips', {}))} "
                      f"ref {s.get('reference', '-'):8} shed {s.get('ref_shed', 0):5.0f} | "
                      + " ".join(f"{p} {sc[p]:.2f}" for p in ("static", "myopic", "nothing") if p in sc)
                      + f" | {s.get('seconds', '?')}s {'KEEP' if s['keep'] else 'drop: ' + s['reason']} | kept {kept}/{args.tasks}", flush=True)
            while len(inflight) < args.jobs and submit():
                pass
    print(("stopped by STOP file" if stop.exists() else "finished") + f": {kept} tasks in {tasks_f}", flush=True)


if __name__ == "__main__":
    main()
