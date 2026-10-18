"""Unit-commitment MILP over the rest of the day (HiGHS through OR-Tools), and the policies built on it.

Every policy acts on a World only through `set_schedule` and `set_battery`, the same calls the
agent's tools make, so each policy's cost comes from the same simulator and its actions can be
replayed exactly.
"""

from __future__ import annotations

import time

import numpy as np
from ortools.linear_solver import pywraplp

from grid_dispatch.world import DUMP, H, RESERVE, VOLL, World

RESERVE_PENALTY = 500.0   # $/MW-h of reserve shortfall, in the solver's objective only
# Spinning reserve the solver keeps each hour: RESERVE x forecast demand, and enough to cover the
# output of the largest running unit (reserve_n=1, the "N-1" rule operators use) or of any two of the
# N2_UNITS largest units (reserve_n=2). Pairs over the whole fleet made the solve time out; only big
# units are given trips, so the largest few are the pairs that matter. reserve_n=0 keeps none.
N2_UNITS = 6


class PlanError(RuntimeError):
    pass


def plan(world: World, demand: np.ndarray, renew: np.ndarray, end: int = H, trips: dict[str, int] | None = None,
         ignore_lead: bool = False, time_limit: float = 120.0, reserve_n: int = 1) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """Cheapest commitment for hours [now, end) given forecasts. Returns full-day schedules
    (hours before now copied, hours after `end` holding the last state), a battery plan and
    solver stats."""
    t0 = time.perf_counter()
    t, b, now = world.task, world.task.battery, world.now
    trips = world.known_trips() if trips is None else trips
    hrs = list(range(now, end))
    s = pywraplp.Solver.CreateSolver("HIGHS")
    s.SetNumThreads(1)
    # a 0.1% optimality gap: far below the differences between policies, and much faster under load
    s.SetSolverSpecificParametersAsString("output_flag=false\nmip_rel_gap=0.001")
    s.SetTimeLimit(int(time_limit * 1000))
    inf = s.infinity()
    obj = s.Objective()
    U, P, R = {}, {}, {}
    total_p = {h: [] for h in hrs}
    spare = {h: [] for h in hrs}
    for u in t.units:
        state, n, p_prev = world.status_before(u.id)
        old = world.sched[u.id]
        trip = trips.get(u.id)
        uu = {h: s.BoolVar("") for h in hrs}
        vv = {h: s.BoolVar("") for h in hrs}
        ww = {h: s.BoolVar("") for h in hrs}
        pp = {h: s.NumVar(0, u.pmax, "") for h in hrs}
        U[u.id], P[u.id] = uu, pp
        for h in hrs:
            prev_u = uu[h - 1] if h > now else int(state)
            s.Add(uu[h] - prev_u == vv[h] - ww[h])
            # minimum up and down times: runs that start in the plan, and the run already going
            s.Add(sum(vv[k] for k in range(max(now, h - u.min_up + 1), h + 1)) <= uu[h])
            s.Add(sum(ww[k] for k in range(max(now, h - u.min_down + 1), h + 1)) <= 1 - uu[h])
            if state and h < now + u.min_up - n and (trip is None or h < trip):
                s.Add(uu[h] == 1)
            if not state and h < now + u.min_down - n:
                s.Add(uu[h] == 0)
            if trip is not None and h >= trip:
                s.Add(uu[h] == 0)
            if not ignore_lead and h < now + u.lead:
                was = old[h - 1] if h > now else int(state)
                s.Add(vv[h] <= int(bool(old[h] and not was)))
            # output limits, start-up and shut-down ramps
            s.Add(pp[h] >= u.pmin * uu[h])
            s.Add(pp[h] <= u.pmax * uu[h] - max(0.0, u.pmax - u.ramp_su) * vv[h])
            if h + 1 < end:
                s.Add(pp[h] <= u.pmax * uu[h] - max(0.0, u.pmax - u.ramp_sd) * ww[h + 1])
            prev_p = pp[h - 1] if h > now else (p_prev if state else 0.0)
            s.Add(pp[h] - prev_p <= u.ramp * prev_u + u.ramp_su * vv[h])
            s.Add(prev_p - pp[h] <= u.ramp * uu[h] + u.pmax * ww[h])
            # piecewise cost: cost at pmin when on, plus a slope per segment
            segs = []
            for (a, ca), (c, cc) in zip(u.curve, u.curve[1:]):
                v = s.NumVar(0, max(0.0, c - a), "")
                s.Add(v <= (c - a) * uu[h])
                obj.SetCoefficient(v, (cc - ca) / (c - a) if c > a else 0)
                segs.append(v)
            s.Add(pp[h] == u.pmin * uu[h] + sum(segs))
            obj.SetCoefficient(uu[h], u.curve[0][1])
            obj.SetCoefficient(vv[h], u.startup_cost)
            total_p[h].append(pp[h])
            # reserve a unit can really give within the hour: spare capacity, capped by its ramp
            r = s.NumVar(0, u.pmax, "")
            s.Add(r <= u.pmax * uu[h] - pp[h])
            s.Add(r <= u.ramp * uu[h])
            spare[h].append(r)
            R.setdefault(u.id, {})[h] = r
    soc_prev = world.soc
    bat = {}
    for h in hrs:
        c, d = s.NumVar(0, b.power, ""), s.NumVar(0, b.power, "")
        soc = s.NumVar(0, b.energy, "")
        s.Add(soc == soc_prev + b.eff * c - d * (1 / b.eff))
        soc_prev = soc
        ren = s.NumVar(0, float(renew[h]), "")
        shed, dump, rshort = s.NumVar(0, inf, ""), s.NumVar(0, inf, ""), s.NumVar(0, inf, "")
        obj.SetCoefficient(shed, VOLL)
        obj.SetCoefficient(dump, DUMP)
        obj.SetCoefficient(rshort, RESERVE_PENALTY)
        s.Add(sum(total_p[h]) + ren + d - c + shed - dump == float(demand[h]))
        if reserve_n >= 1:
            s.Add(sum(spare[h]) + rshort >= RESERVE * float(demand[h]))
            for u in t.units:   # N-1: the others can cover this unit's output if it trips
                s.Add(sum(spare[h]) - R[u.id][h] + rshort >= P[u.id][h])
        if reserve_n >= 2:
            big = sorted(t.units, key=lambda u: -u.pmax)[:N2_UNITS]
            for i, a in enumerate(big):   # N-2: ... and any two of the largest together
                for c2 in big[i + 1:]:
                    s.Add(sum(spare[h]) - R[a.id][h] - R[c2.id][h] + rshort >= P[a.id][h] + P[c2.id][h])
        bat[h] = (c, d)
    obj.SetMinimization()
    status = s.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise PlanError(f"solver status {status} after {time.perf_counter() - t0:.0f}s")
    sched = {}
    for u in t.units:
        x = world.sched[u.id].copy()
        for h in hrs:
            x[h] = int(round(U[u.id][h].solution_value()))
        if end < H:
            x[end:] = x[end - 1] if end > now else x[end:]
        trip = trips.get(u.id)
        if trip is not None and trip > now:   # hindsight: let the trip itself end the run
            x[trip:] = x[trip - 1]
        sched[u.id] = x
    battery = world.battery_plan.copy()
    for h in hrs:
        c, d = bat[h]
        battery[h] = d.solution_value() - c.solution_value()
    stats = {"seconds": time.perf_counter() - t0, "optimal": status == pywraplp.Solver.OPTIMAL,
             "objective": obj.Value(), "bound": obj.BestBound()}
    return sched, battery, stats


def apply(world: World, sched: dict[str, np.ndarray], battery: np.ndarray) -> list[dict]:
    """Apply a plan through the same calls the agent's tools use; returns the actions taken."""
    actions = []
    for uid, x in sched.items():
        if not np.array_equal(x[world.now:], world.sched[uid][world.now:]):
            errs = world.set_schedule(uid, x)
            assert not errs, (uid, errs)
            actions.append({"unit": uid, "from": world.now, "schedule": [int(v) for v in x[world.now:]]})
    world.set_battery(battery)
    actions.append({"battery": [round(float(v), 3) for v in world.battery_plan[world.now:]]})
    return actions


# ------------------------------------------------------------------ policies
def run(world: World, policy: str) -> dict:
    """Play a whole day. Returns cost breakdown, actions per step and solver stats."""
    t = world.task
    steps, solves = [], []
    while world.now < H:
        acts = []
        if policy in ("rolling", "rolling2", "myopic") or (policy == "static" and world.now == 0):
            dem, ren = t.forecast(world.now)
            end = H if policy != "myopic" else min(H, world.now + t.step)
            try:
                sched, bat, st = plan(world, dem, ren, end=end, reserve_n=2 if policy == "rolling2" else 1)
            except PlanError:   # out of time with the cautious reserve: this step plans with N-1
                if policy != "rolling2":
                    raise
                sched, bat, st = plan(world, dem, ren, end=end, reserve_n=1)
                st["fallback"] = True
            solves.append(st)
            acts = apply(world, sched, bat)
        elif policy == "hindsight" and world.now == 0:
            sched, bat, st = plan(world, t.demand, t.renew, trips=t.trips, reserve_n=0)
            solves.append(st)
            acts = apply(world, sched, bat)
        steps.append({"at": world.now, "actions": acts})
        world.advance(t.step)
    return {"cost": world.total(), "parts": dict(world.cost), "shed_mwh": world.shed_mwh, "starts": world.starts,
            "steps": steps, "solve_s": sum(x["seconds"] for x in solves),
            "all_optimal": all(x["optimal"] for x in solves)}
