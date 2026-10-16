"""The live shift: a seeded real day that runs hour by hour while a dispatcher (agent or solver) acts.

Time is in whole hours 0..23. At a step boundary `now`, a policy may change unit orders
(the on/off schedule for hours >= now) and the battery plan, then `advance` runs the next
hours: real demand and renewables are revealed, committed units are dispatched at least cost,
shortfall is shed at VOLL, and seeded trips happen.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from ortools.linear_solver import pywraplp

from grid_dispatch.data import Unit

H = 24
VOLL = 5000.0          # $/MWh of unserved demand
DUMP = 1000.0          # $/MWh of thermal output that had nowhere to go
RESERVE = 0.03         # spinning reserve the solver keeps, share of forecast demand
FORECAST_HORIZON = 6   # hours over which a forecast converges to the actual


@dataclass
class Battery:
    power: float       # MW, charge and discharge
    energy: float      # MWh
    eff: float = 0.96  # one-way efficiency
    soc0: float = 0.5  # starting charge, share of energy


@dataclass
class Task:
    id: str
    ba: str
    date: str
    units: list[Unit]
    battery: Battery
    demand: np.ndarray
    demand_da: np.ndarray        # real day-ahead forecast
    renew: np.ndarray            # actual wind + solar
    renew_da: np.ndarray         # persistence forecast (the day before)
    trips: dict[str, int]        # unit id -> hour it trips (hidden until it happens)
    init_on: dict[str, bool]
    init_p: dict[str, float]
    step: int = 4

    def forecast(self, now: int) -> tuple[np.ndarray, np.ndarray]:
        """Demand and renewables as known at `now`: actuals before now, and forecasts that
        move from the day-ahead value toward the actual as the hour approaches."""
        w = np.clip((np.arange(H) - now) / FORECAST_HORIZON, 0, 1)
        return self.demand + w * (self.demand_da - self.demand), self.renew + w * (self.renew_da - self.renew)


@dataclass
class World:
    task: Task
    now: int = 0
    sched: dict[str, np.ndarray] = field(default_factory=dict)   # planned on/off per hour
    battery_plan: np.ndarray = field(default_factory=lambda: np.zeros(H))   # + discharge, - charge
    on: dict[str, np.ndarray] = field(default_factory=dict)      # realised
    p: dict[str, np.ndarray] = field(default_factory=dict)
    soc: float = 0.0
    log: list[dict] = field(default_factory=list)
    cost: dict[str, float] = field(default_factory=lambda: {"fuel": 0.0, "startup": 0.0, "shed": 0.0, "dump": 0.0})
    shed_mwh: float = 0.0
    starts: int = 0

    def __post_init__(self):
        t = self.task
        for u in t.units:
            self.sched[u.id] = np.full(H, 1 if t.init_on[u.id] else 0)
            self.on[u.id] = np.zeros(H, int)
            self.p[u.id] = np.zeros(H)
        self.soc = t.battery.soc0 * t.battery.energy

    # ------------------------------------------------------------- what is known now
    def unit(self, uid: str) -> Unit:
        return next(u for u in self.task.units if u.id == uid)

    def known_trips(self) -> dict[str, int]:
        return {k: h for k, h in self.task.trips.items() if h < self.now}

    def status_before(self, uid: str) -> tuple[bool, int, float]:
        """(on at now-1, hours in that state up to now, output at now-1)."""
        t = self.task
        hist = [bool(x) for x in self.on[uid][: self.now]]
        state = hist[-1] if hist else t.init_on[uid]
        n = 0
        for x in reversed(hist):
            if x != state:
                break
            n += 1
        u = self.unit(uid)
        if n == len(hist):   # unchanged since midnight: treat the earlier history as long enough
            n += max(u.min_up, u.min_down)
        p = self.p[uid][self.now - 1] if self.now else t.init_p[uid]
        return state, n, p

    # ------------------------------------------------------------- acting
    def check_schedule(self, uid: str, new: np.ndarray) -> list[str]:
        """Rule breaks if unit `uid` followed `new` (hours >= now; earlier hours ignored)."""
        u, errs = self.unit(uid), []
        state, n, _ = self.status_before(uid)
        old = self.sched[uid]
        trip = self.known_trips().get(uid)
        prev = state
        run = n
        for h in range(self.now, H):
            x = bool(new[h])
            if trip is not None and h >= trip and x:
                errs.append(f"{uid} tripped at {trip:02d}:00 and is out for the day")
                break
            if x and not prev:
                if run < u.min_down:
                    errs.append(f"{uid} must stay off at least {u.min_down} h; it can start at {h + u.min_down - run:02d}:00 at the earliest")
                    break
                if h < self.now + u.lead and not (old[h] and not (old[h - 1] if h > self.now else state)):
                    errs.append(f"{uid} needs {u.lead} h notice to start; the earliest new start is {self.now + u.lead:02d}:00")
                    break
            if not x and prev and run < u.min_up:
                errs.append(f"{uid} must run at least {u.min_up} h once started; it can stop at {h + u.min_up - run:02d}:00 at the earliest")
                break
            run = run + 1 if x == prev else 1
            prev = x
        return errs

    def set_schedule(self, uid: str, new: np.ndarray) -> list[str]:
        errs = self.check_schedule(uid, new)
        if not errs:
            self.sched[uid] = np.concatenate([self.sched[uid][: self.now], np.asarray(new[self.now:], int)])
        return errs

    def set_battery(self, plan: np.ndarray) -> None:
        b = self.task.battery
        self.battery_plan = np.concatenate([self.battery_plan[: self.now], np.round(np.clip(plan[self.now:], -b.power, b.power), 3)])

    # ------------------------------------------------------------- the world runs
    def advance(self, hours: int) -> dict:
        t, report = self.task, {"from": self.now, "hours": []}
        for h in range(self.now, min(H, self.now + hours)):
            for uid, hr in t.trips.items():
                if hr == h:
                    self.sched[uid][h:] = 0
                    report.setdefault("trips", []).append(uid)
            report["hours"].append(self._dispatch(h))
        self.now = min(H, self.now + hours)
        report["to"] = self.now
        self.log.append(report)
        return report

    def _dispatch(self, h: int) -> dict:
        t, b = self.task, self.task.battery
        s = pywraplp.Solver.CreateSolver("GLOP")
        obj = s.Objective()
        units = [u for u in t.units if self.sched[u.id][h]]
        pv = {}
        for u in units:
            prev_on = self.on[u.id][h - 1] if h else t.init_on[u.id]
            prev_p = self.p[u.id][h - 1] if h else t.init_p[u.id]
            lo, hi = u.pmin, u.pmax
            if prev_on:
                lo, hi = max(lo, prev_p - u.ramp), min(hi, prev_p + u.ramp)
            else:
                hi = min(hi, u.ramp_su)
            if h + 1 < H and not self.sched[u.id][h + 1]:
                hi = min(hi, u.ramp_sd)
            hi = max(hi, u.pmin)
            lo = min(lo, hi)
            segs = []
            for (a, ca), (c, cc) in zip(u.curve, u.curve[1:]):
                v = s.NumVar(0, max(0.0, c - a), "")
                obj.SetCoefficient(v, (cc - ca) / (c - a) if c > a else 0)
                segs.append(v)
            p = s.NumVar(lo, hi, "")
            s.Add(p == u.pmin + sum(segs))
            pv[u.id] = p
        # battery: follow the plan as far as the charge allows
        bp = float(self.battery_plan[h])
        bp = min(bp, self.soc * b.eff) if bp > 0 else max(bp, -(b.energy - self.soc) / b.eff)
        ren = s.NumVar(0, float(t.renew[h]), "")
        shed = s.NumVar(0, s.infinity(), "")
        dump = s.NumVar(0, s.infinity(), "")
        obj.SetCoefficient(shed, VOLL)
        obj.SetCoefficient(dump, DUMP)
        obj.SetCoefficient(ren, -1e-3)   # use renewables before anything else
        s.Add(sum(pv.values()) + ren + bp + shed - dump == float(t.demand[h]))
        obj.SetMinimization()
        assert s.Solve() == pywraplp.Solver.OPTIMAL
        fuel = 0.0
        for u in t.units:
            on = int(u.id in pv)
            self.on[u.id][h] = on
            if on:
                self.p[u.id][h] = pv[u.id].solution_value()
                fuel += u.cost(self.p[u.id][h])
                was = self.on[u.id][h - 1] if h else t.init_on[u.id]
                if not was:
                    self.cost["startup"] += u.startup_cost
                    self.starts += 1
        self.soc -= bp / b.eff if bp > 0 else bp * b.eff
        self.soc = min(max(self.soc, 0.0), b.energy)
        sh, dp = shed.solution_value(), dump.solution_value()
        self.cost["fuel"] += fuel
        self.cost["shed"] += VOLL * sh
        self.cost["dump"] += DUMP * dp
        self.shed_mwh += sh
        return {"hour": h, "demand": float(t.demand[h]), "renew_used": ren.solution_value(), "battery": bp,
                "shed": sh, "dump": dp, "fuel": fuel}

    def total(self) -> float:
        return sum(self.cost.values())
