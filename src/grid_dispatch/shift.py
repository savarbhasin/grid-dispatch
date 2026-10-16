"""The agent's side of a live shift: tools that look at, check and act on a World.

Each tool returns plain text. Acting tools change the world; `simulate` runs a copy of it on the
current forecast (the truth is hidden), with a budget per step.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np

from grid_dispatch.build import grade
from grid_dispatch.world import DUMP, H, RESERVE, VOLL, World

SIMULATE_PER_STEP = 3


def _hours(x: np.ndarray, start: int) -> str:
    """On-intervals of a 0/1 schedule from `start`, e.g. '06-18, 20-24' or 'off'."""
    out, h = [], start
    while h < H:
        if x[h]:
            e = h
            while e < H and x[e]:
                e += 1
            out.append(f"{h:02d}-{e:02d}")
            h = e
        else:
            h += 1
    return ", ".join(out) or "off"


class Shift:
    def __init__(self, world: World, reference: float, floor: float):
        self.w, self.reference, self.floor = world, reference, floor
        self.sim_left = SIMULATE_PER_STEP

    # ------------------------------------------------------------------ look
    def get_system(self) -> str:
        t = self.w.task
        lines = ["Units (MW; ramp is per hour; lead = hours of notice to start; $/MWh is the average at max output):",
                 "id     kind     pmin  pmax  ramp  min_up min_down lead  start$  $/h@pmin  $/MWh@pmax"]
        for u in t.units:
            lines.append(f"{u.id:6} {u.kind:8} {u.pmin:5.0f} {u.pmax:5.0f} {u.ramp:5.0f} {u.min_up:6d} {u.min_down:8d} "
                         f"{u.lead:4d} {u.startup_cost:7.0f} {u.curve[0][1]:9.0f} {u.marginal():10.1f}")
        b = t.battery
        lines += [f"Battery: {b.power:.0f} MW, {b.energy:.0f} MWh, {b.eff:.0%} efficient each way.",
                  f"Unserved demand costs ${VOLL:,.0f}/MWh. Thermal output with nowhere to go costs ${DUMP:,.0f}/MWh. "
                  "Wind and solar can be curtailed for free.",
                  f"The clock advances {t.step} h at a time. Grid area {t.ba}, {t.date}."]
        return "\n".join(lines)

    def get_status(self) -> str:
        w, t = self.w, self.w.task
        lines = [f"Time {w.now:02d}:00. Cost so far ${w.total():,.0f} (fuel ${w.cost['fuel']:,.0f}, start-ups "
                 f"${w.cost['startup']:,.0f}, unserved ${w.cost['shed']:,.0f}). Unserved so far {w.shed_mwh:,.1f} MWh.",
                 f"Battery charge {w.soc:,.0f} of {t.battery.energy:,.0f} MWh. Battery plan from now (+ discharge, - charge): "
                 + (", ".join(f"{h:02d}:{v:+.0f}" for h, v in enumerate(w.battery_plan) if h >= w.now and abs(v) > 0.5) or "idle"),
                 f"check budget: {self.sim_left} simulate calls left this step.",
                 "id     state  output  hours_in_state  can_stop_from  can_start_from  scheduled_on"]
        trips = w.known_trips()
        for u in t.units:
            state, n, p = w.status_before(u.id)
            if u.id in trips:
                lines.append(f"{u.id:6} TRIPPED at {trips[u.id]:02d}:00, out for the day")
                continue
            stop = max(w.now, w.now + u.min_up - n) if state else None
            start = max(w.now + u.lead, w.now + u.min_down - n) if not state else None
            lines.append(f"{u.id:6} {'on ' if state else 'off'}  {p:6.0f}  {min(n, 99):14d}  "
                         f"{f'{stop:02d}:00' if stop is not None else '-':>13}  {f'{start:02d}:00' if start is not None else '-':>14}  "
                         f"{_hours(w.sched[u.id], w.now)}")
        if w.log:
            r = w.log[-1]
            lines.append(f"Last step {r['from']:02d}:00-{r['to']:02d}:00: " + "; ".join(
                f"{x['hour']:02d}h demand {x['demand']:,.0f} shed {x['shed']:,.0f}" for x in r["hours"])
                + (f". TRIPS: {', '.join(r['trips'])}" if r.get("trips") else ""))
        return "\n".join(lines)

    def get_forecast(self) -> str:
        w, t = self.w, self.w.task
        dem, ren = t.forecast(w.now)
        lines = ["hour  demand  wind+solar  net_load   (hours before now are actuals)"]
        for h in range(H):
            tag = "actual" if h < w.now else "forecast"
            lines.append(f"{h:02d}   {dem[h]:7,.0f}  {ren[h]:10,.0f}  {dem[h] - ren[h]:8,.0f}   {tag}")
        lines.append(f"Forecasts get closer to the truth as the hour approaches. A {RESERVE:.0%} spinning reserve "
                     "plus enough to cover the largest running unit is the usual safety margin.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ act (shared by simulate)
    def _apply(self, w: World, actions: list[dict]) -> list[str]:
        msgs = []
        for a in actions:
            if "start" in a or "stop" in a:
                uid = a.get("start") or a.get("stop")
                h = int(a.get("hour", w.now))
                until = int(a.get("until") or H)
                if uid not in w.sched:
                    msgs.append(f"unknown unit {uid}")
                    continue
                if not w.now <= h < until <= H:
                    msgs.append(f"{uid}: need {w.now} <= hour < until <= 24")
                    continue
                x = w.sched[uid].copy()
                x[h:until] = 1 if "start" in a else 0
                errs = w.set_schedule(uid, x)
                span = f"{h:02d}:00" + (f"-{until:02d}:00" if until < H else "")
                msgs.append(f"{'start' if 'start' in a else 'stop'} {uid} {span}: " + ("; ".join(errs) if errs else "ok"))
            elif "unit" in a and "on" in a:
                uid = a["unit"]
                if uid not in w.sched:
                    msgs.append(f"unknown unit {uid}")
                    continue
                x = w.sched[uid].copy()
                x[w.now:] = 0
                for s, e in a["on"]:
                    x[max(int(s), w.now):min(int(e), H)] = 1
                errs = w.set_schedule(uid, x)
                msgs.append(f"schedule {uid} on {a['on'] or 'never'}: " + ("; ".join(errs) if errs else "ok"))
            elif "battery" in a:
                plan = w.battery_plan.copy()
                for hh, mw in a["battery"].items():
                    if w.now <= int(hh) < H:
                        plan[int(hh)] = float(mw)
                w.set_battery(plan)
                msgs.append("battery plan set (limited to the battery's power)")
            else:
                msgs.append(f"unknown action {a}")
        return msgs

    def act(self, actions: list[dict]) -> str:
        return "\n".join(self._apply(self.w, actions))

    def simulate(self, actions: list[dict], until: int = H) -> str:
        if self.sim_left <= 0:
            return "No simulate calls left this step; advance the clock to get more."
        self.sim_left -= 1
        w = self.w
        dem, ren = w.task.forecast(w.now)
        fake = dataclasses.replace(w.task, demand=dem, renew=ren, trips=w.known_trips())
        w2 = copy.deepcopy(w)
        w2.task = fake
        before = w2.total()
        msgs = self._apply(w2, actions)
        until = max(w.now + 1, min(H, int(until)))
        rows = []
        while w2.now < until:
            h = w2.now
            rep = w2.advance(1)["hours"][0]
            reserve = sum(min(u.pmax - w2.p[u.id][h], u.ramp) for u in w2.task.units if w2.on[u.id][h])
            rows.append(f"{h:02d}h demand {rep['demand']:,.0f} shed {rep['shed']:,.0f} reserve {reserve:,.0f}")
        return "\n".join(msgs + [f"On the current forecast, {w.now:02d}:00-{until:02d}:00 would cost "
                                 f"${w2.total() - before:,.0f}:"] + rows + [f"({self.sim_left} simulate calls left this step)"])

    def advance_clock(self) -> str:
        if self.w.now >= H:
            return "The day is over."
        rep = self.w.advance(self.w.task.step)
        self.sim_left = SIMULATE_PER_STEP
        shed = sum(x["shed"] for x in rep["hours"])
        lines = [f"Ran {rep['from']:02d}:00-{rep['to']:02d}:00. Unserved {shed:,.1f} MWh. Cost so far ${self.w.total():,.0f}."]
        if rep.get("trips"):
            lines.append(f"TRIPPED: {', '.join(rep['trips'])} (out for the rest of the day)")
        if self.w.now >= H:
            lines.append("The day is over.")
        return "\n".join(lines)

    # ------------------------------------------------------------------ result
    def result(self) -> dict:
        cost = self.w.total()
        return {"cost": cost, "score": grade(cost, self.reference, self.floor), "reference": self.reference,
                "floor": self.floor, "shed_mwh": self.w.shed_mwh, "starts": self.w.starts, "parts": dict(self.w.cost)}
