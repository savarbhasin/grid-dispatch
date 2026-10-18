"""Oracle gate: replay each task's recorded reference actions in a fresh world; the cost must match.

usage: uv run python -m grid_dispatch.check TASKS_JSONL
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from grid_dispatch.data import Unit
from grid_dispatch.world import Battery, Task, World


def task_from_row(row: dict) -> Task:
    w = row["world"]
    units = [Unit(**{**u, "curve": [tuple(pt) for pt in u["curve"]]}) for u in w["units"]]
    return Task(id=row["id"], ba=w["ba"], date=w["date"], units=units, battery=Battery(**w["battery"]),
                demand=np.array(w["demand"]), demand_da=np.array(w["demand_da"]), renew=np.array(w["renew"]),
                renew_da=np.array(w["renew_da"]), trips=w["trips"], init_on=w["init_on"], init_p=w["init_p"],
                step=w["step"])


def replay(row: dict) -> float:
    task = task_from_row(row)
    world = World(task)
    for step in row["reference"]["actions"]:
        assert world.now == step["at"]
        for a in step["actions"]:
            if "battery" in a:
                plan = world.battery_plan.copy()
                plan[world.now:] = a["battery"]
                world.set_battery(plan)
            else:
                x = world.sched[a["unit"]].copy()
                x[world.now:] = a["schedule"]
                errs = world.set_schedule(a["unit"], x)
                assert not errs, errs
        world.advance(task.step)
    return world.total()


def main():
    ok = True
    for line in Path(sys.argv[1]).read_text().splitlines():
        row = json.loads(line)
        cost, want = replay(row), row["reference"]["cost"]
        good = abs(cost - want) < 0.05
        ok &= good
        print(f"{'ok ' if good else 'BAD'} {row['id']}: replayed {cost:,.2f} vs reference {want:,.2f}")
    print("ORACLE GATE PASSES" if ok else "ORACLE GATE FAILED")


if __name__ == "__main__":
    main()
