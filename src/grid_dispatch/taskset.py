"""grid_dispatch: run a power grid for one real day as the dispatcher on duty (a live shift).

Each task is a real 2024 day from EIA-930 (demand, its day-ahead forecast, wind and solar) for one
US grid area, with an RTS-GMLC fleet (PGLib-UC) and seeded unit trips. The agent looks, checks
with `simulate`, gives orders and advances the clock 4 hours at a time; the real day unfolds as it
goes. The score is reliability x economy against a solver that re-plans every step using exactly
the information the agent had (see `grading.py` and `build.py`).
"""

import json
from pathlib import Path

import verifiers.v1 as vf
from pydantic import BaseModel

from grid_dispatch.grading import score
from grid_dispatch.shift import SIMULATE_PER_STEP
from grid_dispatch.toolset import GridDispatchToolset, ShiftState, rebuild
from grid_dispatch.world import H

DATA = Path(__file__).with_name("data")
DEFAULT_TASKS = DATA / "v1" / "test.jsonl"   # held-out days; the reference's moves are removed
TRAIN_TASKS = DATA / "v1" / "train.jsonl"

PROMPT = """You are the dispatcher on duty at a power grid control room for one full day, 00:00 to 24:00.

Your job: keep the lights on at the lowest total cost. Every hour, the units you have switched on are \
dispatched automatically at least cost to meet the real demand, after free wind and solar. If the units \
that are on cannot cover demand, the shortfall is unserved energy and costs $5,000/MWh, far more than any fuel. \
Running units costs fuel, and every start-up has a start-up cost.

Your decisions are which units are on in each hour, and the battery. Rules:
- A unit needs `lead` hours of notice to start: from time T you can only order starts at T+lead or later.
- Once on, a unit must run at least `min_up` hours; once off, it must stay off at least `min_down` hours.
- Units can trip without warning and are then out for the day.
- Forecasts of demand and wind+solar are imperfect and get better as the hour approaches.

The day runs in steps: you look, check and act, then call `advance_clock`, which runs the next 4 hours of the \
real day and tells you what happened. Orders you give stay in force until you change them. Each step you may \
call `simulate` up to {sim} times to test changes on the current forecast before committing. The shift ends \
when the clock reaches 24:00; your score depends on the total cost of the whole day compared with a skilled \
dispatcher. Start with `get_system`, `get_status` and `get_forecast`."""


class Baseline(BaseModel):
    cost: float
    shed_mwh: float
    score: float


class Reference(BaseModel):
    policy: str
    """The re-planner that did better on the day: N-1 reserve (rolling) or N-2 (rolling2)."""
    cost: float
    shed_mwh: float
    demand_mwh: float
    baselines: dict[str, Baseline]
    """Every policy played on this day, scored against the reference."""


class GridDispatchData(vf.TaskData):
    world: dict
    reference: Reference


class GridDispatchTaskConfig(vf.TaskConfig):
    max_turns: int = 80


def day_over(state: ShiftState) -> bool:
    step = (state.world or {}).get("step", 4)
    return sum(1 for e in state.log if "advance" in e) * step >= H


def finish(state: ShiftState):
    """The shift doesn't stop: hours the agent didn't reach run on its current orders."""
    shift = rebuild(state.world, state.log)
    auto = shift.w.now < H
    while shift.w.now < H:
        shift.advance_clock()
    return shift, auto


class GridDispatchTask(vf.Task[GridDispatchData, ShiftState, GridDispatchTaskConfig]):
    @property
    def key(self) -> str:
        return self.data.id

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        trace.state.world = self.data.world
        trace.state.log = []
        trace.state.simulate_left = SIMULATE_PER_STEP

    @vf.stop
    async def done(self, trace: vf.Trace) -> bool:
        return day_over(trace.state) or trace.num_turns >= self.config.max_turns

    def _score(self, shift) -> tuple[float, float, float]:
        ref = self.data.reference
        return score(shift.w.total(), shift.w.shed_mwh, ref.cost, ref.shed_mwh, ref.demand_mwh)

    @vf.reward(weight=1.0)
    async def day_score(self, trace: vf.Trace) -> float:
        shift, _ = finish(trace.state)
        return self._score(shift)[0]

    @vf.metric
    async def shift_stats(self, trace: vf.Trace) -> dict[str, float]:
        shift, auto = finish(trace.state)
        ref, cost = self.data.reference, shift.w.total()
        _, rel, eco = self._score(shift)
        return {"reliability": rel, "economy": eco, "cost": cost, "gap_pct": 100 * (cost - ref.cost) / ref.cost,
                "shed_mwh": shift.w.shed_mwh, "excess_shed_mwh": max(0.0, shift.w.shed_mwh - ref.shed_mwh),
                "starts": float(shift.w.starts), "auto_finished": float(auto),
                "steps_played": float(sum(1 for e in trace.state.log if "advance" in e)),
                "orders": float(sum(len(e["act"]) for e in trace.state.log if "act" in e)),
                "simulate_used": float(trace.state.simulate_used),
                "beats_reference": float(cost < ref.cost),
                "beats_static": float(cost < ref.baselines["static"].cost)}


class GridDispatchConfig(vf.TasksetConfig):
    tasks_path: str = str(DEFAULT_TASKS)
    tools: vf.SharedToolsetConfig = vf.SharedToolsetConfig()
    task: GridDispatchTaskConfig = GridDispatchTaskConfig()


class GridDispatchTaskset(vf.Taskset[GridDispatchTask, GridDispatchConfig]):
    @classmethod
    def toolsets(cls, config: GridDispatchConfig) -> list[vf.Toolset]:
        return [GridDispatchToolset(config.tools)]

    def load(self) -> list[GridDispatchTask]:
        tasks = []
        for line in Path(self.config.tasks_path).read_text().splitlines():
            row = json.loads(line)
            ref = row["reference"]
            data = GridDispatchData(id=row["id"], name=row["id"], prompt=PROMPT.format(sim=SIMULATE_PER_STEP),
                                    world=row["world"],
                                    reference=Reference(**{k: ref[k] for k in ("policy", "cost", "shed_mwh",
                                                                               "demand_mwh", "baselines")}))
            tasks.append(GridDispatchTask(data, self.config.task))
        return tasks
