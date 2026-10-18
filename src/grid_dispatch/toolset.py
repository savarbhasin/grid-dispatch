"""The dispatcher's tools over MCP. One shared server per environment worker.

Rollout state is JSON (Verifiers pulls and pushes it around every call), so it holds the task's
world and the log of what the agent has done; each call rebuilds the live world by replaying
that log. The simulator is deterministic, so a replay is exact (see check.py), and it is cheap:
at most 24 hourly dispatch LPs.
"""

import verifiers.v1 as vf
from pydantic import BaseModel

from grid_dispatch.check import task_from_row
from grid_dispatch.shift import SIMULATE_PER_STEP, Shift
from grid_dispatch.world import H, World


class Order(BaseModel):
    """One order. {"start": unit, "hour": h} keeps the unit on from hour h to the end of the day
    (add "until": e to stop at hour e); {"stop": unit, "hour": h} keeps it off from h (with "until",
    only until e). {"unit": unit, "on": [[6, 14], [17, 22]]} replaces the unit's whole schedule from
    now: on in those hour ranges, off otherwise. {"battery": {"17": 80, "18": -40}} sets battery MW
    for those hours (+ discharge, - charge)."""
    start: str | None = None
    stop: str | None = None
    hour: int | None = None
    until: int | None = None
    unit: str | None = None
    on: list[list[int]] | None = None
    battery: dict[str, float] | None = None


class ShiftState(vf.State):
    world: dict | None = None          # the task's world (hidden parts included; never shown as is)
    log: list[dict] = []               # {"act": [orders]} and {"advance": true}, in order
    simulate_left: int = SIMULATE_PER_STEP
    simulate_used: int = 0


def rebuild(world: dict, log: list[dict]) -> Shift:
    shift = Shift(World(task_from_row({"id": "", "world": world})))
    for entry in log:
        if "act" in entry:
            shift.act(entry["act"])
        else:
            shift.advance_clock()
    return shift


def orders(actions: list[Order]) -> list[dict]:
    return [a.model_dump(exclude_none=True) for a in actions]


class GridDispatchToolset(vf.Toolset[vf.SharedToolsetConfig, ShiftState]):
    TOOL_PREFIX = None

    def _shift(self) -> Shift:
        if self.state.world is None:
            raise RuntimeError("no task loaded for this rollout")
        shift = rebuild(self.state.world, self.state.log)
        shift.sim_left = self.state.simulate_left
        return shift

    @vf.tool
    async def get_system(self) -> str:
        """Unit data, battery and costs (fixed for the day)."""
        return self._shift().get_system()

    @vf.tool
    async def get_status(self) -> str:
        """Clock, each unit's state and schedule, battery, costs so far, and what happened last step."""
        return self._shift().get_status()

    @vf.tool
    async def get_forecast(self) -> str:
        """Demand and wind+solar by hour as known now (actuals for past hours)."""
        return self._shift().get_forecast()

    @vf.tool
    async def simulate(self, actions: list[Order], until: int = H) -> str:
        """Test orders on a copy of the grid using the current forecast; the real world is unchanged.
        Limited calls per step."""
        shift = self._shift()
        before = shift.sim_left
        out = shift.simulate(orders(actions), until)
        if shift.sim_left < before:
            self.state.simulate_left = shift.sim_left
            self.state.simulate_used += 1
        return out

    @vf.tool
    async def act(self, actions: list[Order]) -> str:
        """Give orders for real: starts, stops, battery. Each order is confirmed or refused with the reason."""
        shift = self._shift()
        if shift.w.now >= H:
            return "The day is over."
        out = shift.act(orders(actions))
        self.state.log = [*self.state.log, {"act": orders(actions)}]
        return out

    @vf.tool
    async def advance_clock(self) -> str:
        """Run the next 4 hours of the real day and report what happened."""
        shift = self._shift()
        if shift.w.now >= H:
            return "The day is over."
        out = shift.advance_clock()
        self.state.log = [*self.state.log, {"advance": True}]
        self.state.simulate_left = SIMULATE_PER_STEP
        return out


if __name__ == "__main__":
    GridDispatchToolset.run()
