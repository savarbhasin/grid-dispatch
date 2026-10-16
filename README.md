# Grid Dispatch

A Verifiers v1 environment where the agent is the **dispatcher on duty at a power grid for one real day**. The day runs
in 4-hour steps: the agent looks at the grid, tests options with `simulate`, gives orders (start or stop units, set the
battery), then advances the clock. The real day unfolds as it goes: demand differs from the forecast, wind falls short,
units trip. The score compares the day's total cost with a solver that re-plans every step using exactly the
information the agent had.

**Status: prototype.** 10 tasks have been built and the quality gates pass, but no data or tasks are in this repo yet:
see [Data](#data) to build them.

## Why it's agentic

The agent doesn't hand in one answer. Decisions are irreversible and have delayed effects:
- a started unit must run its minimum hours;
- a big unit needs hours of notice to start;
- the truth only arrives as time passes.

The agent has to act before problems show, watch the forecast drift, check options before committing, and recover
from trips.

## Tools (MCP, served by `GridDispatchToolset`)

| Tool | What it does |
| --- | --- |
| `get_system()` | Units (min/max MW, ramp, min up/down, start-up notice, start-up and fuel costs), battery, prices |
| `get_status()` | Clock, each unit's state and schedule, when it may next stop or start, battery charge, costs so far, last step's events |
| `get_forecast()` | Demand and wind+solar by hour as known now (actuals for past hours) |
| `simulate(actions, until)` | Runs the orders on a copy of the grid using the current forecast. 3 calls per step |
| `act(actions)` | Gives orders for real; each is confirmed or refused with the reason |
| `advance_clock()` | Runs the next 4 hours of the real day and reports unserved energy, trips and cost |

Orders:
- `{"start": "CC05", "hour": 11}` or `{"stop": …}`, optionally with `"until": e`;
- `{"unit": "CC05", "on": [[11, 20]]}`, which replaces a unit's whole schedule;
- `{"battery": {"17": 80}}` (+ discharges, − charges).

Every hour, the units that are on are dispatched automatically at least cost. A shortfall is unserved energy at
$5,000/MWh.

Rollout state is JSON: the task's world plus a log of the agent's orders and clock advances. Each tool call rebuilds
the live world by replaying that log. The simulator is deterministic, so the replay is exact.

## Grading

- **Reference:** a rolling re-planner. Every step it solves a unit-commitment MILP (HiGHS) over the rest of the day,
  on the same forecast the agent sees. It keeps a 3% reserve plus enough to cover the largest running unit tripping.
- **Floor:** the cheapest of a plan that knew the whole day in advance and the other policies.
- **Current reward:** PortSim reward v3 on total cost: `0.2 + 0.8·e^(−g/0.5)` with
  `g = (cost − reference) / (reference − floor + 1% of reference)`.

**Known problems, being worked on:**
- When the floor equals the reference, the cushion is only 1%, so a 5% overspend scores like a blackout.
- On two-trip days, "do nothing" (keep every unit on) can beat the reference.

The proposed replacement is **reliability × economy**:
- `R = e^(−excess unserved MWh / 0.5% of the day's demand)`;
- `E = 0.2 + 0.8·e^(−overspend / 10%)`, where overspend is fuel plus start-up cost above the reference's.

**Metrics:**
- `cost`, `gap_pct`, `shed_mwh`, `starts`;
- `steps_played`, `orders`, `simulate_used`;
- `beats_reference`, `beats_static`, `auto_finished`.

## Data

Nothing is checked in. To build tasks:

```bash
uv sync -p 3.12
mkdir -p ../../data/grid && cd ../../data/grid
curl -fO https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2024_Jan_Jun.csv
curl -fO https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2024_Jul_Dec.csv
git clone --depth 1 https://github.com/power-grid-lib/pglib-uc.git
cd -
uv run python -m grid_dispatch.build ../../data/grid ../../data/grid/pglib-uc data/tasks-v0.jsonl --tasks 10 --candidates 60
mkdir -p src/grid_dispatch/data && cp data/tasks-v0.jsonl src/grid_dispatch/data/
uv run python -m grid_dispatch.check src/grid_dispatch/data/tasks-v0.jsonl   # replay gate
```

How a task is made:
- **Real (EIA-930):** a real 2024 day for one grid area (PJM, ISNE, MISO, ERCOT or SPP): hourly demand, its real
  day-ahead forecast, and wind and solar.
- **Real-based fleet:** 17 units sampled from the RTS-GMLC fleet in PGLib-UC.
- **Synthetic:** start-up notice by unit type, the wind and solar forecast (actual plus half of yesterday's error), 0–2
  seeded trips, a battery, and scaling the day so its peak is 80% of the fleet.

A day is kept when the plan made at midnight scores ≤ 0.9, so adapting has to matter.

Data notes:
- PJM's (and CAISO's) day-ahead forecast in the 2024 EIA-930 files runs one hour behind actual demand; the loader
  shifts it back.
- CAISO is excluded (large midday forecast bias) and NYIS (zero rows).

## Check and run

```bash
uv run python experiments/scripted_check.py          # oracle = 1.0, do-nothing and silent = the do-nothing baseline
uv run vf-eval grid_dispatch --env.agent.harness.id null --env.agent.runtime.type subprocess \
    --model <model> --client.base-url <url> --client.api-key-var <VAR> --sampling.max-tokens 65536 -n 10 --no-push
```

`experiments/run_agent.py` is a minimal standalone tool-calling loop (any OpenAI-compatible API), used for the first
runs before the Verifiers port.

## Early results (one attempt each, DeepSeek V4.1 Flash and Nemotron Lightning 3.5 on Fireworks)

| Task | Model | Unserved (reference) | Current grade | Proposed R×E |
| --- | --- | --- | --- | --- |
| PJM 2024-05-05 | DeepSeek | 0 MWh (0) | 1.00 | 1.00 |
| PJM 2024-05-05 | Nemotron | 0 MWh (0), cost +5% | 0.20 | 0.68 |
| SWPP 2024-10-31 | DeepSeek | 1,191 MWh (38) | 0.20 | 0.02 |
| ERCOT 2024-11-25 | DeepSeek | 1,921 MWh (215) | 0.20 | 0.00 |

On the hard days DeepSeek understood the rules, start-up notice included. It still under-committed: it spent about 7%
less than the reference on fuel and start-ups, and shed far more load.

## Licence and attribution

- **Code:** Apache-2.0 ([LICENSE](LICENSE)).
- **EIA-930:** US Energy Information Administration, Hourly Electric Grid Monitor. US government data.
- **PGLib-UC:** [power-grid-lib/pglib-uc](https://github.com/power-grid-lib/pglib-uc), CC BY 4.0. Its RTS-GMLC fleet
  is from the [Grid Modernization Lab Consortium test system](https://github.com/GridMod/RTS-GMLC).
- **Inspired by:** the "real records + solver reference" pattern and the reward formula come from
  [PortSimEnv](https://github.com/adithya-s-k/FineEnvs/tree/main/07-simulation-environments) (FineEnvs). Sister
  environment: [pack-route](https://github.com/savarbhasin/pack-route).
