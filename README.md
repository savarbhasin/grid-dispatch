# Grid Dispatch

**An AI runs a power grid for one real day.** It decides which power plants to switch on and off, and when to use a
battery, so the lights stay on at the lowest cost. The day is real (2024 data from US grids), so things go wrong the
way they do in real life: demand isn't what was forecast, the wind drops, and plants break down.

It's built as a reinforcement-learning environment ([Verifiers v1](https://github.com/PrimeIntellect-ai/verifiers)), so
models can be tested on it and later trained on it.

> **Status: early prototype.** The code works and 10 test days have been built and checked, but **no data is in this
> repo yet.** See [Get the data](#get-the-data) to build it yourself.

## How a day plays out

1. The AI gets **17 power plants** and a battery. Big plants are cheap to run but slow to start (they need hours of
   warning). Small plants start fast but are expensive.
2. The day runs in **4-hour steps**. At each step the AI:
   - looks at the grid and the forecast;
   - tries out ideas with `simulate` (a "what if" on a copy of the grid);
   - gives orders: turn plants on or off, charge or discharge the battery;
   - moves the clock forward 4 hours.
3. When the clock moves, **the real day happens**: real demand arrives, plants may break, and the cost adds up.
4. If there isn't enough power, people lose electricity. That costs **$5,000 per MWh**, far more than any fuel.

The hard part is that decisions can't be undone and take time to work. A big plant needs hours of notice, and once on
it must stay on for a minimum time. A good dispatcher has to **act before the problem arrives**.

## The tools the AI uses

| Tool | What it does |
| --- | --- |
| `get_system` | Lists the plants: size, cost, how fast they start, minimum on/off times. Plus the battery |
| `get_status` | What's on right now, the time, costs so far, and what happened in the last 4 hours |
| `get_forecast` | Expected demand and wind/solar for the rest of the day (these get more accurate closer to the hour) |
| `simulate` | "What would happen if I did this?" Tested on the forecast. 3 tries per step |
| `act` | Give real orders. Each one is confirmed, or refused with a reason |
| `advance_clock` | Move forward 4 hours and see what happened |

Example orders:
- `{"start": "CC05", "hour": 11}`: turn plant CC05 on from 11:00;
- `{"stop": "CT12", "hour": 20}`: turn it off from 20:00;
- `{"unit": "CC05", "on": [[11, 20]]}`: on only from 11:00 to 20:00;
- `{"battery": {"17": 80}}`: discharge 80 MW at 17:00 (negative numbers charge).

## How it's scored

We compare the AI with a **computer solver that plays the same day with the same information**. Every 4 hours the
solver works out the best plan for the rest of the day based on the forecast. It's a skilled player, not a perfect one:
like the AI, it can't see the future. The AI is allowed to beat it.

**Current score:** based on the day's total cost compared with the solver's, from 0.2 (much worse) to 1.0 (as good or
better).

**Known problems, being fixed:**
- Spending a little too much can score as badly as causing a blackout.
- On days where two plants break, the solver sometimes loses to an AI that just keeps every plant switched on.

**Planned new score** (tested, not built yet): two checks multiplied together.
- **Reliability:** did you cause blackouts the solver avoided? Any real blackout pushes this toward 0.
- **Economy:** did you waste money? 5% too much costs a little; 10% costs more.

## Results so far

One attempt per day, models served by Fireworks. Scores use the **planned new score**.

| Day | What makes it hard | DeepSeek V4.1 Flash | Nemotron Lightning 3.5 |
| --- | --- | --- | --- |
| PJM 2024-05-05 | One plant breaks | **1.00** | 0.64 |
| SPP 2024-05-31 | Wind drop + two plants break | **1.00** | 1.00 |
| SPP 2024-10-31 | Big wind drop + a plant breaks | 0.49 | 0.00 |
| ERCOT 2024-11-25 | Two plants break | 0.00 | 0.00 |
| Average over all 10 days | | — | 0.46 |

What we saw:
- **DeepSeek** thinks hard (about 15 minutes per day). It understands the rules, but on hard days it tries to save
  money, switches on too few plants, and gets caught when the wind drops or a plant breaks.
- **Nemotron** is fast (about 30 seconds per day) but mostly does nothing. It leaves the midnight plants running. That's
  safe but wasteful on easy days, and causes big blackouts on hard ones.
- Results vary a lot between attempts (DeepSeek scored 0.02 and then 0.49 on the same day), so more runs per day are
  needed.

## Get the data

Nothing is checked in. To download the data and build the 10 test days:

```bash
uv sync -p 3.12

# real grid data (EIA-930) and the power plant fleet (PGLib-UC)
mkdir -p ../../data/grid && cd ../../data/grid
curl -fO https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2024_Jan_Jun.csv
curl -fO https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2024_Jul_Dec.csv
git clone --depth 1 https://github.com/power-grid-lib/pglib-uc.git
cd -

# build the days, then check the solver's answers replay exactly
uv run python -m grid_dispatch.build ../../data/grid ../../data/grid/pglib-uc data/tasks-v0.jsonl --tasks 10 --candidates 60
mkdir -p src/grid_dispatch/data && cp data/tasks-v0.jsonl src/grid_dispatch/data/
uv run python -m grid_dispatch.check src/grid_dispatch/data/tasks-v0.jsonl
```

**What's real and what's made up:**

| Real | Made up |
| --- | --- |
| Hourly demand and its real day-ahead forecast (EIA-930) | How many hours' notice each plant type needs |
| Wind and solar output (EIA-930) | The wind/solar forecast (actual plus half of yesterday's error) |
| Plant sizes, costs, ramp rates, minimum on/off times (RTS-GMLC test system) | When plants break (0–2 per day, random) |
| Grid areas: PJM, ISNE, MISO, ERCOT, SPP | The battery, and scaling each day to fit the fleet |

A day is only kept if just following the midnight plan does badly, so reacting during the day has to matter.

Data notes: in the 2024 EIA-930 files PJM's forecast is one hour off, and the loader fixes it. California is left out
(its forecast is skewed at midday), and so is New York (missing values).

## Try it

```bash
# checks with a scripted player (no AI, no cost): replaying the solver scores 1.0; doing nothing scores like doing nothing
uv run python experiments/scripted_check.py

# run a real model on the 10 days (any OpenAI-compatible API)
uv run vf-eval grid_dispatch --env.agent.harness.id null --env.agent.runtime.type subprocess \
    --model <model> --client.base-url <url> --client.api-key-var <ENV_VAR_WITH_KEY> \
    --sampling.max-tokens 65536 -n 10 --no-push
```

## What's in the code

| File | What it is |
| --- | --- |
| `src/grid_dispatch/data.py` | Loads the real grid days and the power plants |
| `src/grid_dispatch/world.py` | The simulator: runs the day hour by hour |
| `src/grid_dispatch/planner.py` | The computer solver (HiGHS) and the comparison strategies |
| `src/grid_dispatch/build.py` | Builds the test days |
| `src/grid_dispatch/shift.py` | The tools the AI uses |
| `src/grid_dispatch/toolset.py`, `taskset.py` | The Verifiers environment (tools served over MCP, scoring) |
| `src/grid_dispatch/check.py` | Checks that the solver's moves replay to exactly the same cost |
| `experiments/` | Scripted checks, and a simple standalone runner for any model |

## Licence and credits

- **Code:** Apache-2.0 ([LICENSE](LICENSE)).
- **Grid data:** US Energy Information Administration, EIA-930 Hourly Electric Grid Monitor (US government data).
- **Power plants:** [PGLib-UC](https://github.com/power-grid-lib/pglib-uc) (CC BY 4.0), using the
  [RTS-GMLC](https://github.com/GridMod/RTS-GMLC) test system.
- **Inspired by:** [PortSimEnv](https://github.com/adithya-s-k/FineEnvs/tree/main/07-simulation-environments) (real
  records plus a solver to score against). Sister project: [pack-route](https://github.com/savarbhasin/pack-route).
