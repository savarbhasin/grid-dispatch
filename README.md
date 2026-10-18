# Grid Dispatch

**An AI runs a power grid for one real day.** It decides which power plants to switch on and off, and when to use a
battery, so the lights stay on at the lowest cost. The day is real (2024 data from US grids), so things go wrong the
way they do in real life: demand isn't what was forecast, the wind drops, and plants break down.

It's built as a reinforcement-learning environment ([Verifiers v1](https://github.com/PrimeIntellect-ai/verifiers)), so
models can be tested on it and trained on it.

- **500 hard tasks** are included: 453 for training, 47 held out for testing.
- They're also on Hugging Face: [dev-front/grid-dispatch-tasks](https://huggingface.co/datasets/dev-front/grid-dispatch-tasks).

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
solver works out the best plan for the rest of the day from the forecast. It keeps spare power so that one, or any two,
of its big plants can break. Each day uses whichever of the two versions did better. It's a skilled player, not a
perfect one: like the AI, it can't see the future, and the AI is allowed to beat it.

**Score = reliability × economy**, each measured against the solver:

- **Reliability:** did you cause blackouts the solver avoided? Every extra 0.5% of the day's power left unserved divides
  it by about 2.7, so real blackouts push the score toward 0.
- **Economy:** did you spend more on fuel and start-ups than the solver? 5% more gives about 0.68; 10% more about 0.49.
  Spending less never hurts.

So blackouts can't be bought off with fuel savings, and overspending costs in proportion. The reward is `day_score`;
the metrics also report `reliability`, `economy`, unserved MWh, orders, `simulate` use and whether you beat the solver.

## Results so far

One attempt per day, models served by Fireworks, **no thinking cap**. Scores use the score above.

**The 5 hardest test days.** Each has two plants breaking, simple strategies score 0.00–0.13, and the solver keeps the
lights fully on.

| Day | DeepSeek V4.1 Flash | Nemotron Lightning 3.5 |
| --- | --- | --- |
| MISO 2024-07-29 | **1.00** | 0.02 |
| SPP 2024-04-01 | **0.64** | 0.00 |
| ERCOT 2024-11-13 | 0.15 | 0.00 |
| ERCOT 2024-02-28 | 0.09 | 0.00 |
| SPP 2024-12-31 | 0.07 | **0.58** |
| **Average** | **0.39** | **0.12** |

What we saw:
- **DeepSeek** thinks a lot (75–115k tokens and about 15 minutes per day) and uses the tools well (12–18 simulations,
  15–30 orders). It never overspends, but on hard days it **doesn't keep enough spare power** and gets caught by the
  second breakdown.
- **Nemotron** answers in seconds, rarely simulates, gives few orders, and sometimes stops before the day ends.
- Even with unlimited thinking, the strong model averages 0.39, so there's plenty of room to improve. Scores vary
  between attempts, so use several runs per day.

## The tasks

| Split | Tasks | Real days | File |
| --- | --- | --- | --- |
| Train | 453 | 252 | `src/grid_dispatch/data/v1/train.jsonl` |
| Test (default) | 47 | 28 | `src/grid_dispatch/data/v1/test.jsonl` |

- **Split by day:** the same real day can appear several times with a different fleet and breakdowns, so the split is
  by day, and no day is in both sets.
- **Test set:** the solver's moves are removed; the costs needed for scoring are kept.
- **Grid areas:** SPP 208, ERCOT 187, MISO 49, PJM 35, ISNE 21. Windy grids are weighted up because their forecasts
  miss more.
- **Breakdowns per day:** 2 on 284 days, 1 on 196, 0 on 20.
- **Every task is hard on purpose:** following the midnight plan or doing nothing both score ≤ 0.6 (on average 0.10 and
  0.15), and the solver loses at most 1% of the day's power (none on 333 of 500 days).
- **Checked:** replaying the solver's moves reproduces its exact cost on all 500 tasks.

**What's real and what's made up:**

| Real | Made up |
| --- | --- |
| Hourly demand and its real day-ahead forecast (EIA-930) | How many hours' notice each plant type needs |
| Wind and solar output (EIA-930) | The wind/solar forecast (actual plus half of yesterday's error) |
| Plant sizes, costs, ramp rates, minimum on/off times (RTS-GMLC test system) | When plants break (0–2 per day, mostly in the hours before the evening peak) |
| Grid areas: PJM, ISNE, MISO, ERCOT, SPP | The battery, and scaling each day so its peak fills 85% of the fleet |

Data notes: in the 2024 EIA-930 files PJM's forecast is one hour off, and the loader fixes it. California is left out
(its forecast is skewed at midday), and so is New York (missing values).

## Try it

```bash
uv sync -p 3.12

# checks with a scripted player (no AI, no cost): replaying the solver scores 1.0;
# doing nothing scores exactly like the stored do-nothing baseline
uv run python experiments/scripted_check.py --limit 20
uv run python experiments/scripted_check.py --tasks src/grid_dispatch/data/v1/test.jsonl --policies nothing silent

# run a real model on the test days (any OpenAI-compatible API)
uv run vf-eval grid_dispatch --env.agent.harness.id null --env.agent.runtime.type subprocess \
    --model <model> --client.base-url <url> --client.api-key-var <ENV_VAR_WITH_KEY> \
    --sampling.max-tokens 65536 -n 47 --no-push
# training days: add --env.taskset.tasks-path src/grid_dispatch/data/v1/train.jsonl
```

## Build more tasks

```bash
# real grid data (EIA-930) and the power plant fleet (PGLib-UC)
mkdir -p ../../data/grid && cd ../../data/grid
curl -fO https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2024_Jan_Jun.csv
curl -fO https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2024_Jul_Dec.csv
git clone --depth 1 https://github.com/power-grid-lib/pglib-uc.git
cd -

# build (resumable; `touch data/new/STOP` stops it), then check the solver's moves replay exactly
uv run python -m grid_dispatch.build ../../data/grid ../../data/grid/pglib-uc data/new --tasks 500 --candidates 4000 --jobs 8
uv run python -m grid_dispatch.check data/new/tasks.jsonl
```

Each candidate day takes 1–3 CPU-minutes, and about a third pass the hard-day filter. The 500 tasks here were built on
a 60-vCPU cloud machine in about an hour.

## What's in the code

| File | What it is |
| --- | --- |
| `src/grid_dispatch/data.py` | Loads the real grid days and the power plants |
| `src/grid_dispatch/world.py` | The simulator: runs the day hour by hour |
| `src/grid_dispatch/planner.py` | The computer solver (HiGHS) and the comparison strategies |
| `src/grid_dispatch/grading.py` | The reliability × economy score |
| `src/grid_dispatch/build.py` | Builds and filters the tasks |
| `src/grid_dispatch/shift.py` | The tools the AI uses |
| `src/grid_dispatch/toolset.py`, `taskset.py` | The Verifiers environment (tools served over MCP, scoring) |
| `src/grid_dispatch/check.py` | Checks that the solver's moves replay to exactly the same cost |
| `experiments/` | Scripted checks, and a simple standalone runner for any model |

## Licence and credits

- **Code:** Apache-2.0 ([LICENSE](LICENSE)).
- **Tasks** (`src/grid_dispatch/data/v1/`): **CC BY 4.0**. They're built from:
  - EIA-930 Hourly Electric Grid Monitor data (US Energy Information Administration; US government data);
  - [PGLib-UC](https://github.com/power-grid-lib/pglib-uc) (CC BY 4.0), using the
    [RTS-GMLC](https://github.com/GridMod/RTS-GMLC) test system. Changes: units sampled, renamed, rounded, and given
    start-up notice times.
- **Inspired by:** [PortSimEnv](https://github.com/adithya-s-k/FineEnvs/tree/main/07-simulation-environments) (real
  records plus a solver to score against). Sister project: [pack-route](https://github.com/savarbhasin/pack-route).
