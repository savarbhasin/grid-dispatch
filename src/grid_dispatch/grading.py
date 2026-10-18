"""The score: reliability x economy, each measured against the reference dispatcher.

R (reliability) = e^(-excess unserved MWh / (SHED_SCALE x the day's demand)): shedding no more than
    the reference did gives 1; every extra 0.5% of the day's energy left unserved divides it by e.
E (economy) = 0.2 + 0.8 e^(-overspend / COST_SCALE), where overspend is the running cost (fuel,
    no-load, start-ups, dumped output; unserved energy excluded) above the reference's, as a share
    of it. Matching or beating the reference gives 1; 10% over gives about 0.49.

Blackouts can't be bought off with fuel savings (they only touch R), and overspending costs in
proportion (it only touches E).
"""

import math

from grid_dispatch.world import VOLL

SHED_SCALE = 0.005
COST_SCALE = 0.10


def running_cost(cost: float, shed_mwh: float) -> float:
    return cost - VOLL * shed_mwh


def score(cost: float, shed_mwh: float, ref_cost: float, ref_shed_mwh: float, demand_mwh: float) -> tuple[float, float, float]:
    """(score, R, E) for a day that cost `cost` with `shed_mwh` unserved."""
    r = math.exp(-max(0.0, shed_mwh - ref_shed_mwh) / (SHED_SCALE * demand_mwh))
    ref_run = running_cost(ref_cost, ref_shed_mwh)
    over = max(0.0, running_cost(cost, shed_mwh) - ref_run) / ref_run
    e = 0.2 + 0.8 * math.exp(-over / COST_SCALE)
    return r * e, r, e
