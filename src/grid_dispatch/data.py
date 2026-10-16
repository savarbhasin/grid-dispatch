"""Real hourly grid days from EIA-930 and generator fleets from PGLib-UC (RTS-GMLC).

EIA-930 six-month balance files: https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/
(US government data). PGLib-UC: https://github.com/power-grid-lib/pglib-uc (CC BY 4.0).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class Day:
    ba: str
    date: str
    demand: np.ndarray      # actual, MW, 24 local hours
    forecast: np.ndarray    # EIA-930 day-ahead demand forecast, MW
    wind: np.ndarray        # actual net generation, MW
    solar: np.ndarray
    wind_prev: np.ndarray   # the day before: the persistence forecast for wind and solar
    solar_prev: np.ndarray


def _fuel(df: pd.DataFrame, fuel: str) -> pd.Series:
    """Net generation from a fuel; later files split it by integrated battery storage."""
    cols = [c for c in df.columns if c.startswith(f"Net Generation (MW) from {fuel}")
            and "(Imputed)" not in c and "(Adjusted)" not in c]
    return df[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)


def load_eia930(paths: list[Path], bas: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        df = pd.read_csv(p, thousands=",", low_memory=False)
        df = df[df["Balancing Authority"].isin(bas)]
        frames.append(pd.DataFrame({
            "ba": df["Balancing Authority"], "date": pd.to_datetime(df["Data Date"]).dt.strftime("%Y-%m-%d"),
            "hour": df["Hour Number"].astype(int),
            "fc": pd.to_numeric(df["Demand Forecast (MW)"], errors="coerce"),
            "dem": pd.to_numeric(df["Demand (MW)"], errors="coerce"),
            "wind": _fuel(df, "Wind"), "solar": _fuel(df, "Solar"),
        }))
    return pd.concat(frames, ignore_index=True)


# In the 2024 files these areas' day-ahead forecast runs one hour behind their actual demand
# (the error is smallest when actual[h] is compared with forecast[h - 1]); others line up.
FORECAST_LAG = {"PJM": 1, "CISO": 1}


def days(df: pd.DataFrame, ba: str) -> list[Day]:
    """Complete 24-hour days (daylight-saving days with 23 or 25 hours are skipped)."""
    out, prev = [], None
    d = df[df.ba == ba].sort_values(["date", "hour"]).copy()
    d["fc"] = d["fc"].shift(FORECAST_LAG.get(ba, 0))
    for date, g in d.groupby("date"):
        g = g.fillna({"wind": 0.0, "solar": 0.0})
        ok = len(g) == 24 and g.hour.tolist() == list(range(1, 25)) and g[["fc", "dem"]].notna().all().all() \
            and (g.dem > 0).all() and (g.fc > 0).all()
        if ok and prev is not None:
            out.append(Day(ba, date, g.dem.to_numpy(float), g.fc.to_numpy(float),
                           g.wind.clip(lower=0).to_numpy(float), g.solar.clip(lower=0).to_numpy(float),
                           prev.wind.clip(lower=0).to_numpy(float), prev.solar.clip(lower=0).to_numpy(float)))
        prev = g if ok else None
    return out


@dataclass
class Unit:
    id: str
    kind: str          # nuclear, coal, cc, steam, ct (inferred from size, run times and cost)
    pmin: float
    pmax: float
    ramp: float        # MW per hour, up and down
    ramp_su: float     # max output in the start-up hour
    ramp_sd: float     # max output in the hour before shut-down
    min_up: int
    min_down: int
    lead: int          # hours between ordering a start and the unit coming on (synthetic, by kind)
    startup_cost: float
    curve: list[tuple[float, float]]   # (MW, $/h) points, convex; curve[0] is (pmin, cost at pmin)

    def cost(self, p: float) -> float:
        pts = self.curve
        if p <= pts[0][0]:
            return pts[0][1]
        for (a, ca), (b, cb) in zip(pts, pts[1:]):
            if p <= b + 1e-9:
                return ca + (cb - ca) * (p - a) / (b - a) if b > a else cb
        return pts[-1][1]

    def marginal(self) -> float:
        """Average $/MWh at full output: the merit-order key."""
        return self.curve[-1][1] / self.pmax


LEAD_BY_KIND = {"nuclear": 8, "coal": 6, "steam": 4, "cc": 3, "ct": 1}


def _kind(g: dict) -> str:
    up, pmax = g["time_up_minimum"], g["power_output_maximum"]
    full = g["piecewise_production"][-1]["cost"] / pmax
    if up >= 24:
        return "nuclear" if full < 15 else "coal"
    if up >= 8:
        return "cc" if pmax >= 200 else "steam"
    return "ct"


def rts_gmlc_units(path: Path) -> list[Unit]:
    d = json.loads(path.read_text())
    units = []
    for name, g in sorted(d["thermal_generators"].items()):
        if g["power_output_maximum"] < 10:
            continue
        kind = _kind(g)
        curve = [(pt["mw"], pt["cost"]) for pt in g["piecewise_production"]]
        units.append(Unit(
            id=name, kind=kind, pmin=g["power_output_minimum"], pmax=g["power_output_maximum"],
            ramp=g["ramp_up_limit"], ramp_su=max(g["ramp_startup_limit"], g["power_output_minimum"]),
            ramp_sd=max(g["ramp_shutdown_limit"], g["power_output_minimum"]),
            min_up=int(g["time_up_minimum"]), min_down=int(g["time_down_minimum"]),
            lead=LEAD_BY_KIND[kind],
            startup_cost=float(np.mean([s["cost"] for s in g["startup"]])), curve=curve))
    return units
