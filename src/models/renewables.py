"""Weather -> wind and solar capacity factors, calibrated on Fingrid data.

Wind (target: wind_cf = Fingrid wind 75 / Fingrid forecast capacity 268)
  1. logistic   : cf = c + L / (1 + exp(-k (v - v0))), v = mean 100 m wind speed over the
                  wind points. c lets calm hours go slightly negative (turbine self-use), so
                  predictions are never clipped.
  2. per_point  : the same logistic fitted to each point's wind speed, then combined with
                  weights w >= 0, sum(w) = 1, fitted to wind_cf.
  3. gbm        : HistGradientBoosting on all point wind speeds + month + hour (upper benchmark).
  Validation: leave-one-month-out CV (train on 11 months, predict the 12th).
  Outlier check: hours where |actual - out-of-fold prediction of the best interpretable model|
  exceeds OUTLIER_K robust standard deviations of the residuals. Report only.
  Ramp check: unexplained hour-to-hour changes (jumps) and up-down pairs (spikes). Report only.

Solar
  Baseline profile: solar_cf_fingrid = Fingrid 248 / 267 (from the hourly table). 248 is itself a
  forecast-based estimate; its 09-12 UTC timing artefact is corrected in the table when
  solar.artefact_correction is set (raw copy: solar_cf_fingrid_raw).
  Alternative profile: pvlib per city point (equal capacity shares), fixed tilt/azimuth, PVWatts
  DC with SAPM cell temperature -> raw output per MW. One scale factor is fitted on DAILY totals
  against 248 (avoids its intra-day artefact); it absorbs system losses, snow, shading,
  orientation mix and the capacity estimate. Both profiles are written to
  data/processed/solar_profiles_<year>.parquet.

Wind fitting uses only hours whose flags are "ok" or "filled" (suspect / missing excluded), that
are not labelled curtailment (config/data_issues.yaml) and whose day-ahead price is > 0 (wind
is often curtailed at price <= 0; curtailment is a price-based rule in the scenario engine).

Usage:
    python -m src.models.renewables --year 2024
    python -m src.models.renewables --wind-years 2019 2020 2021 2022 2023 2024 2025
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib
from scipy.optimize import curve_fit, minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.fetch import weather

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
MODELS_DIR = REPO_ROOT / "models"
FIGURES_DIR = REPO_ROOT / "outputs" / "figures"

FIT_FLAGS = ["ok", "filled"]
OUTLIER_K = 5.0  # threshold in robust standard deviations; see outlier_report()
RAMP_K = 8.0  # single unexplained ramp, robust sigmas; see ramp_z()
SPIKE_K = 5.0  # both legs of an up-down (or down-up) spike; see spike_hours()

# Solar assumptions (stated in the report; the scale factor absorbs most of their error).
# 15 deg: flatter than the south-facing optimum, closer to typical Finnish rooftops; fits
# Fingrid 248 daily totals better than 35 deg (2024 daily-total R2: 15 deg 0.939, 35 deg 0.879).
SOLAR_TILT_DEG = 15.0
SOLAR_AZIMUTH_DEG = 180.0  # south
GAMMA_PDC = -0.004  # 1/K, typical crystalline Si
SAPM_MOUNT = "open_rack_glass_polymer"

# Reference palette (dataviz skill), light mode
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
     "text": "#0b0b0b", "muted": "#52514e", "grid": "#e4e3df", "surface": "#fcfcfb", "critical": "#e34948"}


# ---------------------------------------------------------------- wind models

def logistic(v, L, k, v0, c):
    return c + L / (1.0 + np.exp(-k * (v - v0)))


LOGISTIC_P0 = [0.85, 0.8, 7.0, 0.0]
LOGISTIC_BOUNDS = ([0.1, 0.05, 0.0, -0.1], [1.5, 5.0, 20.0, 0.2])


def fit_logistic(v: np.ndarray, y: np.ndarray) -> np.ndarray:
    p, _ = curve_fit(logistic, v, y, p0=LOGISTIC_P0, bounds=LOGISTIC_BOUNDS, maxfev=20000)
    return p


class LogisticModel:
    name = "logistic"

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.params = fit_logistic(X["ws100_mean"].to_numpy(), y.to_numpy())
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return logistic(X["ws100_mean"].to_numpy(), *self.params)

    def to_dict(self) -> dict:
        return {"formula": "c + L / (1 + exp(-k (v - v0)))", "input": "ws100_mean",
                **dict(zip(["L", "k", "v0", "c"], map(float, self.params)))}


class PerPointModel:
    name = "per_point"

    def __init__(self, points: list[str]):
        self.points = points

    def _point_preds(self, X: pd.DataFrame) -> np.ndarray:
        return np.column_stack([logistic(X[f"ws100_{p}"].to_numpy(), *self.params[p]) for p in self.points])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.params = {p: fit_logistic(X[f"ws100_{p}"].to_numpy(), y.to_numpy()) for p in self.points}
        P, yv, n = self._point_preds(X), y.to_numpy(), len(self.points)
        res = minimize(
            lambda w: np.mean((P @ w - yv) ** 2),
            np.full(n, 1.0 / n),
            method="SLSQP",
            bounds=[(0.0, 1.0)] * n,
            constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        )
        self.weights = np.clip(res.x, 0, None) / np.clip(res.x, 0, None).sum()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self._point_preds(X) @ self.weights

    def to_dict(self) -> dict:
        return {
            "formula": "sum_i w_i * (c_i + L_i / (1 + exp(-k_i (v_i - v0_i))))",
            "points": {p: {"weight": float(w), **dict(zip(["L", "k", "v0", "c"], map(float, self.params[p])))}
                       for p, w in zip(self.points, self.weights)},
        }


class GBMModel:
    name = "gbm"

    def __init__(self, points: list[str]):
        self.features = [f"ws100_{p}" for p in points] + ["month_local", "hour_local"]

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                                   min_samples_leaf=40, random_state=0)
        self.model.fit(X[self.features], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X[self.features])


def leave_one_month_out(make_model, t: pd.DataFrame, fit_mask: pd.Series) -> pd.Series:
    """Out-of-fold predictions for every hour (incl. excluded hours), fitting only on fit_mask."""
    oof = pd.Series(np.nan, index=t.index)
    for m in range(1, 13):
        test = t["month_local"] == m
        train = fit_mask & ~test
        model = make_model().fit(t[train], t.loc[train, "wind_cf"])
        oof[test] = model.predict(t[test])
    return oof


def scores(y: pd.Series, pred: pd.Series) -> dict:
    return {"MAE": mean_absolute_error(y, pred), "R2": r2_score(y, pred)}


# ---------------------------------------------------------------- solar model

def solar_raw_per_mw(t: pd.DataFrame, year: int) -> pd.Series:
    """Mean over the city points of PVWatts DC output per MW installed (before scaling)."""
    start, end = f"{year}-01-01", f"{year + 1}-01-01"
    outs = []
    for name, p in weather.load_locations()["solar"].items():
        wx = pd.read_parquet(weather.hourly_path(name, start, end)).reindex(t.index)
        # radiation is the mean over [t, t+1h): evaluate the sun at mid-hour
        mid = t.index + pd.Timedelta("30min")
        sp = pvlib.solarposition.get_solarposition(mid, p["lat"], p["lon"])
        dni_extra = pvlib.irradiance.get_extra_radiation(mid)
        poa = pvlib.irradiance.get_total_irradiance(
            SOLAR_TILT_DEG, SOLAR_AZIMUTH_DEG, sp["apparent_zenith"].to_numpy(), sp["azimuth"].to_numpy(),
            dni=wx["direct_normal_irradiance"].to_numpy(), ghi=wx["shortwave_radiation"].to_numpy(),
            dhi=wx["diffuse_radiation"].to_numpy(), dni_extra=dni_extra.to_numpy(), model="haydavies",
        )
        poa_global = np.nan_to_num(np.asarray(poa["poa_global"], dtype=float)).clip(min=0)
        params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][SAPM_MOUNT]
        t_cell = pvlib.temperature.sapm_cell(poa_global, wx["temperature_2m"].to_numpy(),
                                             wx["wind_speed_10m"].to_numpy(), **params)
        pdc = pvlib.pvsystem.pvwatts_dc(poa_global, t_cell, pdc0=1.0, gamma_pdc=GAMMA_PDC)
        outs.append(pd.Series(pdc, index=t.index, name=name))
    return pd.concat(outs, axis=1).mean(axis=1)


def fit_solar(t: pd.DataFrame, raw: pd.Series) -> dict:
    """One scale factor, least squares on daily totals of MW (= MWh/day)."""
    mask = t["flag_solar"].isin(FIT_FLAGS) & t["flag_solar_capacity"].isin(FIT_FLAGS)
    unscaled_mw = (raw * t["solar_capacity"]).where(mask)
    actual = t["solar"].where(mask)
    d_pred, d_act = unscaled_mw.resample("D").sum(min_count=24), actual.resample("D").sum(min_count=24)
    ok = d_pred.notna() & d_act.notna()
    scale = float((d_pred[ok] * d_act[ok]).sum() / (d_pred[ok] ** 2).sum())
    return {"scale": scale, "daily_pred": scale * d_pred[ok], "daily_actual": d_act[ok]}


# ---------------------------------------------------------------- reporting

def outlier_report(t: pd.DataFrame, resid: pd.Series, fit_mask: pd.Series) -> tuple[pd.Series, float, float]:
    """Flag |resid| > OUTLIER_K * robust sigma (1.4826 * MAD of fit-hour residuals).

    Robust sigma, because the residuals are heavy-tailed and the outliers themselves must not
    inflate the scale. K = 5: for a normal distribution 5 sigma is exceeded in ~1 of 1.7 million
    hours, so with 8,784 hours a flagged hour is very unlikely to be ordinary model error.
    This catches sustained disagreement (curtailment, icing, weather-model bias), NOT short
    data spikes: the known 2024-07-12 23:00 artefact is only 3.7 sigma (2024 check). A ramp
    check (hour-to-hour change unexplained by the model) catches that one (-9.6 sigma).
    """
    r = resid[fit_mask]
    sigma = 1.4826 * (r - r.median()).abs().median()
    threshold = OUTLIER_K * sigma
    flagged = resid.abs() > threshold
    return flagged, sigma, threshold


def ramp_z(t: pd.DataFrame, pred: pd.Series, fit_mask: pd.Series) -> tuple[pd.Series, float]:
    """Hour-to-hour change in actual wind_cf not explained by the change in prediction, in robust sigmas.

    A short data spike (e.g. 2024-07-12 23:00) barely moves the level residual but shows up as a
    large unexplained ramp up followed by one down.
    """
    ramp = t["wind_cf"].diff() - pred.diff()
    ok = fit_mask & fit_mask.shift(1, fill_value=False)
    r = ramp[ok]
    sigma = 1.4826 * (r - r.median()).abs().median()
    return ramp / sigma, float(sigma)


def spike_hours(rz: pd.Series) -> pd.Series:
    """Hour t is a spike if the ramp into t and the ramp out of t are both > SPIKE_K sigma, opposite sign.

    Requiring both legs makes this far more specific than a single-ramp threshold, so SPIKE_K can
    be lower than RAMP_K.
    """
    nxt = rz.shift(-1)
    return (rz.abs() > SPIKE_K) & (nxt.abs() > SPIKE_K) & (np.sign(rz) != np.sign(nxt))


def _style(ax):
    ax.set_facecolor(C["surface"])
    ax.grid(axis="y", color=C["grid"], lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["grid"])
    ax.tick_params(colors=C["muted"], labelsize=9)


def _save(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, facecolor=C["surface"])
    plt.close(fig)
    return path


def plot_wind_curve(t, fit_mask, flagged, logistic_model, year) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6), facecolor=C["surface"])
    _style(ax)
    ok = fit_mask & ~flagged
    ax.scatter(t.loc[ok, "ws100_mean"], t.loc[ok, "wind_cf"], s=4, alpha=0.25, color=C["blue"], lw=0,
               label="Hours used for fitting")
    out = flagged
    ax.scatter(t.loc[out, "ws100_mean"], t.loc[out, "wind_cf"], s=26, facecolors="none",
               edgecolors=C["critical"], lw=1.2, label=f"Outlier check flags ({int(out.sum())} h)")
    excl = ~fit_mask
    if excl.any():
        ax.scatter(t.loc[excl, "ws100_mean"], t.loc[excl, "wind_cf"], s=30, marker="x", color=C["muted"],
                   lw=1.2, label="Excluded (suspect / missing / curtailment)")
    v = np.linspace(0, t["ws100_mean"].max() + 0.5, 300)
    ax.plot(v, logistic(v, *logistic_model.params), color=C["orange"], lw=2.5, label="Fitted logistic curve")
    ax.set_xlabel("Mean 100 m wind speed over the wind points (m/s, ERA5)", color=C["muted"])
    ax.set_ylabel("Wind capacity factor (Fingrid 75 / 268)", color=C["muted"])
    ax.set_xlim(left=0)
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=C["text"])
    ax.set_title(f"Wind capacity factor vs wind speed, {year} (hourly)", loc="left", fontsize=12, color=C["text"])
    fig.text(0.01, 0.005, "Sources: Fingrid / data.fingrid.fi (CC BY 4.0); Open-Meteo ERA5 (CC BY 4.0).",
             fontsize=8, color=C["muted"])
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return _save(fig, f"wind_cf_vs_speed_{year}.png")


def pick_weeks(t: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Monday starts of the windiest and calmest full Monday-Sunday weeks (mean actual wind_cf)."""
    local = t["wind_cf"].tz_convert("Europe/Helsinki")
    weekly = local.resample("W-SUN").agg(["mean", "count"])
    weekly = weekly[weekly["count"] >= 160]  # full weeks only
    to_start = lambda end: (end - pd.Timedelta(days=6)).normalize()
    return to_start(weekly["mean"].idxmax()), to_start(weekly["mean"].idxmin())


def plot_wind_weeks(t, oof, year) -> Path:
    windy, calm = pick_weeks(t)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharey=True, facecolor=C["surface"])
    series = [("actual", "Actual (Fingrid)", C["blue"]), ("logistic", "Logistic, mean wind speed", C["orange"]),
              ("per_point", "Per-point weighted", C["aqua"]), ("gbm", "Gradient boosting", C["yellow"])]
    for ax, (label, start) in zip(axes, [("Windiest week", windy), ("Calmest week", calm)]):
        _style(ax)
        a, b = start.tz_convert("UTC"), (start + pd.Timedelta(days=7)).tz_convert("UTC")
        w = (t.index >= a) & (t.index < b)
        data = {"actual": t.loc[w, "wind_cf"], **{k: oof[k][w] for k in ("logistic", "per_point", "gbm")}}
        for key, name, color in series:
            ax.plot(data[key].index.tz_convert("Europe/Helsinki"), data[key], color=color, lw=2 if key == "actual" else 1.6,
                    label=name)
        mae = {k: np.mean(np.abs(data[k] - data["actual"])) for k in ("logistic", "per_point", "gbm")}
        ax.set_title(f"{label}: {start:%a %d %b} - {start + pd.Timedelta(days=6):%a %d %b %Y}   "
                     f"(MAE logistic {mae['logistic']:.3f}, per-point {mae['per_point']:.3f}, GBM {mae['gbm']:.3f})",
                     loc="left", fontsize=10.5, color=C["text"])
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%a %d", tz="Europe/Helsinki"))
        ax.set_ylabel("Wind capacity factor", color=C["muted"])
        ax.axhline(0, color=C["muted"], lw=0.8)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, 1.22), ncol=4, frameon=False, fontsize=9, labelcolor=C["text"])
    fig.suptitle(f"Wind capacity factor: actual vs out-of-fold predictions ({year}, local time)", x=0.06, ha="left",
                 fontsize=13, color=C["text"], y=0.995)
    fig.text(0.06, 0.005, "Predictions from leave-one-month-out CV: the model never saw the month it predicts.",
             fontsize=8, color=C["muted"])
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    return _save(fig, f"wind_weeks_{year}.png")


def plot_solar_monthly(daily_actual, daily_pred, year) -> Path:
    m_act = daily_actual.resample("MS").sum() / 1000  # GWh
    m_pred = daily_pred.resample("MS").sum() / 1000
    fig, ax = plt.subplots(figsize=(10, 5.5), facecolor=C["surface"])
    _style(ax)
    x = np.arange(len(m_act))
    wdt = 0.38
    ax.bar(x - wdt / 2 - 0.01, m_act.to_numpy(), wdt, color=C["blue"], label="Fingrid 248 (forecast series)")
    ax.bar(x + wdt / 2 + 0.01, m_pred.to_numpy(), wdt, color=C["orange"], label="pvlib model, scaled")
    ax.set_xticks(x, m_act.index.strftime("%b"))
    ax.set_ylabel("GWh per month", color=C["muted"])
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=C["text"])
    ax.set_title(f"Solar output by month, {year}: Fingrid vs model", loc="left", fontsize=12, color=C["text"])
    fig.text(0.01, 0.005, "Scale factor fitted on daily totals. Sources: Fingrid / data.fingrid.fi (CC BY 4.0); "
             "Open-Meteo ERA5 (CC BY 4.0); pvlib.", fontsize=8, color=C["muted"])
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return _save(fig, f"solar_monthly_{year}.png")


# ---------------------------------------------------------------- main

def run(year: int) -> None:
    pd.set_option("display.width", 160)
    t = pd.read_parquet(PROCESSED_DIR / f"hourly_{year}.parquet")
    points = list(weather.load_locations()["wind"])
    fit_mask = wind_fit_mask(t, points)
    print(f"\n=== Wind: {int(fit_mask.sum())} fit hours, {int((~fit_mask).sum())} excluded "
          f"(suspect/missing: {int((~t['flag_wind'].isin(FIT_FLAGS)).sum())}, "
          f"curtailment label: {int(t['curtailment'].sum())}, price <= 0: {int((t['price_eur_mwh'] <= 0).sum())}) ===")

    makers = {"logistic": LogisticModel, "per_point": lambda: PerPointModel(points), "gbm": lambda: GBMModel(points)}
    oof = {k: leave_one_month_out(mk, t, fit_mask) for k, mk in makers.items()}
    full = {k: mk().fit(t[fit_mask], t.loc[fit_mask, "wind_cf"]) for k, mk in makers.items()}

    y = t.loc[fit_mask, "wind_cf"]
    overall = pd.DataFrame({k: {**scores(y, oof[k][fit_mask]),
                                **{f"{m} in-sample": v for m, v in scores(y, full[k].predict(t[fit_mask])).items()}}
                            for k in makers}).T
    print("\nLeave-one-month-out CV (and in-sample for reference):")
    print(overall.round(4).to_string())

    month = t.loc[fit_mask, "month_local"]
    by_month = {}
    for k in makers:
        g = pd.DataFrame({"y": y, "p": oof[k][fit_mask], "m": month}).groupby("m")
        by_month[(k, "MAE")] = g.apply(lambda d: mean_absolute_error(d["y"], d["p"]))
        by_month[(k, "R2")] = g.apply(lambda d: r2_score(d["y"], d["p"]))
    by_month = pd.DataFrame(by_month)
    by_month["mean_cf"] = pd.DataFrame({"y": y, "m": month}).groupby("m")["y"].mean()
    print("\nBy month (out-of-fold):")
    print(by_month.round(3).to_string())

    best = min(("logistic", "per_point"), key=lambda k: overall.loc[k, "MAE"])
    resid = t["wind_cf"] - oof[best]
    flagged, sigma, threshold = outlier_report(t, resid, fit_mask)
    r_fit = resid[fit_mask]
    print(f"\n=== Outlier check (best interpretable model: {best}, out-of-fold residuals) ===")
    print(f"robust sigma = {sigma:.4f} cf; plain std = {r_fit.std():.4f}; "
          f"tail counts |r| > 3/4/5/6 sigma: "
          + ", ".join(f"{(r_fit.abs() > k * sigma).sum()}" for k in (3, 4, 5, 6)))
    print(f"threshold = {OUTLIER_K:g} x robust sigma = {threshold:.3f} cf "
          f"(~{threshold * t['wind_capacity'].mean():.0f} MW at mean capacity)")
    print(f"flagged hours: {int(flagged.sum())} (of which already excluded from fit: {int((flagged & ~fit_mask).sum())}, "
          f"curtailment: {int((flagged & t['curtailment']).sum())})")
    top = pd.DataFrame({"actual_cf": t["wind_cf"], "pred_cf": oof[best], "resid": resid,
                        "resid_MW": resid * t["wind_capacity"], "ws100_mean": t["ws100_mean"],
                        "temp_mean": t["temp_mean"], "flag_wind": t["flag_wind"]})[flagged]
    print(top.reindex(top["resid"].abs().sort_values(ascending=False).index).head(20).round(3).to_string())
    if flagged.any():
        f = flagged[flagged]
        runs = (f.index.to_series().diff() != pd.Timedelta("1h")).cumsum()
        spans = f.index.to_series().groupby(runs.values).agg(["first", "last", "size"])
        print(f"\nFlagged hours form {len(spans)} episodes; longest:")
        print(spans.sort_values("size", ascending=False).head(8).to_string())

    rz, rsigma = ramp_z(t, oof[best], fit_mask)
    jumps = rz.abs() > RAMP_K
    spikes = spike_hours(rz)
    print(f"\n=== Ramp check (report only; hour-to-hour change in actual minus change in prediction) ===")
    print(f"robust sigma = {rsigma:.4f} cf/h; |z| > 5/6/8/10: "
          + ", ".join(f"{int((rz[fit_mask].abs() > k).sum())}" for k in (5, 6, 8, 10)))
    print(f"jumps |z| > {RAMP_K:g}: {int(jumps.sum())} h; spikes (opposite-sign ramps > {SPIKE_K:g} sigma in "
          f"consecutive hours): {int(spikes.sum())} h")
    rep = pd.DataFrame({"ramp_z": rz, "actual_cf": t["wind_cf"], "pred_cf": oof[best], "flag_wind": t["flag_wind"],
                        "jump": jumps, "spike": spikes})[jumps | spikes]
    print(rep.round(3).to_string())

    # solar
    raw = solar_raw_per_mw(t, year)
    sol = fit_solar(t, raw)
    d_act, d_pred = sol["daily_actual"], sol["daily_pred"]
    pred_mw = sol["scale"] * raw * t["solar_capacity"]
    print("\n=== Solar ===")
    print(f"assumptions: capacity split equally over {len(weather.load_locations()['solar'])} city points, "
          f"tilt {SOLAR_TILT_DEG:g} deg, azimuth {SOLAR_AZIMUTH_DEG:g} deg (south), Hay-Davies transposition, "
          f"SAPM cell temperature ({SAPM_MOUNT}), PVWatts DC, gamma {GAMMA_PDC}")
    print(f"scale factor = {sol['scale']:.3f}  (raw DC per MW -> Fingrid 248 MW)")
    print(f"daily-total R2 = {r2_score(d_act, d_pred):.3f}, daily MAE = {mean_absolute_error(d_act, d_pred):.0f} MWh/day")
    print(f"annual: Fingrid 248 = {t['solar'].sum() / 1e6:.3f} TWh, model = {pred_mw.sum() / 1e6:.3f} TWh")
    print(f"hourly R2 (for reference; 248 has the 10-11 UTC artefact) = {r2_score(t['solar'], pred_mw):.3f}")
    m = pd.DataFrame({"fingrid_GWh": d_act.resample("MS").sum() / 1000, "model_GWh": d_pred.resample("MS").sum() / 1000})
    m["ratio"] = m["model_GWh"] / m["fingrid_GWh"]
    m.index = m.index.strftime("%b")
    print(m.round(2).T.to_string())

    # save
    MODELS_DIR.mkdir(exist_ok=True)
    out = {
        "year": year,
        "fit_hours": int(fit_mask.sum()),
        "wind": {
            "target": "wind_cf = Fingrid 75 / Fingrid 268",
            "logistic": full["logistic"].to_dict(),
            "per_point": full["per_point"].to_dict(),
            "cv_leave_one_month_out": overall.round(5).to_dict(orient="index"),
            "outlier_check": {"model": best, "k": OUTLIER_K, "robust_sigma": float(sigma),
                              "threshold_cf": float(threshold), "flagged_hours": int(flagged.sum())},
            "ramp_check": {"robust_sigma": rsigma, "jump_k": RAMP_K, "spike_k": SPIKE_K,
                           "jump_hours": int(jumps.sum()), "spike_hours": int(spikes.sum())},
            "excluded_from_fit": {"suspect_or_missing": int((~t["flag_wind"].isin(FIT_FLAGS)).sum()),
                                  "curtailment": int(t["curtailment"].sum())},
        },
        "solar": {
            "target": "Fingrid 248 (forecast series), daily totals",
            "scale": sol["scale"],
            "points": list(weather.load_locations()["solar"]),
            "tilt_deg": SOLAR_TILT_DEG, "azimuth_deg": SOLAR_AZIMUTH_DEG, "transposition": "haydavies",
            "cell_temperature": f"sapm/{SAPM_MOUNT}", "dc_model": "pvwatts", "gamma_pdc": GAMMA_PDC,
            "daily_r2": float(r2_score(d_act, d_pred)),
            "annual_twh_fingrid": float(t["solar"].sum() / 1e6), "annual_twh_model": float(pred_mw.sum() / 1e6),
        },
    }
    path = MODELS_DIR / f"renewables_{year}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {path}")

    # Solar profiles per MW installed, for the scenario engine (config/scenarios.yaml: solar.profile)
    profiles = pd.DataFrame({
        "solar_cf_fingrid": t["solar_cf_fingrid"],  # baseline: Fingrid 248 / 267 (artefact-corrected if set)
        "solar_cf_fingrid_raw": t["solar_cf_fingrid_raw"],  # uncorrected 248 / 267
        "solar_cf_pvlib": sol["scale"] * raw,  # alternative: pvlib, weather-driven
    })
    ppath = PROCESSED_DIR / f"solar_profiles_{year}.parquet"
    profiles.to_parquet(ppath)
    print(f"Wrote {ppath}  (annual mean cf: " + ", ".join(f"{c} {v:.4f}" for c, v in profiles.mean().items()) + ")")
    for p in (plot_wind_curve(t, fit_mask, flagged, full["logistic"], year), plot_wind_weeks(t, oof, year),
              plot_solar_monthly(d_act, d_pred, year)):
        print(f"Wrote {p}")


def wind_fit_mask(t: pd.DataFrame, points: list[str]) -> pd.Series:
    """Hours used to fit the wind curve: clean flags, not curtailed, and price > 0.

    Wind is often curtailed when the day-ahead price is <= 0, so those hours are excluded and
    the fitted curve describes uncurtailed output. Curtailment is a price-based rule in the
    scenario engine instead.
    """
    return (t["flag_wind"].isin(FIT_FLAGS) & t["flag_wind_capacity"].isin(FIT_FLAGS)
            & ~t["curtailment"] & (t["price_eur_mwh"] > 0)
            & t["wind_cf"].notna() & t[[f"ws100_{p}" for p in points]].notna().all(axis=1))


def run_wind_years(years: list[int], reference_year: int = 2024) -> None:
    """Wind on several years: leave-one-year-out CV vs a model fitted on reference_year only.

    For each test year Y:
      single-year: logistic fitted on reference_year only (for Y == reference_year itself the
                   leave-one-month-out predictions are used, so it is never scored in-sample)
      pooled LOYO: logistic (and GBM benchmark) fitted on all years except Y
    """
    pd.set_option("display.width", 160)
    points = list(weather.load_locations()["wind"])
    t = pd.concat([pd.read_parquet(PROCESSED_DIR / f"hourly_{y}.parquet") for y in years])
    t["year"] = t.index.year
    fit_mask = wind_fit_mask(t, points)
    print(f"\n=== Wind, {years[0]}-{years[-1]}: {int(fit_mask.sum())} fit hours, {int((~fit_mask).sum())} excluded ===")

    # each year's own curve, to see drift in the fleet (newer, taller turbines)
    per_year = {}
    for y in years:
        m = fit_mask & (t["year"] == y)
        p = fit_logistic(t.loc[m, "ws100_mean"].to_numpy(), t.loc[m, "wind_cf"].to_numpy())
        per_year[y] = {"mean_cf": t.loc[m, "wind_cf"].mean(), "mean_ws100": t.loc[m, "ws100_mean"].mean(),
                       "capacity_MW_mean": t.loc[m, "wind_capacity"].mean(), **dict(zip(["L", "k", "v0", "c"], p)),
                       "cf_at_8ms": logistic(8.0, *p)}
    print("\nPer-year logistic fit (in-sample, for drift only):")
    print(pd.DataFrame(per_year).T.round(3).to_string())

    ref = t[t["year"] == reference_year]
    ref_mask = fit_mask[t["year"] == reference_year]
    single = LogisticModel().fit(ref[ref_mask], ref.loc[ref_mask, "wind_cf"])
    ref_lomo = leave_one_month_out(LogisticModel, ref, ref_mask)

    pred = {k: pd.Series(np.nan, index=t.index) for k in ("single_year", "pooled_loyo", "gbm_loyo")}
    for y in years:
        test = t["year"] == y
        train = fit_mask & ~test
        pred["pooled_loyo"][test] = LogisticModel().fit(t[train], t.loc[train, "wind_cf"]).predict(t[test])
        pred["gbm_loyo"][test] = GBMModel(points).fit(t[train], t.loc[train, "wind_cf"]).predict(t[test])
        pred["single_year"][test] = ref_lomo.to_numpy() if y == reference_year else single.predict(t[test])

    rows = {}
    for y in years + ["all"]:
        m = fit_mask if y == "all" else fit_mask & (t["year"] == y)
        yv = t.loc[m, "wind_cf"]
        rows[y] = {f"{k} {s}": v for k in pred for s, v in scores(yv, pred[k][m]).items()}
    res = pd.DataFrame(rows).T
    print(f"\nOut-of-sample scores by test year (single_year = fitted on {reference_year} only):")
    print(res.round(4).to_string())

    pooled = LogisticModel().fit(t[fit_mask], t.loc[fit_mask, "wind_cf"])
    MODELS_DIR.mkdir(exist_ok=True)
    path = MODELS_DIR / f"renewables_wind_{years[0]}_{years[-1]}.json"
    path.write_text(json.dumps({
        "years": years, "fit_hours": int(fit_mask.sum()),
        "logistic_pooled": pooled.to_dict(),
        "per_year_logistic": {str(y): {k: float(v) for k, v in d.items()} for y, d in per_year.items()},
        "scores_by_test_year": {str(k): {c: float(v) for c, v in r.items()} for k, r in res.round(5).iterrows()},
    }, indent=2))
    print(f"\nWrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--year", type=int, help="full single-year run (wind CV by month, checks, solar)")
    which.add_argument("--wind-years", type=int, nargs="+", help="multi-year wind run, leave-one-year-out CV")
    args = p.parse_args()
    if args.year:
        run(args.year)
    else:
        run_wind_years(sorted(args.wind_years))


if __name__ == "__main__":
    main()
