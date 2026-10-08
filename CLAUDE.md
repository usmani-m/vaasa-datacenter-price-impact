# Vaasa Data Center → Finnish Electricity Price Tool

Open-source tool + dashboard estimating how Microsoft's planned data center in the GigaVaasa industrial zone (Vaasa/Mustasaari, Finland) could affect Finnish electricity prices — historically (counterfactual) and in future years (projections). Purpose: portfolio piece and reusable open-source tool.

## Core framing
- Finland is a single bidding zone (FI). The data center adds load to the whole Finnish market; the effect shows in the national day-ahead price. Use national data for demand/production/price; regional data (Ostrobothnia etc.) only for weather → renewables modelling.
- The data center's capacity (MW) is NOT public. Treat size, load factor, ramp-up timeline and any matching renewable PPA as explicit, user-editable scenario assumptions.

## Method (the chain)
1. Weather → renewables: hourly wind speed (100 m) and irradiance at a few points in the main Finnish wind/solar areas → national wind and solar output, **calibrated against Fingrid actual production** per installed MW (not a generic power curve alone).
2. Residual load = demand − wind − solar.
3. Price model: price as a function of residual load (+ net imports as second variable). Start simple and interpretable (binned / piecewise curve); validate on a held-out year. Flag or allow excluding 2022 (energy crisis).
4. Scenario engine: add data center load → new residual load → new price → Δprice by hour / month / year.
5. Projections (v2): future-year system assumptions (demand growth, installed wind/solar, nuclear, interconnectors, DC ramp-up) × ensemble of historical weather years → price distribution per future year (median, P10–P90), with vs without data center.
   - Do NOT extrapolate prices years ahead with a time-series ML model; use ML only inside components (e.g. weather → output).
6. Optional v3: short-term (day-ahead) price forecast module.

## Data sources
| Layer | Source | Access | License / attribution |
|---|---|---|---|
| Consumption, production by type, net import/export, cross-border flows | Fingrid Open Data (data.fingrid.fi), REST API at api.fingrid.fi | Free registration, API key in header `x-api-key`; ~10,000 requests/24 h — fetch in chunks | CC BY 4.0 — "Source Fingrid / data.fingrid.fi, license CC 4.0 BY" |
| Day-ahead prices (FI), backup generation data | ENTSO-E Transparency Platform (`entsoe-py`) | Security token (registration + email to transparency@entsoe.eu, subject "RESTful API access") | Attribute ENTSO-E |
| Weather (prototype) | Open-Meteo Historical API (ERA5-based) | No key | CC BY 4.0 |
| Weather (citable) | ERA5 via Copernicus CDS | Free account + key | Copernicus licence, attribute |
| Station checks | FMI open data | No key | CC BY 4.0 |
| Solar sanity check | PVGIS | No key | Attribute JRC |
| Installed wind/solar capacity by year | Finnish Wind Power Association stats etc. | Manual CSV with source notes | Cite in CSV |

Notes:
- Fingrid national production dataset (ID 74) is 15-min, but hourly before 2023-06-13. Look up other dataset IDs on data.fingrid.fi rather than guessing.
- Per-border 15-min flow datasets (e.g. FI–SE1) only start 2025-09-30; use net import/export for long history.
- Avoid Nord Pool data (restrictive licence).

## Conventions
- Python. pandas, pvlib, scikit-learn / statsmodels.
- Everything resampled to **hourly, UTC** timestamps; stored as Parquet.
- `data/` is git-ignored. Repo ships fetch code, not raw data. A small preprocessed sample may ship so the dashboard works out of the box.
- API keys in `.env` (git-ignored); provide `.env.example`.
- Scenario and future-year assumptions live in `config/` (YAML), clearly labelled as assumptions with sources.
- Fetchers must be defensive: retries, chunking, graceful handling of API downtime, caching of already-fetched ranges.
- Time window: 2019–2025 for history; start development with 2024 only.

## Repo layout (target)
```
config/        scenarios.yaml, future_years.yaml, locations.yaml, capacity.csv
src/fetch/     fingrid.py, entsoe.py, weather.py
src/process/   build_hourly_table.py
src/models/    renewables.py, price.py
src/scenarios/ engine.py, projections.py
notebooks/     exploration + calibration
dashboard/     static site (GitHub Pages) reading precomputed scenario results
tests/
```

## Phases
0. Repo skeleton, licence (MIT or Apache-2.0), .env.example, README outline
1. Fetch scripts: Fingrid + Open-Meteo first; ENTSO-E prices once token arrives
2. Single clean hourly table + sanity plots
3. Renewables model + validation (error by season)
4. Price model + held-out-year validation
5. Scenario engine (historical counterfactual); then projection engine (v2)
6. Dashboard: inputs (DC MW, PPA toggle, weather year / target year); outputs (price-duration curve with/without DC, monthly Δprice, worst hours, renewables breakdown, fan chart for projections)
7. README: methodology, data attribution, limitations (single bidding zone, no hydro strategy modelling, statistical not full market model)

First milestone: phases 1–5 end-to-end for 2024 with deliberately crude models + one chart. Then improve one component at a time.
