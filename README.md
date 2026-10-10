# Vaasa Data Center and Finnish Electricity Prices

An open-source tool for estimating how additional data center demand could affect Finnish electricity prices.

## Project Status

In development. Initial data collection targets 2024.

## Methodology

Weather-informed wind and solar estimates are calibrated against Fingrid production data. A residual-load price model supports historical counterfactuals and scenario projections.

## Data Sources and Attribution

- Fingrid Open Data: CC BY 4.0. Attribute as “Source Fingrid / data.fingrid.fi, license CC 4.0 BY”.
- Open-Meteo Historical Weather API: CC BY 4.0.
- ENTSO-E Transparency Platform: attribution to ENTSO-E.

## Setup

Install dependencies from `requirements.txt`. Copy `.env.example` to `.env` and add the API credentials needed by each data source.

## Data and Usage

Fetched and processed data are stored under `data/` and are not committed to the repository. Usage instructions will be added with the fetch and analysis workflows.

## Limitations

Finland is modeled as a single bidding zone. Scenario results depend on explicit assumptions for data center size, load factor, ramp-up, and renewable power purchase agreements; this is a statistical estimate, not a full electricity-market model.

Solar data:

- **Solar output is not measured.** The baseline solar series is Fingrid's solar power generation forecast updated every 15 minutes (dataset 248). It is a forecast-based estimate built from weather forecasts and assumed capacity and locations, not a measurement. Most small-scale PV is behind the meter and unmetered.
- **Solar capacity is an estimate.** The capacity used for solar capacity factors is the capacity Fingrid assumes in its forecast (dataset 267). Fingrid updates it manually and states that it is not the official amount of installed capacity.
- Dataset 248 has a timing artefact around 09–12 UTC in spring and summer (too high at 09, too low at 11 UTC). Daily totals are unaffected. A total-preserving correction is documented in `src/process/solar_artefact.py` but not applied.
- A weather-driven alternative profile (pvlib, 15° tilt, south-facing, scaled to the daily totals of dataset 248) is kept for sensitivity analysis. It overestimates winter output relative to dataset 248, likely because of snow cover.

Wind data: the wind capacity used for capacity factors is the capacity in Fingrid's wind forecast (dataset 268), i.e. capacity actually producing, not official installed capacity. Projections will use official installed capacity statistics.
