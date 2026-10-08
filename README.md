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
