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
- Dataset 248 has a timing artefact around 09–12 UTC in spring and summer, strongest in 2024 (too high at 09, too low at 11 UTC). It is corrected by default: each day's 09–12 UTC total is redistributed using the shape of Fingrid's day-ahead solar forecast (dataset 247), which keeps daily totals unchanged. The setting is `solar.artefact_correction` in `config/scenarios.yaml`, and uncorrected values are kept alongside.
- Fingrid's solar capacity (dataset 267) starts on 2019-04-23, so solar capacity factors are unavailable for January–April 2019.
- A weather-driven alternative profile (pvlib, 15° tilt, south-facing, scaled to the daily totals of dataset 248) is kept for sensitivity analysis. It overestimates winter output relative to dataset 248, likely because of snow cover.

Wind data: the wind capacity used for capacity factors is the capacity in Fingrid's wind forecast (dataset 268), i.e. capacity actually producing, not official installed capacity. Projections will use official installed capacity statistics.

The wind curve is fitted on 2023–2025 and excludes hours with a day-ahead price of zero or below, because wind farms are often curtailed then; curtailment is handled as a price-based rule in the scenario engine instead. Even without those hours, output per MW at a given wind speed keeps falling: at 8 m/s the fitted capacity factor drops from 0.57 (2021) to 0.52 (2023), 0.50 (2024) and 0.48 (2025). This is probably a measurement effect, from the capacity figure (dataset 268) and from new wind farms being built away from the weather points, not turbines getting worse. A model fitted on older years would overestimate recent output.

Price model: price is modelled from residual load, fitted on 2023–2025 and excluding the 2022 energy crisis. The tool shows a binned supply curve alongside a gradient-boosting model whose price can only rise with residual load. The estimated effect of extra load keeps imports, hydro and nuclear output fixed. In reality these respond to higher prices, so the result is a statistical estimate, not a simulation of the market.

Price coupling: Finland often has the same price as Sweden. In those hours extra Finnish demand is largely met by imports, so the plain price slope overstates its effect. The coupling-aware estimate (`src/models/coupling.py`) classifies each hour by comparing the Finnish price with SE1, SE3 and Estonian prices and with import capacity. In coupled hours, extra load up to the spare import capacity is met by imports; anything beyond that, and all load in price-separated hours, follows the Finnish price curve. Import capacities come from Fingrid's official day-ahead values. Since the Nordic day-ahead market moved to flow-based coupling on 29 October 2024, there are no official values for the Swedish borders, and documented fixed values are used instead (`config/coupling.yaml`). They are approximate.

How hours are classified (`config/coupling.yaml`):

- **Coupled with Sweden.** Finland has the same price as SE1, or the same price as SE3 while differing from SE1. In the second case the link to northern Sweden is full but the link to central Sweden is not, so extra load can still be imported, and both cases count as coupled.
- **Coupled only with Estonia.** Finland and Estonia then form a small price area cut off from the Nordic market, so extra demand still raises their joint price. These hours are treated as separated and follow the Finnish price curve.
- **Finland cheaper than Sweden (export-constrained).** Extra Finnish load mainly reduces exports, so the price increase is capped at the gap to the SE1 price: once Finland reaches the Swedish price, the two are coupled again.
- **Small price differences after 29 October 2024.** Flow-based coupling often leaves small price differences between zones that are not really congested. In this period, a difference of up to 2 €/MWh still counts as coupled, provided that border has at least 300 MW of spare import capacity.

Estimated price effect of a flat +300 MW (fitted on 2023–2025):

| Year | Central estimate | Range | Without coupling (upper bound) |
|---|---|---|---|
| 2023 | +6.2 €/MWh | +3.4 to +6.2 | +9.1 |
| 2024 | +4.7 €/MWh | +3.4 to +4.7 | +8.4 |
| 2025 | +5.2 €/MWh | +2.1 to +5.2 | +7.9 |

The central estimate uses the gradient-boosting model. The range also includes the binned supply curve, and a structural variant in which the imported part of the load raises the price by Finland's share of Finnish and Swedish (SE1 + SE3) load times the Finnish price slope. The result depends on the year: in 2023, 80% of hours were coupled with Sweden, compared with 53% in 2025, when far more hours were price-separated under flow-based coupling.

Prices: day-ahead prices for the Finnish bidding zone come from the ENTSO-E Transparency Platform. Since 1 October 2025 they are set per 15 minutes; this tool averages them to hourly values so all years are comparable. On 24 November 2023 (13–22 UTC) the price was −500 €/MWh, which was widely reported as the result of a bidding error rather than market conditions.

Data quality, 2019–2025:

- Hours with frozen data (the same value for 6 hours or more) are marked suspect and excluded from model fitting. Most of them fall between June and September 2023, just after Fingrid switched to 15-minute data.
- Consumption should equal production plus net imports. Before June 2023 it is lower by a steady 25–60 MW; from November 2023 the gap averages about 20 MW. Between June and October 2023 the hourly gap is often several hundred MW, so data from that period is less reliable.
- Net imports include the Finland–Russia link, which carried about 3–9 TWh a year until trade stopped in May 2022.
