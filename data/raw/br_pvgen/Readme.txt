# BR-PVGen — The Brazilian Photovoltaic Generation Dataset

**Operational (inverter-level) and meteorological data from 51 distributed solar photovoltaic plants across Brazil**

---

## 1. Overview

BR-PVGen is an open-access dataset comprising synchronized, high-resolution operational and meteorological measurements from **51 distributed photovoltaic (PV) plants** located across different climatic regions of Brazil. Data were collected through a cloud-based SCADA system between **March 26, 2024 and June 9, 2025**, and are provided at a standardized **15-minute temporal resolution**.

After preprocessing (temporal standardization, weighted moving average aggregation, and targeted interpolation of short gaps), the dataset contains:

- **14,400,480** valid inverter records
- **1,151,232** valid solarimetric (meteorological) station records

The dataset is designed to support research in photovoltaic generation forecasting, performance benchmarking, loss decomposition, fault diagnosis, power quality assessment, and microgrid control validation.

## 2. How to Cite

If you use this dataset, please cite **both** the dataset itself and the associated data-descriptor article.

**Dataset:**
> De Jesus, C. D., Ferreira, R. N., & de Aguiar, E. P. (2026). *BR-PVGen - The Brazilian Photovoltaic Generation Dataset* 

**Associated article:**
> De Jesus, C. D., Ferreira, R. N., & de Aguiar, E. P. "BR-PVGen - The Brazilian Photovoltaic Generation Dataset." *IEEE Access* (in press). DOI to be updated upon publication.


## 3. Authors and Affiliations

| Author | Affiliation | Contact | ORCID |
|---|---|---|---|
| Caian D. de Jesus | Graduate Program in Computational Modeling, Federal University of Juiz de Fora (UFJF), Brazil | caian.jesus@estudante.ufjf.br | — |
| Rhuan N. Ferreira | Information Systems Graduate Program, UFJF, Brazil | rhuan.nascimento@estudante.ufjf.br | — |
| Eduardo P. de Aguiar *(Corresponding author)* | Department of Mechanical Engineering, UFJF, Brazil | eduardo.aguiar@ufjf.br | — |

*(Please insert ORCID iDs before final publication — Zenodo strongly encourages this for author disambiguation.)*

## 4. Dataset Structure

The dataset is organized into three logical layers, distributed as separate archives:

```
BR-PVGen/
├── README.md                              (this file)
├── CITATION.cff
├── metadata.csv                       # one row per PV plant
├── inverter/
│   └── plant_XXX_inverter.csv             # one file per plant
└── solar_station/
    └── plant_XXX_solar_station.csv        # one file per plant
```

`XXX` denotes the anonymized plant identifier (`ps_id`), e.g. `PS_001`. Each plant has one inverter file and one solar station file covering its full period of operation within the dataset window.

## 5. Data Dictionary

### 5.1 Metadata (`metadata/metadata.csv`)
Plant-level technical specifications (one row per plant).

| Column | Type | Description |
|---|---|---|
| `id` | Integer | Unique identifier for the PV plant metadata record. |
| `nominal_power_mw` | Float | Installed power capacity of the plant (MW). |
| `is_panel_bifacial` | Boolean | `true` if panels are bifacial, `false` if monofacial. |
| `panel_temperature_coefficient` | Float | Percentage power loss per °C increase above STC. |
| `panel_bifaciality_coefficient` | Float | Rear-to-front efficiency ratio of bifacial panels (0–1). |
| `panel_area_mm2` | Float | Area of a single PV module (mm²). |
| `panel_efficiency_percentage` | Float | Conversion efficiency of PV modules (%). |
| `number_of_panels` | Integer | Total number of PV modules installed. |
| `brazil_federative_unit` | String | Brazilian state where the plant is located (e.g., `"SP"`). |
| `structure_type` | String | PV structure type: `TRACKER` or `FIXED`. |

### 5.2 Inverter (`inverter/plant_XXX_inverter.csv`)
Electrical measurements recorded at the inverter level, at 15-minute resolution.

| Column | Type | Description |
|---|---|---|
| `datetime` | String | Timestamp, ISO 8601 (`YYYY-MM-DDThh:mm:ssZ`). |
| `total_active_power_w` | Float | Total active power output (W). |
| `total_reactive_power_var` | Float | Total reactive power (VAR). |
| `total_dc_power_w` | Float | Total DC input power (W). |
| `internal_temperature_celsius` | Float | Internal inverter temperature (°C). |
| `inverter_id` | Integer | Unique identifier for the inverter within the plant. |
| `ps_id` | String | Unique (anonymized) plant identifier. |
| `interpolated_keys_*` | Boolean | One flag column per variable; `true` if the value at that timestamp was linearly interpolated. |
| `document_count_*` | Integer | One column per variable; number of raw ~5-min samples used in the weighted-moving-average aggregation for that timestamp. |

### 5.3 Solar Station (`solar_station/plant_XXX_solar_station.csv`)
On-site meteorological and irradiance measurements, at 15-minute resolution.

| Column | Type | Description |
|---|---|---|
| `datetime` | String | Timestamp, ISO 8601. |
| `poa_irradiance_wm2` | Float | Plane-of-Array irradiance (W/m²). |
| `ghi_irradiance_wm2` | Float | Global Horizontal Irradiance (W/m²). |
| `gri_irradiance_wm2` | Float | Ground Reflected Irradiance (W/m²). |
| `panel_temperature_celsius` | Float | PV module surface temperature (°C). |
| `ambient_temperature_celsius` | Float | Ambient air temperature (°C). |
| `wind_speed_ms` | Float | Wind speed (m/s). |
| `wind_direction_degrees` | Float | Wind direction (°). |
| `tracker_albedo_index` | Float | Ground albedo index (0–1), reported for tracking systems. |
| `precipitation_accumulated_mm` | Float | Accumulated rainfall (mm). |
| `battery_voltage` | Float | Weather station battery voltage (V), used for station health monitoring. |
| `ps_id` | String | Unique (anonymized) plant identifier. |
| `interpolated_keys_*` | Boolean | Per-variable interpolation flags (see 5.2). |
| `document_count_*` | Integer | Per-variable raw sample count used in aggregation (see 5.2). |

## 6. Data Acquisition and Processing Methodology (Summary)

1. **Acquisition:** local data loggers polled inverters and solarimetric stations every ~5 minutes via the Modbus protocol, and transmitted readings to a cloud API over HTTPS, with local buffering during connectivity outages.
2. **Segregation:** records were split by source into inverter data and solarimetric station data.
3. **Temporal standardization:** irregular raw timestamps were resampled onto a uniform 15-minute grid using a **distance-weighted moving average (WMA)**, which reduces the influence of outliers while preserving local trends.
4. **Interpolation:** short gaps (≤ 15 minutes, bounded by valid neighboring values) in instantaneous variables were filled via targeted linear interpolation; cumulative variables were excluded from this step to avoid distorting their semantics.
5. **Anonymization:** all plant and inverter identifiers were irreversibly replaced with randomly generated anonymous IDs. Precise geographic coordinates are **not** included; location is restricted to the Brazilian federative unit (state) level to protect asset owners while preserving regional climatological resolution.

Full methodological detail, including the theoretical generation model (IEC 61724-2), and the bifacial gain, temperature, clipping, and unavailability loss decomposition, is provided in the associated article (Section III–IV).

## 7. Temporal and Spatial Coverage

- **Period:** March 26, 2024 – June 9, 2025 (coverage varies per plant, as plants were progressively incorporated into the SCADA monitoring system).
- **Resolution:** 15 minutes.
- **Spatial coverage:** 51 plants across multiple Brazilian states and climatic zones.
- **Plant capacity range:** 1.7 MWp – 5 MWp (average ≈ 2.5 MWp).
- **Structure types:** both fixed-tilt and single-axis tracking systems; all systems use bifacial modules.

## 8. Known Limitations

- Temporal coverage per individual plant is not uniform; plants joined the SCADA system at different dates (see the monthly record-count distribution in the associated article, Fig. 3).
- Missing-data rates vary by plant and variable; a per-plant, per-variable completeness summary is provided in the associated article (Fig. 4).
- The dataset spans approximately 14 months and does not yet capture multi-year degradation effects.
- Example analysis/forecasting scripts are not yet bundled with this release; they are planned for a future version (see Section 10).

## 9. License and Terms of Use

This dataset is distributed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. You are free to share and adapt the data for any purpose, including commercial use, **provided that appropriate credit is given** (see Section 2, "How to Cite"). No additional restrictions may be applied.

## 10. Version History

| Version | Date | Notes |
|---|---|---|
| v1.0 | 2026-07-24 | Initial public release on Zenodo, accompanying the IEEE Access submission. |

Future releases may extend the temporal coverage, add new plants/variables, and include example reproducibility scripts/notebooks, as noted in the associated article.

## 11. Acknowledgments

This dataset was produced as part of research conducted at the Federal University of Juiz de Fora (UFJF), Graduate Program in Computational Modeling / Department of Mechanical Engineering, Brazil.

## 12. Key References

- International Electrotechnical Commission. *IEC 61724-2: Photovoltaic system performance – Part 2: Capacity evaluation method.* IEC, 2016.
- Ghosh, S., Roy, J. N., & Chakraborty, C. "A model to determine soiling, shading and thermal losses from PV yield data." *Clean Energy*, 6(4), 372–391, 2022.

## 13. Contact

For questions, corrections, or collaboration inquiries, please contact the corresponding author (Section 3) or open an issue via the Zenodo record's linked communication channel.