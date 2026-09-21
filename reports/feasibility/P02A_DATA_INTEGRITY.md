# P02A — BR-PVGen data integrity and schema audit

P02A decision: **PASS**

## Documented expectations

- Official DOI `10.5281/zenodo.21511487` (Zenodo record 21511487).
- Distributed objects: metadata CSV, inverter zip, meteorological zip, Readme.
- Readme: 51 plants, 15-minute ISO-8601 timestamps, inverter + solar station files per plant.
- `tracker_albedo_index` inventoried only; not used for QC/repair.

## Observed evidence

- Raw root: `data/raw/br_pvgen`
- Acquisition: local copy of official Zenodo objects; md5 verified against record 21511487
- Found files: .gitkeep, BR-PVGen_inverter.zip, BR-PVGen_metadata.csv, BR-PVGen_meteorological.zip, Readme.txt
- Metadata rows: 51; plants: 51
- Inverter rows: 14400480
- Meteorological rows: 1183968
- Matched plants across metadata/inverter/meteo: 51
- Joinability: plant-level temporal panel constructible in principle via official ps_id and datetime

## Discrepancies

- Missing Zenodo objects: none
- Unexpected Zenodo objects: ['.gitkeep']
- Missing inner zip members vs PS_XXX.csv (51 plants): none
- Unexpected inner zip members: none
- Readme documents `plant_{ps}_inverter.csv` / `plant_{ps}_solar_station.csv`; Zenodo zips contain `BR-PVGen_{family}/PS_XXX.csv`. Packaging discrepancy only; join keys remain official `id`/`ps_id` and `datetime`.

## Implications for later P-02 tasks

- P-02B may use official `ps_id` / metadata fields only if P02A is PASS.
- P-02C owns plant-level panel construction; this audit does not build that panel.
- `tracker_albedo_index` remains reserved for P-02H.

## P02A PASS/FAIL

**PASS**

