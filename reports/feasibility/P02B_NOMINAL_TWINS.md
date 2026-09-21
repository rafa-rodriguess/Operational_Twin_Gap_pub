# P02B — Exact nominal twin audit

P02B decision: **PASS**

## Observed metadata evidence

- Source `data/raw/br_pvgen/BR-PVGen_metadata.csv` SHA-256 `ba45f4dfb557bbfca37c3ee16c8b7d2a5f0aec9fad4e6b6774be760c4852237c`
- Plants: 51; eligible for exact key: 51; ineligible missing key: 0
- Columns match the P02A-confirmed BR-PVGen metadata schema.
- Official documentation lists inverter measurements in time-series files, not as static metadata.

## Final exact-twin key

- Ordered fields: `is_panel_bifacial`, `nominal_power_mw`, `number_of_panels`, `panel_area_mm2`, `panel_bifaciality_coefficient`, `panel_efficiency_percentage`, `panel_temperature_coefficient`, `structure_type`

## Included attributes and rationale

- Every available static technical configuration attribute from official metadata is included.
- `is_panel_bifacial` and `panel_bifaciality_coefficient` are both retained: the data dictionary defines distinct semantics (boolean class vs rear/front ratio). Sample constancy is not a documented redundant encoding.

## Excluded attributes and rationale

- `id`: plant identifier.
- `brazil_federative_unit`: contextual state only; not used to form or split groups.

## Canonical equality representation

- Decimal fields: lexical CSV token, strip, `decimal.Decimal`, `format(normalize(), 'f')`. No binary float, no tolerance, no chosen decimal places.
- `number_of_panels`: canonical integer.
- `is_panel_bifacial`: lowercase `true`/`false`.
- `structure_type`: trim whitespace only.
- Missing key values never equal missing; plants with any missing key field are ineligible.

## Missing/unobserved technical metadata

- No inverter model, inverter nominal capacity, or inverter quantity as static metadata.
- No orientation/azimuth or tilt in official metadata.
- Inverter IDs exist only in inverter time-series; unused for P02B.

## Exact twin groups

- `TW_04b9f5d95694` n=2 states=["RJ"] members=PS_002, PS_003 key={"is_panel_bifacial":"true","nominal_power_mw":"1","number_of_panels":"2560","panel_area_mm2":"6688.93184","panel_bifaciality_coefficient":"0.7","panel_efficiency_percentage":"20.7","panel_temperature_coefficient":"0.34","structure_type":"FIXED"}
- `TW_4467a039e00b` n=10 states=["BA","GO","MS"] members=PS_042, PS_043, PS_044, PS_045, PS_046, PS_047, PS_048, PS_049, PS_050, PS_051 key={"is_panel_bifacial":"true","nominal_power_mw":"2.5","number_of_panels":"5070","panel_area_mm2":"15749.20464","panel_bifaciality_coefficient":"0.7","panel_efficiency_percentage":"21.2","panel_temperature_coefficient":"0.34","structure_type":"TRACKER"}
- `TW_7b33a493697b` n=8 states=["SP"] members=PS_004, PS_017, PS_020, PS_022, PS_023, PS_024, PS_026, PS_030 key={"is_panel_bifacial":"true","nominal_power_mw":"2","number_of_panels":"4480","panel_area_mm2":"13916.45696","panel_bifaciality_coefficient":"0.7","panel_efficiency_percentage":"21.1","panel_temperature_coefficient":"0.34","structure_type":"FIXED"}
- `TW_88a108921af7` n=2 states=["SP"] members=PS_035, PS_039 key={"is_panel_bifacial":"true","nominal_power_mw":"4","number_of_panels":"8160","panel_area_mm2":"25347.83232","panel_bifaciality_coefficient":"0.7","panel_efficiency_percentage":"21.2","panel_temperature_coefficient":"0.34","structure_type":"TRACKER"}
- `TW_f8e8375afe38` n=8 states=["SP"] members=PS_018, PS_019, PS_021, PS_025, PS_027, PS_029, PS_031, PS_032 key={"is_panel_bifacial":"true","nominal_power_mw":"3","number_of_panels":"6720","panel_area_mm2":"20874.68544","panel_bifaciality_coefficient":"0.7","panel_efficiency_percentage":"21.1","panel_temperature_coefficient":"0.34","structure_type":"FIXED"}

## Pair structure

- Exact twin groups: 5
- Plants in groups: 30
- Eligible singletons: 21
- Undirected pairs: 103
- Directional transfers: 206

## P02B PASS/FAIL

**PASS**

