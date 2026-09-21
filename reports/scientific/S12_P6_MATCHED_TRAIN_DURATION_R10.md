# S12 P6 / R10 — matched TRAIN elapsed-duration sensitivity

Secondary arm. Canonical TEST evaluation. Strict primary CS4 IQR (no R9). Trailing matched windows; D_pair = min(source, target) TRAIN elapsed days; both endpoints included. Equal-duration tolerance = 1 second.

Status: `S12_P6_R10_MATCHED_DURATION_MATERIALIZED`. R10 computable 85/94 (primary 85); lost 0; new 0.
Plant TRAIN duration days min/median/max 39.916666666666664/61.453125/267.03125. D_pair min/median/max 39.916666666666664/52.947916666666664/266.3645833333333. Shorter: source 47, target 47, equal 0.
RQ1 PB primary 0.009723604857785641 vs R10 0.009746510387605893 (Δ 2.2905529820252055e-05); overlap n=85 Δ 2.2905529820252055e-05.
Delta_OTG_duration median -4.104134175057783e-06; 42/85 positive. Spearman |imbalance| vs Delta rho=0.08878920413824291 CI [-0.14563845222277977, 0.36962430120959006].
RQ2 PB |A| primary 0.026622430133826737 vs R10 0.026967526234026127 (n_pairs 38); overlap pairs 38.
R10 support-retention (n=85 computable directions) min/P25/median/P75/P95/max/mean 0.08204518430439953/0.2570281124497992/0.546875/0.6979785969084423/0.863859649122807/0.9481327800829875/0.5031960484224973.
Full-history duration imbalance days (n=94 planned directions) min/P25/median/P75/P95/max/mean 0.0729166666666714/3.1848958333333375/6.062500000000007/15.132812500000002/23.01041666666667/25.000000000000007/9.131648936170217.
Group-level R10 RQ2: G1 n_pairs=1 n_plants=2 PB |A|=0.00036415743995248107 median |A|=0.00036415743995248107; G2 n_pairs=36 n_plants=9 PB |A|=0.037071058697511414 median |A|=0.01838521271227312; G4 n_pairs=1 n_plants=2 PB |A|=0.008104998942415967 median |A|=0.008104998942415967.
Spearman |full-history duration difference| vs primary |A| rho=0.1940037203195098; vs |A_primary|-|A_R10| rho=-0.014990699201225517.
RQ3: `R10_RQ3_NOT_RUN_MATCHED_DURATION_SCOPE`. Cached matched-window models: 58.
Interpretation: R10 computable 85/94 (primary 85); lost 0; newly computable 0. RQ1 PB primary 0.009723604857785641 vs R10 0.009746510387605893 (Δ 2.2905529820252055e-05). Overlap n=85 PB primary 0.009723604857785641 vs R10 0.009746510387605893 (Δ 2.2905529820252055e-05). Median Delta_OTG_duration -4.104134175057783e-06 (positive = R10 reduced penalty); 42/85 positive, 43/85 negative. Spearman |duration imbalance| vs Delta_OTG rho=0.08878920413824291 CI [-0.14563845222277977, 0.36962430120959006]. RQ2 PB |A| primary 0.026622430133826737 vs R10 0.026967526234026127 (overlap pairs 38). Spearman |duration imbalance| vs |A_primary| rho=0.1940037203195098 ; vs |A| change rho=-0.014990699201225517. R10 remains a sensitivity arm, not a new primary protocol.
Reconciliation max |Δ|=0 ok=True.
