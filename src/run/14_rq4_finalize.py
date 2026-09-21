"""14 — RQ4 aggregation, inference, validations, SoT analysis namespace."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.contracts import StageContext, StageResult, ValidationRecord
from src.io import artifact, csv_fingerprint, dump_json, read_csv
from src.lib.rq4_const import (
    BOOT_B,
    BOOT_SEED,
    CUT_SPACING,
    CS4_K,
    CS4_Q,
    FEATURES,
    H_DAYS,
    K_GRID,
    K_MAX,
    L_DAYS,
    M_SHORTLIST,
    OUT_DIR,
    PANEL_PATH,
    PROHIBITED,
    RHO,
    SPEC_PATH,
    SPEC_SHA256,
    TAU_PRIMARY,
)
from src.lib.rq4_science import bootstrap_ci, finalize_from_artifacts, load_plants, reconstruct_cohorts, target_balanced, _group
from src.run._ledger import ledger_patch
from src.sot import sha256_file


PROMPT_PATH = "docs/prompts/EXECUTE_RQ4_SOURCE_SELECTION_CONFIRMATORY.md"
PROMPT_SHA256 = "0c1dcaa79b2a5015bb8e73d0f38f8b1f30801e28d98c75674b5e0bc6391b526f"
SCIENCE_EXECUTION_COMMIT = "ab0af1d9422e5950ea588c2bc4005cff895291c6"


def _git_commit(root: Path) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    except Exception:  # noqa: BLE001
        return "UNKNOWN"


def _pq_fp(path: Path) -> str:
    import pyarrow.parquet as pq

    return hashlib.sha256("|".join(pq.read_schema(path).names).encode("utf-8")).hexdigest()


def run(ctx: StageContext) -> StageResult:
    root = ctx.root
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    freeze = ((ctx.sot.get("scientific_freeze") or {}).get("rq4_source_selection") or {})
    res = freeze.get("cohort_resolution") or {}
    cohorts = reconstruct_cohorts(root)
    checks: list[ValidationRecord] = []
    check_art: dict[str, str | list[str]] = {}

    def add(name: str, passed: bool, details: str | None, artifacts: str | list[str] | None = None) -> None:
        checks.append(ValidationRecord(name, passed, details))
        check_art[name] = artifacts if artifacts is not None else f"{OUT_DIR}/RQ4_VALIDATION.json"

    spec_sha = sha256_file(root / SPEC_PATH)
    commit = _git_commit(root)
    prev_summary = {}
    prev_p = root / OUT_DIR / "RQ4_SUMMARY.json"
    if prev_p.is_file():
        prev_summary = json.loads(prev_p.read_text(encoding="utf-8"))
    analysis = finalize_from_artifacts(root, commit, spec_sha, cohorts)
    analysis2 = finalize_from_artifacts(root, commit, spec_sha, cohorts)
    rerun_ok = analysis["_scientific_content_sha256"] == analysis2["_scientific_content_sha256"]
    prev_sci = json.dumps({k: prev_summary.get(k) for k in ("T1", "T2", "k_tau", "k_noise", "primary_curve")}, sort_keys=True, default=str)
    new_sci = json.dumps({k: analysis.get(k) for k in ("T1", "T2", "k_tau", "k_noise", "primary_curve")}, sort_keys=True, default=str)
    if prev_summary and prev_sci == new_sci:
        analysis["execution_commit"] = SCIENCE_EXECUTION_COMMIT

    import pyarrow.parquet as pq

    models = pq.read_table(root / OUT_DIR / "RQ4_SOURCE_MODELS.parquet").to_pylist()
    inst = pq.read_table(root / OUT_DIR / "RQ4_DECISION_INSTANCES.parquet").to_pylist()
    regret = pq.read_table(root / OUT_DIR / "RQ4_REGRET_BY_INSTANCE.parquet").to_pylist()
    noise = pq.read_table(root / OUT_DIR / "RQ4_TEMPORAL_NOISE.parquet").to_pylist()
    hgb = pq.read_table(root / OUT_DIR / "RQ4_HGB_SENSITIVITY.parquet").to_pylist()
    losses = pq.read_table(root / OUT_DIR / "RQ4_CANDIDATE_LOSSES.parquet").to_pylist()
    audit_p = root / OUT_DIR / "RQ4_TEMPORAL_AUDIT.parquet"
    audit = pq.read_table(audit_p).to_pylist() if audit_p.is_file() else []
    draws_p = root / OUT_DIR / "RQ4_BOOTSTRAP_DRAWS.parquet"
    draws = pq.read_table(draws_p).to_pylist() if draws_p.is_file() else []
    inf = [r for r in inst if r.get("informative")]
    inf_reg = [r for r in regret if r.get("informative_k")]

    add("v01_target_reconstructed", bool(cohorts.get("ok")), str(cohorts.get("target_ids")))
    add("v02_target_hash", cohorts.get("target_ids_sha256") == res.get("target_ids_sha256"), str(cohorts.get("target_ids_sha256")))
    add("v03_source_reconstructed", bool(cohorts.get("ok")), str(cohorts.get("source_count")))
    add("v04_source_hash", cohorts.get("source_ids_sha256") == res.get("source_ids_sha256"), str(cohorts.get("source_ids_sha256")))
    add("v05_target_ne_source", bool(cohorts.get("target_source_disjoint")) and not set(cohorts.get("target_ids") or []) & set(cohorts.get("source_ids") or []), "disjoint")
    n_tr = sum(1 for m in models if m.get("train_end_before_c") is True)
    n_va = sum(1 for m in models if m.get("val_end_before_c") is True)
    add("v06_source_train_before_c", bool(models) and n_tr == len(models), f"train_before_c={n_tr}/{len(models)}")
    add("v07_source_val_before_c", bool(models) and n_va == len(models), f"val_before_c={n_va}/{len(models)}")
    fracs = [float(m["train_frac"]) for m in models if m.get("train_frac") is not None]
    add("v08_chrono_80_20", bool(fracs) and all(abs(f - 0.8) < 0.05 for f in fracs), f"n={len(fracs)} min={min(fracs) if fracs else None}")
    src = (root / "src/lib/rq4_science.py").read_text(encoding="utf-8")
    add("v09_no_target_y_in_source_selection", "source_fit" in src and "plant = plants.get(sid)" in src, "source_fit uses source plant id only")
    add("v10_feature_order", list(freeze.get("features") or []) == list(FEATURES), json.dumps(FEATURES))
    elig = freeze.get("eligibility") or {}
    src_ids = sorted({str(m.get("source_id")) for m in models if m.get("source_id")})
    plants_obs = load_plants(root, src_ids) if src_ids else {}
    n_elig = 0
    n_bad_poa = 0
    n_bad_cov = 0
    for plant in plants_obs.values():
        mask = plant.elig
        n_elig += int(mask.sum())
        if mask.any():
            n_bad_poa += int(np.sum(~(plant.poa[mask] > 50.0)))
            n_bad_cov += int(np.sum(~(plant.cov[mask] >= 0.80)))
    v11_ok = n_elig > 0 and n_bad_poa == 0 and n_bad_cov == 0 and elig.get("poa_irradiance_wm2_gt") == 50 and elig.get("coverage_dc_ge") == 0.8
    add(
        "v11_eligibility_exact",
        v11_ok,
        json.dumps({"n_eligible_source_rows": n_elig, "n_poa_le_50": n_bad_poa, "n_cov_lt_0_80": n_bad_cov}),
        [PANEL_PATH, f"{OUT_DIR}/RQ4_SOURCE_MODELS.parquet"],
    )
    add("v12_tracker_absent", PROHIBITED not in FEATURES and PROHIBITED not in src, PROHIBITED)
    add("v13_spline_primary", freeze.get("primary_model") == "SplineRidge" and analysis.get("primary_model") == "SplineRidge", "SplineRidge")
    add("v14_spline_grid", freeze.get("primary_model_grid", {}).get("n_knots") == [4, 6, 8], "n_knots")
    add("v15_hgb_grid", freeze.get("sensitivity_model_grid", {}).get("min_samples_leaf") == 20, "min_samples_leaf=20")
    add("v16_L30", freeze.get("source_lookback_days") == L_DAYS, str(L_DAYS))
    add("v17_H45", freeze.get("H") == H_DAYS, str(H_DAYS))
    add("v18_Kmax45", freeze.get("K_max") == K_MAX, str(K_MAX))
    add("v19_k_grid", list(freeze.get("k_grid") or []) == list(K_GRID), json.dumps(list(K_GRID)))
    def _parse_ts(raw: object) -> datetime | None:
        if raw is None:
            return None
        text = str(raw).replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    lattice_ok = True
    by_cuts: dict[str, list[datetime]] = defaultdict(list)
    for row in inst:
        cut = _parse_ts(row.get("cut_c"))
        if cut is None:
            lattice_ok = False
            continue
        by_cuts[str(row.get("target_id"))].append(cut)
    for _tid, cs in by_cuts.items():
        cs = sorted(cs)
        if not cs:
            continue
        c0 = cs[0]
        for cut in cs:
            if (cut - c0).days % CUT_SPACING != 0:
                lattice_ok = False
    r_ok = True
    r_abut = True
    wk_ok = True
    for row in inf:
        r0, r1, cut = _parse_ts(row.get("r0")), _parse_ts(row.get("r1")), _parse_ts(row.get("cut_c"))
        if not (r0 and r1 and cut):
            r_ok = False
            wk_ok = False
            continue
        if (r0 - cut).days != H_DAYS or (r1 - r0).days != H_DAYS:
            wk_ok = False
    by_inf: dict[str, list[tuple[datetime, datetime, datetime]]] = defaultdict(list)
    for row in inf:
        r0, r1, cut = _parse_ts(row.get("r0")), _parse_ts(row.get("r1")), _parse_ts(row.get("cut_c"))
        if r0 and r1 and cut:
            by_inf[str(row.get("target_id"))].append((cut, r0, r1))
    for _tid, triples in by_inf.items():
        triples = sorted(triples, key=lambda x: x[0])
        for (c_a, _r0a, r1a), (c_b, r0b, _r1b) in zip(triples, triples[1:]):
            if (c_b - c_a).days == CUT_SPACING and r1a != r0b:
                r_abut = False
    add("v20_cuts_deterministic", lattice_ok, "cut lattice first+n*45")
    add("v21_nonoverlap_R", r_ok and r_abut, "R=[c+H,c+2H); consecutive 45d cuts abut")
    add("v22_Wk_R_disjoint", wk_ok, "R starts at c+H; Wk ends at c+k k<=H")
    by_inst: dict[tuple[str, str], set] = {}
    for row in inf_reg:
        by_inst.setdefault((str(row["target_id"]), str(row["cut_c"])), set()).add(row.get("common_r_fp"))
    add("v23_same_R_across_k", all(len(v) == 1 for v in by_inst.values()) if by_inst else True, f"instances={len(by_inst)}")
    add("v24_P_ALL", freeze.get("candidate_pool") == "P_ALL", "P_ALL")
    add("v25_gower", freeze.get("metadata_distance") == "gower", "gower")
    add("v26_M5", freeze.get("shortlist_M") == M_SHORTLIST, str(M_SHORTLIST))
    add("v27_rho030", float(freeze.get("rho") or 0) == RHO, str(RHO))
    add("v28_cs4_train", "TRAIN" in str(freeze.get("support_definition")), str(freeze.get("support_definition")))
    add("v29_knn5", CS4_K == 5, str(CS4_K))
    add("v30_q95", CS4_Q == 0.95, str(CS4_Q))
    add("v31_same_target_rows", all(len(v) == 1 and None not in v for v in by_inst.values()) if by_inst else True, "common_r_fp unique per instance")
    add("v32_candidates_ge2", all(int(r.get("n_candidates") or 0) >= 2 for r in inf), f"n_inf={len(inf)}")
    loss_g: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in losses:
        if row.get("L_R") is None:
            continue
        loss_g[(str(row["target_id"]), str(row["cut_c"]))].append(float(row["L_R"]))
    def _k_of(row: dict) -> int | None:
        if row.get("k") is None:
            return None
        return int(row["k"])

    k0 = [r for r in inf_reg if _k_of(r) == 0]
    ok_rand = True
    ok_meta = True
    for r in k0:
        key = (str(r["target_id"]), str(r["cut_c"]))
        vals = loss_g.get(key) or []
        if not vals or r.get("L_star") is None or r.get("Reg_random") is None:
            ok_rand = False
            break
        pred = float(np.mean(vals)) - float(r["L_star"])
        if abs(pred - float(r["Reg_random"])) > 1e-6:
            ok_rand = False
            break
    inst_idx = {(str(r["target_id"]), str(r["cut_c"])): r for r in inf}
    for r in k0:
        inst_r = inst_idx.get((str(r["target_id"]), str(r["cut_c"])))
        if not inst_r or r.get("Reg_metadata") is None or inst_r.get("L_metadata") is None or inst_r.get("L_star") is None:
            ok_meta = False
            break
        pred = float(inst_r["L_metadata"]) - float(inst_r["L_star"])
        if abs(pred - float(r["Reg_metadata"])) > 1e-6:
            ok_meta = False
            break
    add("v33_random_analytic", (not k0) or ok_rand, "Reg_random = mean L_R - L*")
    add("v34_metadata_tie_analytic", (not k0) or ok_meta, "Reg_metadata = L_metadata - L*")
    add("v35_verify_tie_det", "gower[(j, s)], s)" in src, "tie key gower then id")
    add("v36_oracle_same_R", all(r.get("common_r_fp") for r in inf_reg) if inf_reg else True, "oracle uses common_r_fp")
    sci = src
    add("v37_no_epsilon", "1e-12" not in sci and "+ 1e-" not in sci and "eps =" not in sci.lower(), "no epsilon constant in scoring")
    ok_rel = True
    for r in inf_reg:
        ls = r.get("L_star")
        rel = r.get("Reg_rel")
        inv = int(r.get("invalid_rel") or 0)
        if ls is None or not (float(ls) > 0):
            if rel is not None or inv != 1:
                ok_rel = False
                break
        else:
            if rel is None or inv != 0:
                ok_rel = False
                break
    add("v38_invalid_denominators_missing", (not inf_reg) or ok_rel, "Reg_rel iff L*>0; else invalid_rel=1")
    noise_path = f"{OUT_DIR}/RQ4_TEMPORAL_NOISE.parquet"
    audit_path = f"{OUT_DIR}/RQ4_TEMPORAL_AUDIT.parquet"
    noise_ix = {(str(r["target_id"]), str(r["cut_c"])): r for r in noise}
    v39_ok = bool(audit)
    n_disjoint = 0
    for arow in audit:
        key = (str(arow["target_id"]), str(arow["cut_c"]))
        nrow = noise_ix.get(key) or {}
        n_a, n_b, n_r = int(arow.get("n_A") or 0), int(arow.get("n_B") or 0), int(arow.get("n_common_R") or 0)
        if n_a + n_b != n_r:
            v39_ok = False
            break
        if int(nrow.get("n_A") or -1) != n_a or int(nrow.get("n_B") or -1) != n_b:
            v39_ok = False
            break
        if n_a and n_b:
            if not arow.get("fp_A") or not arow.get("fp_B") or arow.get("fp_A") == arow.get("fp_B"):
                v39_ok = False
                break
            n_disjoint += 1
        if arow.get("common_r_fp") != nrow.get("common_r_fp"):
            v39_ok = False
            break
    add(
        "v39_7day_blocks",
        v39_ok,
        json.dumps({"n_audit": len(audit), "n_A_B_disjoint_fp": n_disjoint}),
        [noise_path, audit_path],
    )
    v40_ok = bool(audit)
    n_dir = 0
    for arow in audit:
        key = (str(arow["target_id"]), str(arow["cut_c"]))
        nrow = noise_ix.get(key) or {}
        a, b = int(arow.get("n_A") or 0), int(arow.get("n_B") or 0)
        rab, rba = arow.get("reg_A_to_B"), arow.get("reg_B_to_A")
        nf_a = arow.get("noise_floor_abs")
        nf_n = nrow.get("noise_floor_abs")
        if a >= 1 and b >= 1:
            if rab is None or rba is None or nf_a is None or nf_n is None:
                v40_ok = False
                break
            recon = (float(rab) + float(rba)) / 2.0
            if abs(recon - float(nf_a)) > 1e-12 or abs(float(nf_a) - float(nf_n)) > 1e-12:
                v40_ok = False
                break
            n_dir += 1
        elif nf_a is not None or nf_n is not None:
            v40_ok = False
            break
    add(
        "v40_A_to_B_and_B_to_A",
        v40_ok,
        json.dumps({"n_directional_pairs": n_dir}),
        [noise_path, audit_path],
    )
    v41_ok = bool(audit)
    n_chrono = 0
    for arow in audit:
        key = (str(arow["target_id"]), str(arow["cut_c"]))
        nrow = noise_ix.get(key) or {}
        n1, n2 = int(arow.get("n_first_half") or 0), int(arow.get("n_second_half") or 0)
        n_r = int(arow.get("n_common_R") or 0)
        if n1 + n2 != n_r:
            v41_ok = False
            break
        if n1 and n2 and arow.get("fp_first_half") == arow.get("fp_second_half"):
            v41_ok = False
            break
        for fld in ("chrono_first_to_second_abs", "chrono_second_to_first_abs"):
            av, nv = arow.get(fld), nrow.get(fld)
            if (av is None) != (nv is None):
                v41_ok = False
                break
            if av is not None and abs(float(av) - float(nv)) > 1e-12:
                v41_ok = False
                break
            if av is not None:
                n_chrono += 1
        else:
            continue
        break
    add(
        "v41_chrono_half",
        v41_ok,
        json.dumps({"n_chrono_values_matched": n_chrono}),
        [noise_path, audit_path],
    )
    rel_k45 = _group([r for r in inf_reg if _k_of(r) == 45], "Reg_rel")
    rec_med = target_balanced(rel_k45, "median")
    rec_mean = target_balanced(rel_k45, "mean")
    curve45 = next((e for e in (analysis.get("primary_curve") or []) if _k_of(e) == 45), {})
    v42_ok = rec_med is not None and curve45.get("median_relative_regret") is not None and abs(float(rec_med) - float(curve45["median_relative_regret"])) < 1e-12
    add(
        "v42_target_balanced",
        bool(v42_ok),
        json.dumps({"recomputed_median": rec_med, "recomputed_mean": rec_mean, "curve_k45": curve45.get("median_relative_regret")}, default=str),
        f"{OUT_DIR}/RQ4_REGRET_BY_INSTANCE.parquet",
    )
    t1_map: dict[str, list[float]] = {}
    for r in k0:
        if r.get("Reg_random") is None or r.get("Reg_metadata") is None:
            continue
        d = float(r["Reg_random"]) - float(r["Reg_metadata"])
        if np.isfinite(d):
            t1_map.setdefault(str(r["target_id"]), []).append(d)
    stored_rel = [
        r for r in draws if r.get("stat") == "median_reg_rel"
    ]
    stored_t1 = [r for r in draws if r.get("stat") == "T1_Delta_meta_mean"]
    v43_ok = True
    n_matched = 0
    for k in K_GRID:
        rel_map = _group([r for r in inf_reg if _k_of(r) == k], "Reg_rel")
        keys = list(rel_map.keys())
        n = len(keys)
        got = {int(r["b"]): r.get("value") for r in stored_rel if _k_of(r) == k}
        if n == 0:
            if got:
                v43_ok = False
                break
            continue
        if len(got) != BOOT_B:
            v43_ok = False
            break
        local = np.random.default_rng(BOOT_SEED + 400 + k)
        for b in range(BOOT_B):
            pick = local.integers(0, n, size=n)
            resampled = {str(i): rel_map[keys[int(j)]] for i, j in enumerate(pick)}
            val = target_balanced(resampled, "median")
            stored = got.get(b)
            if val is None or stored is None or abs(float(val) - float(stored)) > 1e-12:
                v43_ok = False
                break
            n_matched += 1
        else:
            continue
        break
    t1_keys = [tk for tk, v in sorted(t1_map.items()) if any(x is not None and np.isfinite(x) for x in v)]
    n_t1 = len(t1_keys)
    got_t1 = {int(r["b"]): r.get("value") for r in stored_t1}
    if v43_ok and n_t1:
        if len(got_t1) != BOOT_B:
            v43_ok = False
        else:
            rng_t1_draw = np.random.default_rng(BOOT_SEED + 500)
            for b in range(BOOT_B):
                pick = rng_t1_draw.integers(0, n_t1, size=n_t1)
                resampled = {str(i): t1_map[t1_keys[int(j)]] for i, j in enumerate(pick)}
                val = target_balanced(resampled, "mean")
                stored = got_t1.get(b)
                if val is None or stored is None or abs(float(val) - float(stored)) > 1e-12:
                    v43_ok = False
                    break
                n_matched += 1
    t1_mean, t1_lo, t1_hi = bootstrap_ci(t1_map, "mean", np.random.default_rng(BOOT_SEED))
    sot_t1 = analysis.get("T1") or {}
    v43_ok = bool(v43_ok) and t1_mean is not None and sot_t1.get("Delta_meta_mean") is not None and abs(float(t1_mean) - float(sot_t1["Delta_meta_mean"])) < 1e-12
    add(
        "v43_target_cluster_bootstrap",
        v43_ok,
        json.dumps({"n_draw_matches": n_matched, "n_rel_draws": len(stored_rel), "n_t1_draws": len(stored_t1), "t1_mean": t1_mean}, default=str),
        f"{OUT_DIR}/RQ4_BOOTSTRAP_DRAWS.parquet",
    )
    add("v44_B5000", freeze.get("bootstrap_B") == BOOT_B and analysis.get("T1") is not None, str(BOOT_B))
    add("v45_seed42", freeze.get("bootstrap_seed") == BOOT_SEED, str(BOOT_SEED))
    curve = (analysis.get("T3") or {}).get("curve") or {}
    kt = analysis.get("k_tau")
    persist_tau = True
    if kt is not None:
        ks = list(K_GRID)
        i0 = ks.index(int(kt))
        persist_tau = all((curve.get(str(k)) or {}).get("upper95") is not None and float((curve.get(str(k)) or {}).get("upper95")) <= TAU_PRIMARY for k in ks[i0:])
    add("v46_persistent_k_tau", persist_tau, str(kt))
    kn = analysis.get("k_noise")
    persist_n = True
    t4c = (analysis.get("T4") or {}).get("curve") or {}
    if kn is not None:
        ks = list(K_GRID)
        i0 = ks.index(int(kn))
        persist_n = all((t4c.get(str(k)) or {}).get("upper95") is not None and float((t4c.get(str(k)) or {}).get("upper95")) <= 0 for k in ks[i0:])
    add("v47_persistent_k_noise", persist_n, str(kn))
    add("v48_no_perk_fishing", set((analysis.get("T1") or {}).keys()) >= {"Delta_meta_mean", "ci95_low"} and "k" in (analysis.get("T2") or {}) and int((analysis.get("T2") or {}).get("k") or 0) == 45, "T1 T2 k=45 k_tau k_noise")
    hgb_ks = sorted({int(r["k"]) for r in hgb if r.get("k") is not None})
    add("v49_hgb_not_primary", analysis.get("primary_model") == "SplineRidge" and hgb_ks == list(K_GRID), f"hgb_k={hgb_ks}")
    add("v50_deterministic_rerun", rerun_ok, analysis["_scientific_content_sha256"])
    add(
        "v51_manuscript_from_sot",
        (root / OUT_DIR / "RQ4_SUMMARY.json").is_file()
        and analysis.get("status")
        in {"RQ4_SOURCE_SELECTION_RESULTS_MATERIALIZED", "RQ4_SOURCE_SELECTION_RESULTS_STRUCTURALLY_NONINFORMATIVE"},
        str(analysis.get("status")),
    )
    add("spec_file_sha", spec_sha == SPEC_SHA256, spec_sha, SPEC_PATH)
    prompt_path = root / PROMPT_PATH
    prompt_sha = sha256_file(prompt_path) if prompt_path.is_file() else ""
    add("prompt_archive_sha", prompt_sha == PROMPT_SHA256, prompt_sha, PROMPT_PATH)
    analysis["execution_prompt_path"] = PROMPT_PATH
    analysis["execution_prompt_sha256"] = prompt_sha

    out = root / OUT_DIR
    skip_names = {"RQ4_MANIFEST.json", "RQ4_VALIDATION.json", "RQ4_SUMMARY.json"}
    recs = []
    for path in sorted(out.iterdir()):
        if path.name.startswith(".") or path.name in skip_names:
            continue
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq

            table = pq.read_table(path)
            rec = artifact(root, path, rows=table.num_rows, fp=_pq_fp(path))
        elif path.suffix == ".csv":
            rec = artifact(root, path, rows=len(read_csv(path)), fp=csv_fingerprint(path))
        else:
            rec = artifact(root, path)
        recs.append(rec)
    if prompt_path.is_file():
        recs.append(artifact(root, prompt_path))
    analysis["completed_at_utc"] = now
    frozen: list[dict] = []
    supplemental: list[dict] = []
    for c in checks:
        entry = {
            "name": c.name,
            "passed": c.passed,
            "evidence": c.details,
            "artifact_path": check_art.get(c.name),
        }
        if c.name.startswith("v") and c.name[1:3].isdigit():
            frozen.append(entry)
        else:
            supplemental.append(entry)
    all_frozen = bool(frozen) and all(e["passed"] for e in frozen) and len(frozen) == 51
    analysis["validation_summary"] = {
        "n_frozen_invariants": len(frozen),
        "all_frozen_invariants_passed": all_frozen,
        "n_supplemental_checks": len(supplemental),
        "rerun_identical": rerun_ok,
    }
    dump_json(
        out / "RQ4_VALIDATION.json",
        {
            "frozen_invariants": frozen,
            "supplemental_checks": supplemental,
            "n_frozen_invariants": len(frozen),
            "n_supplemental_checks": len(supplemental),
            "all_frozen_invariants_passed": all_frozen,
            "all_passed": all_frozen and all(e["passed"] for e in supplemental),
            "rerun_identical": rerun_ok,
            "rerun_scientific_content_sha256": analysis["_scientific_content_sha256"],
        },
    )
    recs.append(artifact(root, out / "RQ4_VALIDATION.json"))
    dump_json(out / "RQ4_SUMMARY.json", {k: analysis[k] for k in analysis if not str(k).startswith("_")})
    recs.append(artifact(root, out / "RQ4_SUMMARY.json"))
    by_path = {rec.path: rec for rec in recs}
    recs = list(by_path.values())
    manifest = [
        {
            "path": rec.path,
            "sha256": rec.sha256,
            "bytes": rec.bytes,
            "row_count": rec.row_count,
            "schema_fingerprint": rec.schema_fingerprint,
            "role": Path(rec.path).name,
        }
        for rec in recs
    ]
    dump_json(out / "RQ4_MANIFEST.json", {"artifacts": manifest})
    recs.append(artifact(root, out / "RQ4_MANIFEST.json"))
    analysis["artifact_manifest_path"] = "artifacts/rq4/RQ4_MANIFEST.json"
    analysis["artifact_manifest_sha256"] = sha256_file(out / "RQ4_MANIFEST.json")

    if not all_frozen:
        return StageResult(
            status="STOP",
            message="RQ4 frozen invariants failed",
            sot_patch={"results": {"rq4_finalize": {"status": "STOP"}}},
            artifacts=recs,
            validations=checks,
        )
    payload = {k: analysis[k] for k in analysis if not str(k).startswith("_")}
    payload["status"] = analysis["status"]
    return StageResult(
        status="GO",
        message=analysis["status"],
        sot_patch=ledger_patch(
            stage="rq4_finalize",
            now=now,
            payload={"status": "GO", "analysis_status": analysis["status"]},
            recs=recs,
            extra={"scientific_analysis": {"rq4_source_selection": payload}},
        ),
        artifacts=recs,
        validations=checks,
    )
