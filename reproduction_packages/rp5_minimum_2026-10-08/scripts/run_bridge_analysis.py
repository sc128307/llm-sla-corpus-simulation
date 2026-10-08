"""Run the local exploratory feature-to-regional-transfer bridge.

The script consumes only the passed U1 audit JSON. It does not regenerate
texts, recompute features, train classifiers, or overwrite frozen inputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.formula.api as smf


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "results" / "bridge_input_audit.json"
JOINED = ROOT / "results" / "bridge_feature_transfer.csv"
LEAVEOUT = ROOT / "results" / "bridge_feature_transfer_model_leaveout.csv"
PERMUTATION = ROOT / "results" / "bridge_feature_transfer_permutation.csv"
BOOTSTRAP = ROOT / "results" / "bridge_feature_transfer_model_bootstrap.csv"
SUMMARY = ROOT / "results" / "bridge_feature_transfer_summary.json"
SEED = 42
PERMUTATIONS = 10_000
BOOTSTRAPS = 5_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def fit_fixed_effects(frame: pd.DataFrame, outcome: str) -> dict[str, float | int | str]:
    if frame["model"].nunique() < 2 or frame["condition"].nunique() < 2:
        return {"status": "insufficient_groups", "n": int(len(frame))}
    fit = smf.ols(f"{outcome} ~ mean_abs_d_z + C(model) + C(condition)", data=frame).fit()
    ci_low, ci_high = fit.conf_int().loc["mean_abs_d_z"]
    return {
        "status": "ok",
        "n": int(len(frame)),
        "n_models": int(frame["model"].nunique()),
        "n_conditions": int(frame["condition"].nunique()),
        "slope": float(fit.params["mean_abs_d_z"]),
        "se": float(fit.bse["mean_abs_d_z"]),
        "t": float(fit.tvalues["mean_abs_d_z"]),
        "p_value": float(fit.pvalues["mean_abs_d_z"]),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "df_resid": float(fit.df_resid),
        "r_squared": float(fit.rsquared),
    }


def nuisance_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Intercept plus model and condition fixed effects, with reference levels."""
    model = pd.get_dummies(frame["model"], drop_first=True, dtype=float)
    condition = pd.get_dummies(frame["condition"], drop_first=True, dtype=float)
    return np.column_stack([np.ones(len(frame)), model.to_numpy(), condition.to_numpy()])


def fast_fixed_effects_slope(frame: pd.DataFrame, outcome: str) -> float:
    """Compute the same fixed-effect slope by residualized linear algebra."""
    if frame["model"].nunique() < 2 or frame["condition"].nunique() < 2:
        return float("nan")
    z = nuisance_matrix(frame)
    x = frame["mean_abs_d_z"].to_numpy(float)
    y = frame[outcome].to_numpy(float)
    z_pinv = np.linalg.pinv(z)
    x_resid = x - z @ (z_pinv @ x)
    y_resid = y - z @ (z_pinv @ y)
    denominator = float(x_resid @ x_resid)
    return float((x_resid @ y_resid) / denominator) if denominator > 0 else float("nan")


def correlation(frame: pd.DataFrame, outcome: str) -> dict[str, float | int]:
    x = frame["mean_abs_d_z"].to_numpy(float)
    y = frame[outcome].to_numpy(float)
    pearson = stats.pearsonr(x, y)
    spearman = stats.spearmanr(x, y)
    return {
        "n": int(len(frame)),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise RuntimeError("U1 bridge input audit is not passed")
    frame = pd.DataFrame(audit["joined_cells"])
    required = {"model", "condition", "mean_abs_d_z", "accuracy", "macro_f1"}
    if set(required) - set(frame.columns):
        raise RuntimeError(f"Missing bridge columns: {sorted(set(required) - set(frame.columns))}")
    if len(frame) != 24 or frame.duplicated(["model", "condition"]).any():
        raise RuntimeError("Bridge input must contain 24 unique model-condition cells")
    frame["mean_abs_d_z"] = frame["mean_abs_d_z"].astype(float)
    frame["accuracy"] = frame["accuracy"].astype(float)
    frame["macro_f1"] = frame["macro_f1"].astype(float)
    frame = frame.sort_values(["model", "condition"], ignore_index=True)
    frame.to_csv(JOINED, index=False, float_format="%.12g")

    observed = {
        "macro_f1": fit_fixed_effects(frame, "macro_f1"),
        "accuracy": fit_fixed_effects(frame, "accuracy"),
    }
    correlations = {
        "macro_f1": correlation(frame, "macro_f1"),
        "accuracy": correlation(frame, "accuracy"),
    }

    leave_rows = []
    for model in sorted(frame["model"].unique()):
        subset = frame.loc[frame["model"] != model].copy()
        for outcome in ("macro_f1", "accuracy"):
            result = fit_fixed_effects(subset, outcome)
            leave_rows.append({"omitted_model": model, "outcome": outcome, **result})
    leave_frame = pd.DataFrame(leave_rows)
    leave_frame.to_csv(LEAVEOUT, index=False, float_format="%.12g")

    rng = np.random.default_rng(SEED)
    permutation_rows = []
    observed_slope = observed["macro_f1"]["slope"]
    models = sorted(frame["model"].unique())
    for iteration in range(PERMUTATIONS):
        permuted = frame.copy()
        for model in models:
            mask = permuted["model"] == model
            permuted.loc[mask, "macro_f1"] = rng.permutation(permuted.loc[mask, "macro_f1"].to_numpy())
        permutation_rows.append({"iteration": iteration, "slope": fast_fixed_effects_slope(permuted, "macro_f1")})
    permutation_frame = pd.DataFrame(permutation_rows)
    permutation_frame["abs_at_least_observed"] = (
        permutation_frame["slope"].abs() >= abs(observed_slope)
    )
    permutation_frame.to_csv(PERMUTATION, index=False, float_format="%.12g")
    permutation_p = float(
        (1 + permutation_frame["abs_at_least_observed"].sum()) / (1 + len(permutation_frame))
    )

    bootstrap_rows = []
    for iteration in range(BOOTSTRAPS):
        sampled_models = rng.choice(models, size=len(models), replace=True)
        pieces = []
        for block_id, model in enumerate(sampled_models):
            block = frame.loc[frame["model"] == model].copy()
            block["bootstrap_block"] = block_id
            pieces.append(block)
        boot = pd.concat(pieces, ignore_index=True)
        bootstrap_rows.append({"iteration": iteration, "slope": fast_fixed_effects_slope(boot, "macro_f1"), "n_models": int(boot["model"].nunique())})
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    bootstrap_frame.to_csv(BOOTSTRAP, index=False, float_format="%.12g")
    valid_boot = bootstrap_frame["slope"].dropna()
    boot_ci = [float(x) for x in valid_boot.quantile([0.025, 0.975]).tolist()]

    summary = {
        "status": "complete",
        "analysis": "exploratory_feature_to_regional_transfer_bridge",
        "estimand": "association between mean absolute 22-feature d_z and Primary held-out regional performance",
        "scope": "24 l2_primary model-by-condition cells; raw Primary classifier outcomes; fixed 1040-essay human test",
        "input_audit": str(AUDIT.relative_to(ROOT)).replace("\\", "/"),
        "input_audit_sha256": sha256(AUDIT),
        "script": str(Path(__file__).relative_to(ROOT)).replace("\\", "/"),
        "script_sha256": sha256(Path(__file__)),
        "seed": SEED,
        "permutations": PERMUTATIONS,
        "model_bootstraps": BOOTSTRAPS,
        "predictor": "mean_abs_d_z across 22 frozen features; lower means closer feature profile",
        "outcome_primary": "macro_f1",
        "outcome_secondary": "accuracy",
        "fixed_effects_formula": "outcome ~ mean_abs_d_z + C(model) + C(condition)",
        "observed_fixed_effects": observed,
        "raw_correlations": correlations,
        "permutation": {
            "within_block": "permute macro_f1 within each model across its four conditions",
            "two_sided_add_one_p": permutation_p,
            "observed_slope": observed_slope,
        },
        "model_bootstrap_macro_f1": {
            "valid_replicates": int(len(valid_boot)),
            "percentile_ci95": boot_ci,
            "note": "model-level sensitivity only; six model blocks and duplicate sampled blocks are retained",
        },
        "outputs": {
            "joined_cells": str(JOINED.relative_to(ROOT)).replace("\\", "/"),
            "leave_one_model_out": str(LEAVEOUT.relative_to(ROOT)).replace("\\", "/"),
            "permutation": str(PERMUTATION.relative_to(ROOT)).replace("\\", "/"),
            "model_bootstrap": str(BOOTSTRAP.relative_to(ROOT)).replace("\\", "/"),
        },
        "interpretation_boundary": "descriptive/exploratory association; not causal, not an aggregate fidelity score, and not evidence about model internals or L1 transfer",
    }
    summary["output_sha256"] = {
        "joined_cells": sha256(JOINED),
        "leave_one_model_out": sha256(LEAVEOUT),
        "permutation": sha256(PERMUTATION),
        "model_bootstrap": sha256(BOOTSTRAP),
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": summary["status"], "summary": str(SUMMARY), "permutation_p": permutation_p, "boot_ci": boot_ci}, ensure_ascii=False))


if __name__ == "__main__":
    main()
