"""Turns a per-model score table into the numbers the paper reports."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from .config import Config
from .methods import BASELINES, METHODS, calls_per_question
from .metrics import (aurac, auroc, bootstrap_auroc, bootstrap_auroc_difference, delong_test, holm,
                      impute)
from .scoring.combine import logistic_out_of_fold, rank_mean

GROUPS = ("CW", "CR", "INC", "EXCL")
MIN_PER_CLASS = 5


def matched_frame(table: pd.DataFrame) -> pd.DataFrame:
    frame = table[table["matched"] & table["group"].isin(["CW", "CR"])].copy()
    frame["label"] = (frame["group"] == "CW").astype(int)
    if frame["cross_family"].notna().any():
        cps, _ = impute(frame["cps"].to_numpy())
        cross, _ = impute(frame["cross_family"].to_numpy())
        frame["cps_cross"] = rank_mean(cps, cross)
    else:
        frame["cps_cross"] = np.nan
    return frame


def counts(table: pd.DataFrame) -> dict:
    out = {"total": int(len(table))}
    out.update({g: int((table["group"] == g).sum()) for g in GROUPS})
    out["strict_CW"] = int((table["strict_group"] == "CW").sum())
    out["strict_CR"] = int((table["strict_group"] == "CR").sum())
    out["per_dataset"] = {
        dataset: {g: int((part["group"] == g).sum()) for g in GROUPS} | {"total": int(len(part))}
        for dataset, part in table.groupby("dataset")
    }
    return out


def method_results(config: Config, table: pd.DataFrame, frame: pd.DataFrame) -> dict:
    calls = calls_per_question(config)
    consistent = table[table["group"].isin(["CW", "CR", "INC"])]
    out = {}
    for method in METHODS:
        if frame[method.key].isna().all():
            continue
        scores, missing = impute(frame[method.key].to_numpy())
        interval = bootstrap_auroc(frame["label"].to_numpy(), scores, config.bootstrap_samples, config.seed)
        entry = {"label": method.label, "access": method.access, "is_ours": method.is_ours,
                 "calls": calls[method.key], "auroc_matched": asdict(interval), "n_missing": missing,
                 "auroc_all": None, "aurac_all": None}
        if method.scope == "all" and consistent[method.key].notna().any():
            all_scores, _ = impute(consistent[method.key].to_numpy())
            wrong = (~consistent["correct"].astype(bool)).astype(int).to_numpy()
            entry["auroc_all"] = auroc(wrong, all_scores)
            entry["aurac_all"] = aurac(consistent["correct"].astype(bool).to_numpy(), all_scores)
        out[method.key] = entry
    return out


def best_baseline(methods: dict, access: str | None = "text") -> str | None:
    candidates = [m.key for m in BASELINES if m.key in methods and (access is None or m.access == access)]
    if not candidates:
        return None
    return max(candidates, key=lambda k: methods[k]["auroc_matched"]["value"])


def comparisons(config: Config, frame: pd.DataFrame, methods: dict, ours: str = "cps") -> dict:
    labels = frame["label"].to_numpy()
    ours_scores, _ = impute(frame[ours].to_numpy())
    rows, p_values = {}, []
    for method in BASELINES:
        if method.key not in methods:
            continue
        other, _ = impute(frame[method.key].to_numpy())
        _, _, z, p = delong_test(labels, ours_scores, other)
        diff = bootstrap_auroc_difference(labels, ours_scores, other, config.bootstrap_samples, config.seed)
        rows[method.key] = {"z": z, "p": p, "difference": asdict(diff)}
        p_values.append(p)
    for key, adjusted in zip(rows, holm(p_values)):
        rows[key]["p_holm"] = adjusted
    return rows


def flip_rates(config: Config, frame: pd.DataFrame) -> dict:
    out = {}
    rng = np.random.default_rng(config.seed)
    for group in ("CW", "CR"):
        part = frame[frame["group"] == group]
        out[group] = {}
        for column in ("f_counter", "f_doubt", "cps"):
            values = part[column].dropna().to_numpy()
            if len(values) == 0:
                continue
            boots = [rng.choice(values, len(values)).mean() for _ in range(config.bootstrap_samples)]
            low, high = np.percentile(boots, [2.5, 97.5])
            out[group][column] = {"value": float(values.mean()), "low": float(low), "high": float(high),
                                  "n": int(len(values))}
    return out


def per_dataset(frame: pd.DataFrame, methods: dict, baseline_key: str | None) -> dict:
    out = {}
    for dataset, part in frame.groupby("dataset"):
        n_cw, n_cr = int(part["label"].sum()), int((1 - part["label"]).sum())
        entry = {"n_cw": n_cw, "n_cr": n_cr, "cps": None, "baseline": None}
        if n_cw >= MIN_PER_CLASS and n_cr >= MIN_PER_CLASS:
            entry["cps"] = auroc(part["label"].to_numpy(), impute(part["cps"].to_numpy())[0])
            if baseline_key:
                entry["baseline"] = auroc(part["label"].to_numpy(), impute(part[baseline_key].to_numpy())[0])
        out[dataset] = entry
    return out


def ablations(config: Config, table: pd.DataFrame, frame: pd.DataFrame, baseline_key: str | None) -> dict:
    labels = frame["label"].to_numpy()

    def score(column: str, part: pd.DataFrame | None = None) -> float | None:
        part = frame if part is None else part
        if column not in part or part[column].isna().all() or part["label"].nunique() < 2:
            return None
        return auroc(part["label"].to_numpy(), impute(part[column].to_numpy())[0])

    out: dict = {"pressure_count": {}, "signal": {}, "strict": {}, "random_alternative": {}, "combination": {}}
    for k, cps_col, counter_col in ((1, "cps_k1", "f_counter_k1"), (2, "cps_k2", "f_counter_k2"),
                                    (3, "cps", "f_counter")):
        out["pressure_count"][str(k)] = {"cps": score(cps_col), "f_counter": score(counter_col),
                                         "calls": 2 + 2 * k}
    out["signal"] = {"cps": score("cps"), "f_counter": score("f_counter"), "f_doubt": score("f_doubt")}
    if score("cps") is not None and score("f_counter") is not None:
        diff = bootstrap_auroc_difference(labels, impute(frame["cps"].to_numpy())[0],
                                          impute(frame["f_counter"].to_numpy())[0],
                                          config.bootstrap_samples, config.seed)
        out["signal"]["cps_minus_f_counter"] = asdict(diff)

    strict = frame[frame["strict_group"].isin(["CW", "CR"])]
    out["strict"] = {"n_cw": int((strict["strict_group"] == "CW").sum()),
                     "n_cr": int((strict["strict_group"] == "CR").sum()),
                     "cps": score("cps", strict),
                     "baseline": score(baseline_key, strict) if baseline_key else None}

    with_random = frame[frame["cps_random"].notna()]
    if len(with_random) and with_random["label"].nunique() == 2:
        out["random_alternative"] = {"n": int(len(with_random)), "cps_self": score("cps", with_random),
                                     "cps_random": score("cps_random", with_random)}

    plausible = frame[frame["alternative_equivalent"] == 0]
    out["alternative_equivalent_count"] = int((frame["alternative_equivalent"] == 1).sum())
    out["excluding_equivalent_alternatives"] = score("cps", plausible)

    out["combination"] = {"cps": score("cps"), "cross_family": score("cross_family"),
                          "rank_mean": score("cps_cross")}
    features = frame[["f_counter", "f_doubt", "cross_family"]].astype(float)
    features = features.loc[:, features.notna().any()]
    if not features.empty:
        filled = np.column_stack([impute(features[c].to_numpy())[0] for c in features.columns])
        oof = logistic_out_of_fold(filled, labels, config.cv_folds, config.seed)
        if not np.isnan(oof).all():
            out["combination"]["logistic_cv"] = auroc(labels, oof)
    return out


def hypotheses(methods: dict, comparison: dict, ablation: dict, baseline_key: str | None) -> dict:
    """Pre-registered criteria from the proposal, evaluated mechanically."""
    def includes_half(key: str) -> bool | None:
        if key not in methods:
            return None
        interval = methods[key]["auroc_matched"]
        return interval["low"] <= 0.5 <= interval["high"]

    out = {"H1": {"semantic_entropy_ci_includes_half": includes_half("semantic_entropy"),
                  "selfcheck_ci_includes_half": includes_half("selfcheck_nli")}}
    h1_values = [v for v in out["H1"].values() if v is not None]
    out["H1"]["supported"] = bool(h1_values) and all(h1_values)

    cps = methods.get("cps", {}).get("auroc_matched", {}).get("value")
    if cps is not None and baseline_key:
        base = methods[baseline_key]["auroc_matched"]["value"]
        p_holm = comparison.get(baseline_key, {}).get("p_holm")
        out["H2"] = {"cps": cps, "best_text_baseline": baseline_key, "baseline_auroc": base, "p_holm": p_holm,
                     "supported": bool(cps >= 0.65 and cps > base and p_holm is not None and p_holm < 0.05)}
    signal = ablation.get("signal", {})
    if signal.get("cps") is not None and signal.get("f_counter") is not None:
        out["H3"] = {"cps": signal["cps"], "f_counter": signal["f_counter"],
                     "supported": signal["cps"] > signal["f_counter"]}
    combo = ablation.get("combination", {})
    if combo.get("rank_mean") is not None and combo.get("cross_family") is not None:
        out["H4"] = {**combo, "supported": combo["rank_mean"] > max(combo["cps"], combo["cross_family"])}
    return out
