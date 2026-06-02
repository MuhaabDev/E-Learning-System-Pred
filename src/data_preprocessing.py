"""
Preprocessing mirrored from notebooks (Milestone 1 regression, Milestone 2 classification).
Regression and classification pipelines differ — do not merge them.
"""
from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from src.config import DATA_CLASSIFICATION, DATA_REGRESSION


def load_tables(kind: str) -> Dict[str, pd.DataFrame]:
    base = DATA_REGRESSION if kind == "regression" else DATA_CLASSIFICATION

    assessments = pd.read_csv(base / "assessments.csv")
    courses = pd.read_csv(base / "courses.csv")

    student_assessment = pd.read_csv(
        base / "StudentAssesments.csv",
        engine="python",
        dtype={"id_assessment": np.int32, "id_student": np.int64, "date_submitted": np.int64, "is_banked": np.int8},
    )
    student_info = pd.read_csv(base / "studentInfo.csv")
    student_registration = pd.read_csv(base / "studentRegistration.csv")

    sv_dtypes = {
        "code_module": "object",
        "code_presentation": "object",
        "id_student": np.int32,
        "id_site": np.int32,
        "date": np.int32,
        "sum_click": np.int32,
    }
    student_vle_chunks: List[pd.DataFrame] = []
    for chunk in pd.read_csv(
        base / "studentVle.csv",
        dtype=sv_dtypes,
        chunksize=30_000,
        engine="c",
        on_bad_lines="warn",
    ):
        student_vle_chunks.append(chunk)
    student_vle = pd.concat(student_vle_chunks, ignore_index=True)
    for col in ["code_module", "code_presentation"]:
        student_vle[col] = student_vle[col].astype("category")

    vle = pd.read_csv(base / "vle.csv", dtype={"id_site": np.int32})
    vle[["code_module", "code_presentation", "activity_type"]] = vle[
        ["code_module", "code_presentation", "activity_type"]
    ].astype("category")
    return {
        "assessments": assessments,
        "courses": courses,
        "student_assessment": student_assessment,
        "student_info": student_info,
        "student_registration": student_registration,
        "student_vle": student_vle,
        "vle": vle,
    }


def prepare_vle_assessments(vle: pd.DataFrame, assessments: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    vle = vle.copy()
    assessments = assessments.copy()
    vle.replace("?", np.nan, inplace=True)
    vle["week_from"] = pd.to_numeric(vle["week_from"], errors="coerce")
    vle["week_to"] = pd.to_numeric(vle["week_to"], errors="coerce")
    vle["week_from"] = vle["week_from"].fillna(vle["week_from"].median())
    vle["week_to"] = vle["week_to"].fillna(vle["week_to"].median())

    assessments.replace("?", np.nan, inplace=True)
    assessments["date"] = pd.to_numeric(assessments["date"], errors="coerce")
    assessments["date"] = assessments["date"].fillna(assessments["date"].median())
    return vle, assessments


def regression_vle_aggregate(student_vle: pd.DataFrame, vle: pd.DataFrame) -> pd.DataFrame:
    student_vle_full = student_vle.merge(vle, on=["id_site", "code_module", "code_presentation"])
    agg = (
        student_vle_full.groupby(["code_module", "code_presentation", "id_student"], observed=False)
        .agg(
            total_clicks=("sum_click", "sum"),
            avg_clicks_per_day=("sum_click", "mean"),
            interaction_days=("date", "nunique"),
            total_interactions=("id_site", "count"),
            unique_activity_types=("activity_type", "nunique"),
            max_single_day_clicks=("sum_click", "max"),
            avg_week_from=("week_from", "mean"),
            week_span=("week_to", lambda x: x.max() - x.min()),
        )
        .reset_index()
    )
    return agg


def classification_vle_aggregate(student_vle: pd.DataFrame, vle: pd.DataFrame) -> pd.DataFrame:
    student_vle_full = student_vle.merge(vle, on=["id_site", "code_module", "code_presentation"])
    return (
        student_vle_full.groupby(["code_module", "code_presentation", "id_student"], observed=False)
        .agg(
            total_clicks=("sum_click", "sum"),
            avg_clicks=("sum_click", "mean"),
            interaction_days=("date", "nunique"),
            total_interactions=("id_site", "count"),
            unique_activity_types=("activity_type", "nunique"),
        )
        .reset_index()
    )


def merge_regression_master(tab: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    vle, assessments = prepare_vle_assessments(tab["vle"], tab["assessments"])
    student_vle_agg = regression_vle_aggregate(tab["student_vle"], vle)
    df_raw = (
        tab["student_assessment"]
        .merge(assessments, on="id_assessment")
        .merge(tab["student_info"], on=["id_student", "code_module", "code_presentation"])
        .merge(tab["courses"], on=["code_module", "code_presentation"])
        .merge(tab["student_registration"], on=["id_student", "code_module", "code_presentation"])
        .merge(student_vle_agg, on=["id_student", "code_module", "code_presentation"])
    )
    if "final_result" in df_raw.columns:
        df_raw = df_raw.drop(columns=["final_result"])
    return df_raw


def merge_classification_master(tab: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    vle, assessments = prepare_vle_assessments(tab["vle"], tab["assessments"])
    student_vle_agg = classification_vle_aggregate(tab["student_vle"], vle)
    df = tab["student_assessment"].merge(assessments, on="id_assessment", how="left")
    df = df.merge(tab["student_info"], on=["id_student", "code_module", "code_presentation"], how="left")
    df = df.merge(tab["courses"], on=["code_module", "code_presentation"], how="left")
    df = df.merge(tab["student_registration"], on=["id_student", "code_module", "code_presentation"], how="left")
    df = df.merge(student_vle_agg, on=["id_student", "code_module", "code_presentation"], how="left")
    return df


def clean_regression_before_encoding(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    df.replace("?", np.nan, inplace=True)
    for col in [
        "score",
        "date_submitted",
        "date_registration",
        "date_unregistration",
        "num_of_prev_attempts",
        "studied_credits",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["score"])
    df["date_submitted"] = df["date_submitted"].fillna(df["date_submitted"].median())
    df["date_registration"] = df["date_registration"].fillna(df["date_registration"].median())
    df["date_unregistration"] = df["date_unregistration"].fillna(-1)
    df["imd_band"] = df["imd_band"].fillna(df["imd_band"].mode()[0])

    if "week_span" in df.columns:
        df["week_span"] = df["week_span"].fillna(0)
    if "avg_week_from" in df.columns:
        df["avg_week_from"] = df["avg_week_from"].fillna(df["avg_week_from"].median())

    if df.duplicated().any():
        df = df.drop_duplicates()
    return df


def fit_transform_regression_categorical(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    ordinal_col_names: List[str] = []
    ordinal_specs: List[List[Any]] = []

    if "imd_band" in df.columns:
        imd_vals = sorted(df["imd_band"].dropna().unique().tolist())
        ordinal_col_names.append("imd_band")
        ordinal_specs.append(imd_vals)

    age_order = ["0-35", "35-55", "55<="]
    education_order = [
        "No Formal quals",
        "Lower Than A Level",
        "A Level or Equivalent",
        "HE Qualification",
        "Post Graduate Qualification",
    ]

    if "age_band" in df.columns:
        age_vals = [v for v in age_order if v in set(df["age_band"].unique())]
        ordinal_col_names.append("age_band")
        ordinal_specs.append(age_vals)

    if "highest_education" in df.columns:
        edu_vals = [v for v in education_order if v in set(df["highest_education"].unique())]
        ordinal_col_names.append("highest_education")
        ordinal_specs.append(edu_vals)

    nominal_cols_full = ["code_module", "code_presentation", "assessment_type", "gender", "region", "disability"]
    nominal_cols = [c for c in nominal_cols_full if c in df.columns]

    df_out = df.copy()
    ord_enc: Optional[OrdinalEncoder] = None
    if ordinal_col_names:
        ord_enc = OrdinalEncoder(
            categories=ordinal_specs,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        df_out[ordinal_col_names] = ord_enc.fit_transform(df_out[ordinal_col_names])

    nom_enc: Optional[OrdinalEncoder] = None
    if nominal_cols:
        nom_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        df_out[nominal_cols] = nom_enc.fit_transform(df_out[nominal_cols])

    artifacts = {
        "ordinal_encoder": ord_enc,
        "ordinal_columns": ordinal_col_names,
        "nominal_encoder": nom_enc,
        "nominal_columns": nominal_cols,
    }
    return df_out, artifacts


def transform_regression_categorical(df: pd.DataFrame, artifacts: Mapping[str, Any]) -> pd.DataFrame:
    df_out = df.copy()
    ord_enc = artifacts.get("ordinal_encoder")
    ord_cols: List[str] = list(artifacts.get("ordinal_columns", []))
    nom_enc = artifacts.get("nominal_encoder")
    nom_cols: List[str] = list(artifacts.get("nominal_columns", []))
    if ord_cols and ord_enc is not None:
        df_out[ord_cols] = ord_enc.transform(df_out[ord_cols])
    if nom_cols and nom_enc is not None:
        df_out[nom_cols] = nom_enc.transform(df_out[nom_cols])
    return df_out


def add_regression_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["submission_timing"] = df["date_submitted"] - df["date"]
    for col in ["total_clicks", "avg_clicks_per_day", "total_interactions", "max_single_day_clicks"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])

    log_engagement_cols = [c for c in df.columns if c.startswith("log_")]
    if log_engagement_cols:
        df["engagement_score"] = df[log_engagement_cols].mean(axis=1)
    if "engagement_score" in df.columns and "weight" in df.columns:
        df["weighted_engagement"] = df["weight"] * df["engagement_score"]

    df["registration_earliness"] = -df["date_registration"]
    if "total_clicks" in df.columns and "interaction_days" in df.columns:
        df["clicks_per_day"] = df["total_clicks"] / (df["interaction_days"] + 1)
        df["log_clicks_per_day"] = np.log1p(df["clicks_per_day"])

    if "studied_credits" in df.columns and "num_of_prev_attempts" in df.columns:
        df["credit_x_attempts"] = df["studied_credits"] * (df["num_of_prev_attempts"] + 1)
    return df


def drop_regression_noise_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_drop = [
        "id_assessment",
        "id_student",
        "total_clicks",
        "avg_clicks_per_day",
        "total_interactions",
        "max_single_day_clicks",
        "clicks_per_day",
        "date_submitted",
        "date",
        "date_registration",
        "is_banked",
        "gender",
        "region",
    ]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    return df.drop(columns=cols_to_drop)


def build_regression_dataset(tab: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    df_raw = merge_regression_master(tab)
    cleaned = clean_regression_before_encoding(df_raw)
    encoded, encoders = fit_transform_regression_categorical(cleaned)
    feat = add_regression_engineered_features(encoded)
    final_df = drop_regression_noise_columns(feat)
    X = final_df.drop(columns=["score"])
    y = final_df["score"]
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(lambda col: col.fillna(col.median(skipna=True)).fillna(0.0))
    return X, y, encoders


def classification_preprocess_before_learning(df_merge: pd.DataFrame) -> pd.DataFrame:
    """Through VLE/imputation; still has ids and textual target."""
    df = df_merge.copy()
    df.replace("?", np.nan, inplace=True)
    df["has_unregistered"] = df["date_unregistration"].notna().astype(int)
    df.drop(columns=["id_assessment", "id_student", "date_unregistration"], inplace=True)

    df["date"] = pd.to_numeric(df["date"], errors="coerce")
    df["date_registration"] = pd.to_numeric(df["date_registration"], errors="coerce")
    df["date_submitted"] = pd.to_numeric(df["date_submitted"], errors="coerce")
    df["date"] = df["date"].fillna(df["date"].median())
    df["date_registration"] = df["date_registration"].fillna(df["date_registration"].median())

    df["imd_band"] = df["imd_band"].fillna("Unknown")

    vle_cols = ["total_clicks", "avg_clicks", "interaction_days", "total_interactions", "unique_activity_types"]
    df[vle_cols] = df[vle_cols].fillna(0)
    return df


def label_encode_then_dummies(
    df: pd.DataFrame, target_encoder: LabelEncoder, fit: bool
) -> Tuple[pd.DataFrame, List[str]]:
    """Notebook order: LE on target → get_dummies on remaining objects → caller handles bool/int."""
    y_col = df["assessmentClass"]
    if fit:
        df = df.copy()
        df["assessmentClass"] = target_encoder.fit_transform(y_col.astype(str))
    else:
        df = df.copy()
        df["assessmentClass"] = target_encoder.transform(y_col.astype(str))

    cat_cols = df.select_dtypes(include="object").columns.tolist()
    df_ohe = pd.get_dummies(df, columns=cat_cols)
    bool_cols = df_ohe.select_dtypes(include="bool").columns
    if len(bool_cols):
        df_ohe[bool_cols] = df_ohe[bool_cols].astype(np.int64)
    return df_ohe, cat_cols


def add_classification_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["submission_offset"] = df["date_submitted"] - df["date"]
    df["submitted_late"] = (df["submission_offset"] > 0).astype(int)
    df["submitted_early"] = (df["submission_offset"] < -3).astype(int)
    df["submission_timing_ratio"] = df["submission_offset"] / (df["module_presentation_length"] + 1)

    df["click_per_day"] = df["total_clicks"] / (df["interaction_days"] + 1)
    df["avg_clicks_per_interaction"] = df["total_clicks"] / (df["total_interactions"] + 1)
    df["activity_diversity_ratio"] = df["unique_activity_types"] / (df["total_interactions"] + 1)
    df["engagement_ratio"] = (df["interaction_days"] / (df["module_presentation_length"] + 1)).clip(upper=1.0)

    click_threshold = df["total_clicks"].quantile(0.75)
    df["is_heavy_engager"] = (df["total_clicks"] >= click_threshold).astype(int)

    df["registration_offset"] = df["date_registration"]
    df["has_prior_attempts"] = (df["num_of_prev_attempts"] > 0).astype(int)
    median_credits = df["studied_credits"].median()
    df["high_credit_load"] = (df["studied_credits"] > median_credits).astype(int)

    df["engaged_but_late"] = ((df["is_heavy_engager"] == 1) & (df["submitted_late"] == 1)).astype(int)
    df["low_engagement_late"] = ((df["is_heavy_engager"] == 0) & (df["submitted_late"] == 1)).astype(int)

    df.drop(columns=["avg_clicks", "total_interactions", "interaction_days"], inplace=True, errors="ignore")
    return df


def build_classification_dataset(
    tab: Dict[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.Series, LabelEncoder, List[str], Dict[str, Any]]:
    df_merged = merge_classification_master(tab)
    df = classification_preprocess_before_learning(df_merged)
    le = LabelEncoder()
    df_dd, cat_cols_observed = label_encode_then_dummies(df, le, fit=True)

    inference_stats = {
        "click_quantile75": float(df_dd["total_clicks"].quantile(0.75)),
        "studied_credits_median": float(df_dd["studied_credits"].median()),
        "median_date": float(df_dd["date"].median()),
        "median_date_registration": float(df_dd["date_registration"].median()),
    }

    df_fe = add_classification_engineered(df_dd)
    y = df_fe["assessmentClass"]
    X = df_fe.drop(columns=["assessmentClass"])
    return X, y, le, cat_cols_observed, inference_stats


def classification_preprocess_without_target(
    df_in: pd.DataFrame, fill_medians: Optional[Mapping[str, float]] = None
) -> pd.DataFrame:
    df = df_in.copy()
    df.replace("?", np.nan, inplace=True)
    if "has_unregistered" not in df.columns:
        if "date_unregistration" in df.columns:
            df["has_unregistered"] = df["date_unregistration"].notna().astype(int)
        else:
            df["has_unregistered"] = 0
    if "date_unregistration" in df.columns:
        df.drop(columns=["date_unregistration"], inplace=True, errors="ignore")

    df.drop(columns=["id_assessment", "id_student"], inplace=True, errors="ignore")

    fm = dict(fill_medians or {})
    df["date"] = pd.to_numeric(df["date"], errors="coerce")
    df["date_registration"] = pd.to_numeric(df["date_registration"], errors="coerce")
    df["date_submitted"] = pd.to_numeric(df["date_submitted"], errors="coerce")
    md = fm.get("median_date")
    mr = fm.get("median_date_registration")
    df["date"] = df["date"].fillna(md if md is not None else float(df["date"].median()))
    df["date_registration"] = df["date_registration"].fillna(
        mr if mr is not None else float(df["date_registration"].median())
    )

    if "imd_band" in df.columns:
        df["imd_band"] = df["imd_band"].fillna("Unknown")

    vle_cols = ["total_clicks", "avg_clicks", "interaction_days", "total_interactions", "unique_activity_types"]
    for vc in vle_cols:
        if vc in df.columns:
            df[vc] = pd.to_numeric(df[vc], errors="coerce").fillna(0)
    return df


def add_classification_engineered_infer(
    df: pd.DataFrame,
    training_click_quantile: float,
    *,
    studied_credits_median_training: Optional[float] = None,
) -> pd.DataFrame:
    df = df.copy()
    df["submission_offset"] = df["date_submitted"] - df["date"]
    df["submitted_late"] = (df["submission_offset"] > 0).astype(int)
    df["submitted_early"] = (df["submission_offset"] < -3).astype(int)
    df["submission_timing_ratio"] = df["submission_offset"] / (df["module_presentation_length"] + 1)

    df["click_per_day"] = df["total_clicks"] / (df["interaction_days"] + 1)
    df["avg_clicks_per_interaction"] = df["total_clicks"] / (df["total_interactions"] + 1)
    df["activity_diversity_ratio"] = df["unique_activity_types"] / (df["total_interactions"] + 1)
    df["engagement_ratio"] = (df["interaction_days"] / (df["module_presentation_length"] + 1)).clip(upper=1.0)

    if np.isfinite(training_click_quantile):
        df["is_heavy_engager"] = (df["total_clicks"] >= training_click_quantile).astype(int)
    else:
        df["is_heavy_engager"] = (
            df["total_clicks"] >= df["total_clicks"].quantile(0.75)
        ).astype(int)

    df["registration_offset"] = df["date_registration"]
    df["has_prior_attempts"] = (df["num_of_prev_attempts"] > 0).astype(int)
    median_credits = (
        studied_credits_median_training
        if studied_credits_median_training is not None
        else float(df["studied_credits"].median())
    )
    df["high_credit_load"] = (df["studied_credits"] > median_credits).astype(int)

    df["engaged_but_late"] = ((df["is_heavy_engager"] == 1) & (df["submitted_late"] == 1)).astype(int)
    df["low_engagement_late"] = ((df["is_heavy_engager"] == 0) & (df["submitted_late"] == 1)).astype(int)

    df.drop(columns=["avg_clicks", "total_interactions", "interaction_days"], inplace=True, errors="ignore")
    return df


def transform_classification_inference_row(
    row: Mapping[str, Any],
    training_columns: Sequence[str],
    numeric_cols_scale: Sequence[str],
    inference_stats: Mapping[str, float],
) -> pd.DataFrame:
    df_in = pd.DataFrame([dict(row)])
    df = classification_preprocess_without_target(
        df_in,
        fill_medians={
            "median_date": float(inference_stats["median_date"]),
            "median_date_registration": float(inference_stats["median_date_registration"]),
        },
    )

    df_ohe = pd.get_dummies(df, columns=df.select_dtypes(include="object").columns.tolist())
    bool_cols = df_ohe.select_dtypes(include="bool").columns
    if len(bool_cols):
        df_ohe[bool_cols] = df_ohe[bool_cols].astype(np.int64)

    df_fe = add_classification_engineered_infer(
        df_ohe,
        float(inference_stats["click_quantile75"]),
        studied_credits_median_training=float(inference_stats["studied_credits_median"]),
    )
    aligned = df_fe.reindex(columns=list(training_columns), fill_value=0)
    aligned[list(numeric_cols_scale)] = (
        aligned[list(numeric_cols_scale)].apply(pd.to_numeric, errors="coerce").fillna(0)
    )
    return aligned
