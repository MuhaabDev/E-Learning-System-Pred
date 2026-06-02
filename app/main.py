"""
streamlit run app/main.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.model_loader import load_all
from src.predict import (
    predict_classification,
    predict_classification_batch,
    predict_regression,
    predict_regression_batch,
)
from utils.preprocess_input import regression_feature_frame

# ─── Design tokens ────────────────────────────────────────────────────────────
ACCENT      = "#6C63FF"
ACCENT2     = "#FF6584"
ACCENT3     = "#43D9AD"
ACCENT4     = "#FFB347"
DARK_BG     = "#0F1117"
CARD_BG     = "#1A1D27"
BORDER      = "#2A2D3E"
TEXT_MUTED  = "#8B8FA8"
TEXT_MAIN   = "#E8E9F0"
PLOTLY_COLORS = [ACCENT, ACCENT2, ACCENT3, ACCENT4, "#A78BFA", "#FB923C"]
PLOTLY_TEMPLATE = "plotly_dark"
IMAGES_DIR = ROOT / "images"
CLF_IMGS   = IMAGES_DIR / "classification"
REG_IMGS   = IMAGES_DIR / "regression"

# ─── CSS ──────────────────────────────────────────────────────────────────────
GLOBAL_CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
  html, body, [class*="css"] {{ font-family: 'Space Grotesk', sans-serif; }}
  .stApp {{ background: {DARK_BG}; color: {TEXT_MAIN}; }}
  [data-testid="stSidebar"] {{ background: {CARD_BG} !important; border-right: 1px solid {BORDER}; }}
  [data-testid="metric-container"] {{
    background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 12px;
    padding: 16px 20px !important; transition: border-color 0.2s ease;
  }}
  [data-testid="metric-container"]:hover {{ border-color: {ACCENT}; }}
  [data-testid="metric-container"] [data-testid="stMetricLabel"] {{
    color: {TEXT_MUTED} !important; font-size: 0.75rem !important;
    font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
  }}
  [data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {TEXT_MAIN} !important; font-size: 1.6rem !important;
    font-weight: 700; font-family: 'JetBrains Mono', monospace;
  }}
  .stTabs [data-baseweb="tab-list"] {{
    gap: 4px; background: {CARD_BG}; border-radius: 10px;
    padding: 4px; border: 1px solid {BORDER};
  }}
  .stTabs [data-baseweb="tab"] {{
    border-radius: 7px !important; color: {TEXT_MUTED} !important;
    font-weight: 500; padding: 6px 16px; font-size: 0.85rem;
  }}
  .stTabs [aria-selected="true"] {{ background: {ACCENT} !important; color: white !important; }}
  [data-testid="stExpander"] {{
    background: {CARD_BG}; border: 1px solid {BORDER}; border-radius: 12px; margin-bottom: 8px;
  }}
  h1 {{ font-size: 1.9rem !important; font-weight: 700; color: {TEXT_MAIN} !important; letter-spacing: -0.02em; }}
  h2 {{ font-size: 1.25rem !important; font-weight: 600; color: {TEXT_MAIN} !important; }}
  hr {{ border-color: {BORDER} !important; }}
  .stButton > button {{
    background: {ACCENT} !important; color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important; padding: 8px 24px !important;
    font-family: 'Space Grotesk', sans-serif !important;
  }}
  .insight-card {{
    background: linear-gradient(135deg, rgba(108,99,255,0.1), rgba(67,217,173,0.07));
    border: 1px solid rgba(108,99,255,0.25); border-radius: 10px;
    padding: 14px 18px; margin-top: 10px;
    font-size: 0.85rem; color: {TEXT_MUTED}; line-height: 1.6;
  }}
  .insight-card b {{ color: {ACCENT3}; }}
  .section-label {{
    font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: {ACCENT}; margin-bottom: 6px;
  }}
  .page-header {{
    background: linear-gradient(135deg, rgba(108,99,255,0.12), rgba(255,101,132,0.06));
    border: 1px solid rgba(108,99,255,0.2); border-radius: 14px;
    padding: 20px 24px; margin-bottom: 28px;
  }}
  .status-pill {{
    display: inline-block; background: rgba(67,217,173,0.15); color: {ACCENT3};
    border: 1px solid rgba(67,217,173,0.3); border-radius: 20px;
    padding: 3px 12px; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.05em;
  }}
</style>
"""

# ─── Cache ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model artifacts…")
def artifacts():
    return load_all()

# ─── Utilities ────────────────────────────────────────────────────────────────
def insight(text):
    st.markdown(f'<div class="insight-card">{text}</div>', unsafe_allow_html=True)

def section_label(text):
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)

def page_header(title, subtitle, icon="📊"):
    st.markdown(f"""
    <div class="page-header">
      <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;
                  color:{ACCENT};margin-bottom:4px;">E-Learning ML Dashboard</div>
      <div style="font-size:1.7rem;font-weight:700;color:{TEXT_MAIN};letter-spacing:-0.02em;">
        {icon} {title}
      </div>
      <div style="font-size:0.88rem;color:{TEXT_MUTED};margin-top:4px;">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def _no_eval_warning():
    st.warning(
        "⚠️ **No evaluation data found.** Run training scripts:\n"
        "```\npython src/train_regression.py\npython src/train_classification.py\n```",
        icon="⚠️",
    )

def _img(path, caption=""):
    if Path(path).is_file():
        st.image(str(path), caption=caption, use_container_width=True)
        return True
    return False

def _plotly_layout(fig, title="", height=400):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Space Grotesk, sans-serif", color=TEXT_MUTED),
        title=dict(text=title, font=dict(size=14, color=TEXT_MAIN), x=0.01),
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=BORDER),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER)
    return fig

# ─── Band helpers ─────────────────────────────────────────────────────────────
BAND_COLORS = {"Excellent": ACCENT3, "Very Good": ACCENT, "Good": ACCENT4, "Fail": ACCENT2}
BAND_ICONS  = {"Excellent": "🏆", "Very Good": "⭐", "Good": "✅", "Fail": "⚠️"}

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Overview
# ──────────────────────────────────────────────────────────────────────────────
def page_overview(art):
    page_header("E-Learning ML Analytics Platform",
        "Student assessment outcome prediction — Regression & Classification · E-Learning Analytics Dataset", "🎓")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="section-label">Platform Status</div>', unsafe_allow_html=True)
        st.markdown('<span class="status-pill">🟢 MODELS LOADED</span>', unsafe_allow_html=True)
    reg_meta = art.metadata.get("regression", {})
    clf_meta = art.metadata.get("classification", {})
    c2.metric("Regression Features", str(len(reg_meta.get("feature_columns", []))))
    n_cls = len(clf_meta.get("class_names", clf_meta.get("label_classes", [])))
    c3.metric("Classification Classes", str(n_cls) if n_cls else "4")
    c4.metric("Dataset", "OULAD")
    st.divider()

    col_r, col_c = st.columns(2, gap="large")
    with col_r:
        st.markdown(f"""
        <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:14px;padding:20px;border-top:3px solid {ACCENT};">
          <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{ACCENT};margin-bottom:8px;">REGRESSION MODEL</div>
          <div style="font-size:1.2rem;font-weight:700;color:{TEXT_MAIN};margin-bottom:4px;">📉 Score Prediction</div>
          <div style="font-size:0.84rem;color:{TEXT_MUTED};line-height:1.6;">Predicts a student's numeric assessment score (0–100) from engagement, demographics, and submission behaviour. Best: <b style="color:{ACCENT3};">Gradient Boosting</b> R²≈0.28.</div>
          <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;">
            <span style="background:rgba(108,99,255,0.12);color:{ACCENT};border:1px solid rgba(108,99,255,0.3);border-radius:6px;padding:3px 10px;font-size:0.75rem;font-weight:600;">MAE≈12.1</span>
            <span style="background:rgba(108,99,255,0.12);color:{ACCENT};border:1px solid rgba(108,99,255,0.3);border-radius:6px;padding:3px 10px;font-size:0.75rem;font-weight:600;">RMSE≈15.9</span>
            <span style="background:rgba(108,99,255,0.12);color:{ACCENT};border:1px solid rgba(108,99,255,0.3);border-radius:6px;padding:3px 10px;font-size:0.75rem;font-weight:600;">R²≈0.28</span>
          </div>
        </div>""", unsafe_allow_html=True)
    with col_c:
        st.markdown(f"""
        <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:14px;padding:20px;border-top:3px solid {ACCENT2};">
          <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{ACCENT2};margin-bottom:8px;">CLASSIFICATION MODEL</div>
          <div style="font-size:1.2rem;font-weight:700;color:{TEXT_MAIN};margin-bottom:4px;">🏷️ Band Prediction</div>
          <div style="font-size:0.84rem;color:{TEXT_MUTED};line-height:1.6;">Classifies students into bands: <b style="color:{ACCENT3};">Excellent, Very Good, Good, Fail</b>. Best: <b style="color:{ACCENT3};">Random Forest</b> Accuracy≈51%.</div>
          <div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;">
            <span style="background:rgba(255,101,132,0.1);color:{ACCENT2};border:1px solid rgba(255,101,132,0.25);border-radius:6px;padding:3px 10px;font-size:0.75rem;font-weight:600;">Accuracy≈51%</span>
            <span style="background:rgba(255,101,132,0.1);color:{ACCENT2};border:1px solid rgba(255,101,132,0.25);border-radius:6px;padding:3px 10px;font-size:0.75rem;font-weight:600;">4 Classes</span>
            <span style="background:rgba(255,101,132,0.1);color:{ACCENT2};border:1px solid rgba(255,101,132,0.25);border-radius:6px;padding:3px 10px;font-size:0.75rem;font-weight:600;">F1 Macro</span>
          </div>
        </div>""", unsafe_allow_html=True)
    st.divider()
    section_label("DATASET SNAPSHOT — REGRESSION TASK")
    _img(REG_IMGS / "1.png")
    insight("<b>Score Distribution:</b> Scores cluster toward 75–100 with mean≈75.8. Clicks are heavily right-skewed and log-normalised before training.")
    st.divider()
    section_label("DATASET SNAPSHOT — CLASSIFICATION TASK")
    _img(CLF_IMGS / "1.png")
    insight("<b>Class Distribution:</b> 'Very Good' dominates (~58k). Moderate class imbalance — model may under-predict 'Good'. Consider weighted training for better minority recall.")
    st.divider()
    with st.expander("ℹ️ About this platform"):
        st.markdown(f"""<div style="font-size:0.88rem;color:{TEXT_MUTED};line-height:1.8;">
        OULAD-based ML models trained across two milestones.<br>
        <b style="color:{TEXT_MAIN};">Milestone 1 Regression</b>: Predict numeric scores from engagement features.<br>
        <b style="color:{TEXT_MAIN};">Milestone 2 Classification</b>: Predict student bands from demographic + behavioural features.<br>
        Re-run training scripts to refresh evaluation data.</div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Regression Analytics
# ──────────────────────────────────────────────────────────────────────────────
def page_regression_analytics(art):
    page_header("Regression Analytics", "Student score prediction — engagement features → numeric grade", "📉")
    reg_eval = art.eval_data.get("regression") if art.eval_data else None
    reg_meta = art.metadata.get("regression", {})

    if reg_eval:
        metrics   = reg_eval["metrics"]
        mae_val   = metrics.get("MAE", 0)
        mse_val   = metrics.get("MSE", 0)
        rmse_val  = metrics.get("RMSE", 0)
        r2_val    = metrics.get("R²", 0)
        best_name = reg_eval.get("best_model_name", reg_meta.get("best_model_val", "Gradient Boosting"))
    elif reg_meta:
        mae_val   = reg_meta.get("test_mae", 0)
        mse_val   = reg_meta.get("test_mse", reg_meta.get("test_rmse", 0)**2)
        rmse_val  = reg_meta.get("test_rmse", 0)
        r2_val    = reg_meta.get("test_r2", 0)
        best_name = reg_meta.get("best_model_val", "Gradient Boosting")
    else:
        _no_eval_warning(); return

    section_label("MODEL PERFORMANCE SUMMARY")
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("MAE",       f"{mae_val:.3f}",  help="Mean Absolute Error")
    c2.metric("RMSE",      f"{rmse_val:.3f}", help="Root Mean Squared Error")
    c3.metric("MSE",       f"{mse_val:.3f}",  help="Mean Squared Error")
    c4.metric("R² Score",  f"{r2_val:.4f}",   help="Variance explained")
    c5.metric("Best Model",best_name)
    st.divider()

    tab_perf, tab_resid, tab_feat, tab_compare, tab_eda = st.tabs([
        "🎯 Performance", "📐 Residuals", "🔍 Features", "🏆 Comparison", "📊 EDA"])

    with tab_perf:
        col_l, col_r = st.columns(2, gap="large")
        with col_l:
            section_label("ACTUAL VS PREDICTED — ALL MODELS")
            _img(REG_IMGS / "5.png")
            insight("<b>Gradient Boosting</b> tightest spread (R²=0.28). Decision Tree shows grid artefact from quantised leaves — classic overfitting.")
        with col_r:
            if reg_eval:
                section_label("INTERACTIVE — TEST SET")
                y_test = reg_eval["y_test"]; y_pred = reg_eval["y_pred"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers",
                    marker=dict(color=ACCENT, size=4, opacity=0.4), name="Predictions"))
                lims = [min(float(y_test.min()),float(y_pred.min())), max(float(y_test.max()),float(y_pred.max()))]
                fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                    line=dict(color=ACCENT2, dash="dash", width=2), name="Perfect"))
                _plotly_layout(fig, "Actual vs Predicted (Test Set)", 380)
                fig.update_xaxes(title_text="Actual Score"); fig.update_yaxes(title_text="Predicted Score")
                st.plotly_chart(fig, use_container_width=True)
                insight(f"Test R²: <b>{r2_val:.4f}</b> — model explains ~{r2_val*100:.1f}% of variance. Spread below diagonal at high scores = under-prediction of top performers.")

    with tab_resid:
        col_l, col_r = st.columns(2, gap="large")
        with col_l:
            section_label("RESIDUAL DISTRIBUTION (GRADIENT BOOSTING)")
            _img(REG_IMGS / "6.png")
            insight("Mean residual≈0.15 (near zero — <b>no bias</b>). Fan shape in scatter shows heteroscedasticity: errors larger for mid-range predictions.")
        with col_r:
            if reg_eval:
                section_label("INTERACTIVE RESIDUALS — TEST SET")
                y_test = reg_eval["y_test"]; y_pred = reg_eval["y_pred"]
                residuals = y_test - y_pred
                fig = make_subplots(rows=1, cols=2, subplot_titles=["Residuals vs Predicted", "Residual Distribution"])
                fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers",
                    marker=dict(color=ACCENT2, size=3, opacity=0.3), name="Residuals"), row=1, col=1)
                fig.add_hline(y=0, line_color=ACCENT3, line_dash="dash", line_width=1.5)
                fig.add_trace(go.Histogram(x=residuals, nbinsx=50, marker_color=ACCENT, opacity=0.8), row=1, col=2)
                _plotly_layout(fig, "", 340); st.plotly_chart(fig, use_container_width=True)
                ca,cb,cc = st.columns(3)
                ca.metric("Mean Residual", f"{float(residuals.mean()):.3f}")
                cb.metric("Residual Std",  f"{float(residuals.std()):.3f}")
                cc.metric("Max |Error|",   f"{float(np.abs(residuals).max()):.1f}")

    with tab_feat:
        section_label("FEATURE IMPORTANCE — GRADIENT BOOSTING & RIDGE")
        _img(REG_IMGS / "7.png")
        insight("<b>assessment_type</b> tops — TMA/CMA/Exam differ markedly in scoring. <b>weight</b> and <b>interaction_days</b> follow. Engagement proxies confirm active platform use predicts better outcomes.")
        if reg_eval:
            fi = reg_eval.get("feature_importance")
            if fi is not None:
                st.divider(); section_label("INTERACTIVE — TRAINED MODEL")
                top_n = st.slider("Top N features", 5, min(30, len(fi)), 15, key="reg_fi_n")
                fi_top = fi.head(top_n).sort_values()
                fig = go.Figure(go.Bar(x=fi_top.values, y=fi_top.index, orientation="h",
                    marker=dict(color=fi_top.values, colorscale=[[0,"#2A2D3E"],[1,ACCENT]], showscale=False),
                    text=[f"{v:.4f}" for v in fi_top.values], textposition="outside",
                    textfont=dict(size=10, color=TEXT_MUTED)))
                _plotly_layout(fig, f"Top {top_n} Feature Importances", max(300, top_n*28))
                fig.update_xaxes(title_text="Importance"); st.plotly_chart(fig, use_container_width=True)

    with tab_compare:
        section_label("VALIDATION — MODEL COMPARISON")
        _img(REG_IMGS / "4.png")
        insight("<b>Gradient Boosting wins</b> (R²=0.28, RMSE=15.9). Lasso worst — linear penalty too aggressive. Random Forest close second (R²=0.25).")
        st.divider(); section_label("5-FOLD CROSS-VALIDATION")
        _img(REG_IMGS / "8.png")
        insight("Ranking stable across folds. GBM mean R²≈0.27–0.29 — good generalisation, low variance.")
        st.divider(); section_label("OVERFITTING & LEARNING CURVE")
        _img(REG_IMGS / "9.png")
        insight("Train R²≈0.32 vs Val R²≈0.25 — <b>mild overfitting</b> that reduces with data. Val R² still rising: more data would help.")

    with tab_eda:
        section_label("SCORE & FEATURE DISTRIBUTIONS")
        _img(REG_IMGS / "1.png")
        st.divider(); section_label("CORRELATION MATRIX")
        _img(REG_IMGS / "2.png")
        insight("Click features form a strongly correlated cluster. Consider PCA or dropping redundant members to reduce multicollinearity.")
        st.divider(); section_label("IMD BAND vs SCORE")
        _img(REG_IMGS / "3.png")
        insight("Score distributions similar across IMD bands — deprivation index alone is not a strong predictor.")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Classification Analytics
# ──────────────────────────────────────────────────────────────────────────────
def page_classification_analytics(art):
    page_header("Classification Analytics", "Student band prediction — Excellent · Very Good · Good · Fail", "🏷️")
    clf_eval = art.eval_data.get("classification") if art.eval_data else None
    clf_meta = art.metadata.get("classification", {})

    if clf_eval:
        metrics     = clf_eval["metrics"]
        acc         = metrics.get("Accuracy", 0)
        prec        = metrics.get("Precision (macro)", 0)
        rec         = metrics.get("Recall (macro)", 0)
        f1          = metrics.get("F1 Score (macro)", 0)
        roc_auc     = clf_eval.get("roc_auc")
        best_name   = clf_eval.get("best_model_name", "Random Forest")
        class_names = clf_eval["class_names"]
        cm          = clf_eval["confusion_matrix"]
    elif clf_meta:
        acc=clf_meta.get("accuracy_test",0); prec=clf_meta.get("macro_prec_test",0)
        rec=clf_meta.get("macro_rec_test",0); f1=clf_meta.get("macro_f1_test",0)
        roc_auc=None; best_name="Random Forest"
        class_names=["Excellent","Fail","Good","Very Good"]; cm=None
    else:
        _no_eval_warning(); return

    section_label("MODEL PERFORMANCE SUMMARY")
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    c1.metric("Accuracy",          f"{acc:.4f}")
    c2.metric("Precision (macro)", f"{prec:.4f}")
    c3.metric("Recall (macro)",    f"{rec:.4f}")
    c4.metric("F1 Macro",          f"{f1:.4f}")
    c5.metric("ROC-AUC",           f"{roc_auc:.4f}" if roc_auc else "—")
    c6.metric("Best Model",        best_name)
    st.divider()

    tab_matrix, tab_dist, tab_feat, tab_report, tab_compare, tab_eda = st.tabs([
        "🔲 Confusion Matrix","📊 Class Distribution","🔍 Features","📋 Report","🏆 Comparison","📊 EDA"])

    with tab_matrix:
        col_l, col_r = st.columns([1.1,1], gap="large")
        with col_l:
            section_label("CONFUSION MATRICES — ALL MODELS")
            _img(CLF_IMGS / "7.png")
            insight("<b>Random Forest</b>: best diagonal — Excellent 63%, Fail 55%. 'Good' hardest across all models — confused with 'Very Good'.")
        with col_r:
            if cm is not None:
                section_label("INTERACTIVE — TEST SET")
                cm_arr = np.array(cm, dtype=float)
                cm_norm = (cm_arr.T / cm_arr.sum(axis=1)).T
                fig = go.Figure(go.Heatmap(
                    z=cm_norm, x=list(class_names), y=list(class_names),
                    colorscale=[[0,DARK_BG],[0.5,ACCENT+"88"],[1,ACCENT]],
                    text=[[f"{cm_norm[i][j]:.2f}\n({int(cm_arr[i][j])})" for j in range(len(class_names))] for i in range(len(class_names))],
                    texttemplate="%{text}", textfont=dict(size=11), showscale=True))
                _plotly_layout(fig, "Normalised Confusion Matrix", 370)
                fig.update_xaxes(title_text="Predicted"); fig.update_yaxes(title_text="True", autorange="reversed")
                st.plotly_chart(fig, use_container_width=True)
                insight("Diagonal = per-class recall. Off-diagonal = misclassification rate between pairs.")

    with tab_dist:
        col_l, col_r = st.columns(2, gap="large")
        with col_l:
            section_label("TARGET CLASS DISTRIBUTION")
            _img(CLF_IMGS / "1.png")
            insight("'Very Good' dominates (~58k). 'Good' is smallest (~26k). Consider balanced training.")
        with col_r:
            if clf_eval:
                td = clf_eval.get("target_distribution")
                if td is not None:
                    section_label("CLASS COUNTS — TEST SPLIT")
                    fig = go.Figure(go.Bar(
                        x=list(td.index), y=list(td.values),
                        marker=dict(color=PLOTLY_COLORS[:len(td)], line=dict(width=0)),
                        text=[f"{v:,}" for v in td.values], textposition="outside",
                        textfont=dict(size=12, color=TEXT_MUTED)))
                    _plotly_layout(fig, "Class Distribution (Test Set)", 350)
                    fig.update_xaxes(title_text="Class"); fig.update_yaxes(title_text="Count")
                    st.plotly_chart(fig, use_container_width=True)

    with tab_feat:
        section_label("TOP FEATURES CORRELATED WITH ASSESSMENT CLASS")
        col_l, col_r = st.columns(2, gap="large")
        with col_l:
            _img(CLF_IMGS / "2.png")
            insight("<b>assessment_type_TMA/CMA</b> strongest predictors. Assessment type shapes outcomes more than demographics.")
        with col_r:
            _img(CLF_IMGS / "4.png")
            insight("<b>submitted_late</b> and <b>low_engagement_late</b> dominate engineered features. Late low-engagement = strong Fail predictor.")
        if clf_eval:
            fi = clf_eval.get("feature_importance")
            if fi is not None:
                st.divider(); section_label("INTERACTIVE — TRAINED MODEL")
                top_n = st.slider("Top N features", 5, min(30, len(fi)), 15, key="clf_fi_n")
                fi_top = fi.head(top_n).sort_values()
                fig = go.Figure(go.Bar(x=fi_top.values, y=fi_top.index, orientation="h",
                    marker=dict(color=fi_top.values, colorscale=[[0,"#2A2D3E"],[1,ACCENT2]], showscale=False),
                    text=[f"{v:.4f}" for v in fi_top.values], textposition="outside",
                    textfont=dict(size=10, color=TEXT_MUTED)))
                _plotly_layout(fig, f"Top {top_n} Feature Importances", max(300, top_n*28))
                fig.update_xaxes(title_text="Importance"); st.plotly_chart(fig, use_container_width=True)

    with tab_report:
        section_label("PER-CLASS METRICS")
        if clf_eval:
            report = clf_eval.get("classification_report", {})
            rows = []
            for label in class_names:
                if label in report:
                    r = report[label]
                    rows.append({"Class":label,"Precision":round(r["precision"],4),
                        "Recall":round(r["recall"],4),"F1-Score":round(r["f1-score"],4),"Support":int(r["support"])})
            if rows:
                df_report = pd.DataFrame(rows)
                fig = go.Figure()
                for col_name, color in [("Precision",ACCENT),("Recall",ACCENT2),("F1-Score",ACCENT3)]:
                    fig.add_trace(go.Bar(x=df_report["Class"], y=df_report[col_name],
                        name=col_name, marker_color=color, opacity=0.85))
                _plotly_layout(fig, "Per-Class Precision / Recall / F1", 360)
                fig.update_layout(barmode="group"); st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df_report.style.background_gradient(
                    subset=["Precision","Recall","F1-Score"], cmap="Blues"),
                    use_container_width=True, hide_index=True)
                insight("'Good' has lowest F1 across models. 'Fail' detection critical for early intervention — prioritise its recall.")
        else:
            st.info("Classification report not available — re-run training.")

    with tab_compare:
        section_label("ACCURACY & TRAINING TIME — ALL MODELS")
        _img(CLF_IMGS / "6.png")
        insight("<b>Random Forest wins (51.2%)</b> with fast test time (0.012s). GBM second (48.3%) but 564s training. RF = best accuracy/speed trade-off for production.")

    with tab_eda:
        section_label("FULL CORRELATION MATRIX")
        _img(CLF_IMGS / "3.png")
        insight("Strong cluster: click/interaction features highly inter-correlated. Gender/region nearly orthogonal. IMD band forms negative block from dummy encoding.")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Data Insights
# ──────────────────────────────────────────────────────────────────────────────
def page_data_insights(art, task):
    page_header("Data Insights", f"{task} — dataset exploration, distributions, feature analysis", "🔬")
    is_reg = task == "Regression"
    meta = art.metadata.get("regression" if is_reg else "classification", {})
    feat_cols = meta.get("feature_columns", [])
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Features", str(len(feat_cols)) if feat_cols else "—")
    c2.metric("Target", "Score (0–100)" if is_reg else "Assessment Band")
    c3.metric("Task", task)
    c4.metric("Dataset", "OULAD")
    st.divider()
    if is_reg:
        tab1,tab2,tab3 = st.tabs(["📊 Distributions","🔗 Correlations","📋 Feature List"])
        with tab1:
            section_label("SCORE & ENGAGEMENT DISTRIBUTIONS")
            _img(REG_IMGS / "1.png")
            insight("Score bimodal with peaks near 50 and 100. Clicks extremely skewed — log-transformed to near-bell for training.")
            st.divider(); section_label("IMD BAND vs SCORE")
            _img(REG_IMGS / "3.png")
        with tab2:
            section_label("CORRELATION MATRIX")
            _img(REG_IMGS / "2.png")
        with tab3:
            if feat_cols:
                st.dataframe(pd.DataFrame({"#":range(1,len(feat_cols)+1),"Feature":feat_cols}),
                    use_container_width=True, hide_index=True, height=400)
    else:
        tab1,tab2,tab3 = st.tabs(["📊 Distributions","🔗 Correlations","📋 Feature List"])
        with tab1:
            section_label("TARGET CLASS DISTRIBUTION")
            _img(CLF_IMGS / "1.png")
        with tab2:
            section_label("FULL FEATURE CORRELATION MATRIX")
            _img(CLF_IMGS / "3.png")
            st.divider(); section_label("TOP FEATURES CORRELATED WITH TARGET")
            cl, cr = st.columns(2, gap="large")
            with cl: _img(CLF_IMGS / "2.png")
            with cr: _img(CLF_IMGS / "4.png")
        with tab3:
            keys = feat_cols or list((meta.get("default_input_row") or {}).keys())
            if keys:
                st.dataframe(pd.DataFrame({"#":range(1,len(keys)+1),"Feature":keys}),
                    use_container_width=True, hide_index=True, height=400)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Regression Predict
# ──────────────────────────────────────────────────────────────────────────────
def page_regression_predict(art):
    page_header("Score Prediction", "Enter student feature values to predict assessment score", "🔮")
    mode = st.radio("Mode", ["✏️ Manual Input","📂 CSV Batch"], horizontal=True)
    st.divider()
    if mode.startswith("✏️"):
        _regression_manual(art)
    else:
        _regression_batch(art)

def _regression_manual(art):
    meta = art.metadata["regression"]
    cols = meta["feature_columns"]; defaults = meta.get("defaults_numeric", {})
    st.markdown(f'<div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:12px;padding:16px 20px;margin-bottom:20px;font-size:0.84rem;color:{TEXT_MUTED};">Defaults are dataset medians from the training fold.</div>', unsafe_allow_html=True)
    values = {}; n = len(cols)
    with st.expander("📋 Feature Inputs", expanded=True):
        for i in range(0, n, 3):
            row_cols = st.columns(min(3, n-i))
            for j, col in enumerate(cols[i:i+3]):
                with row_cols[j]:
                    values[col] = st.number_input(col, value=float(defaults.get(col,0.0)), format="%.6g", key=f"reg_{col}")
    st.divider()
    if st.button("🚀 Predict Score", key="predict_reg"):
        X = regression_feature_frame(values, cols)
        preds = predict_regression(art, X)
        pred_val = float(preds[0])
        grade_color = ACCENT3 if pred_val>=70 else ACCENT4 if pred_val>=50 else ACCENT2
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(108,99,255,0.12),rgba(67,217,173,0.07));
                    border:2px solid {grade_color};border-radius:16px;padding:28px 32px;text-align:center;margin-top:20px;">
          <div style="font-size:0.8rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{TEXT_MUTED};margin-bottom:8px;">PREDICTED ASSESSMENT SCORE</div>
          <div style="font-size:4rem;font-weight:800;color:{grade_color};font-family:'JetBrains Mono',monospace;line-height:1.1;">{pred_val:.1f}</div>
          <div style="font-size:0.9rem;color:{TEXT_MUTED};margin-top:8px;">out of 100</div>
        </div>""", unsafe_allow_html=True)

def _regression_batch(art):
    st.markdown(f'<div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:12px;padding:16px 20px;margin-bottom:20px;font-size:0.84rem;color:{TEXT_MUTED};">CSV columns should match regression features. Missing filled with 0.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="reg_batch_upload")
    if uploaded is None: return
    try: df_in = pd.read_csv(uploaded)
    except Exception as e: st.error(f"Failed: {e}"); return
    c1,c2 = st.columns(2); c1.metric("Rows",f"{len(df_in):,}"); c2.metric("Cols",str(len(df_in.columns)))
    with st.expander("👁️ Preview"): st.dataframe(df_in.head(), use_container_width=True)
    if st.button("🚀 Run Batch Prediction", key="reg_batch_predict"):
        with st.spinner("Predicting…"): preds = predict_regression_batch(art, df_in)
        df_out = df_in.copy(); df_out["predicted_score"] = preds
        st.success(f"✅ {len(df_out):,} predictions done.")
        ca,cb,cc = st.columns(3)
        ca.metric("Mean Score",f"{preds.mean():.2f}"); cb.metric("Min",f"{preds.min():.2f}"); cc.metric("Max",f"{preds.max():.2f}")
        fig = go.Figure(go.Histogram(x=preds, nbinsx=40, marker_color=ACCENT, opacity=0.85))
        _plotly_layout(fig, "Predicted Score Distribution", 280)
        fig.update_xaxes(title_text="Score"); fig.update_yaxes(title_text="Count")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_out.head(20), use_container_width=True)
        st.download_button("⬇️ Download CSV", df_out.to_csv(index=False).encode(), "regression_predictions.csv", "text/csv")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: Classification Predict
# ──────────────────────────────────────────────────────────────────────────────
def page_classification_predict(art):
    page_header("Band Prediction", "Predict assessment band: Excellent · Very Good · Good · Fail", "🔮")
    mode = st.radio("Mode", ["✏️ Manual Input","📂 CSV Batch"], horizontal=True)
    st.divider()
    if mode.startswith("✏️"): _classification_manual(art)
    else: _classification_batch(art)

def _classification_manual(art):
    clf_meta = art.metadata["classification"]
    seed_full = clf_meta.get("default_input_row") or {}; vocab = clf_meta.get("categorical_vocab", {})
    st.markdown(f'<div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:12px;padding:16px 20px;margin-bottom:20px;font-size:0.84rem;color:{TEXT_MUTED};">Defaults from a reference training row. Categorical = dropdowns; numeric = number inputs.</div>', unsafe_allow_html=True)
    touched = {}
    with st.expander("🏷️ Categorical Fields", expanded=True):
        cat_cols_sorted = sorted(vocab.keys())
        for i in range(0,len(cat_cols_sorted),3):
            row_cols = st.columns(min(3,len(cat_cols_sorted)-i))
            for j,col in enumerate(cat_cols_sorted[i:i+3]):
                opts = vocab[col]; cur = seed_full.get(col, opts[0] if opts else "")
                ix = opts.index(cur) if cur in opts else 0
                with row_cols[j]: touched[col] = st.selectbox(col, options=opts, index=ix, key=f"clf_cat_{col}")
    leftover = sorted(set(seed_full.keys())-set(touched.keys())-{"assessmentClass"})
    with st.expander("🔢 Numeric Fields", expanded=True):
        for i in range(0,len(leftover),3):
            row_cols = st.columns(min(3,len(leftover)-i))
            for j,col in enumerate(leftover[i:i+3]):
                try: v0 = float(seed_full.get(col,0))
                except: v0 = 0.0
                with row_cols[j]: touched[col] = st.number_input(str(col), value=v0, format="%.6g", key=f"clf_num_{col}")
    st.divider()
    if st.button("🚀 Predict Band", type="primary", key="predict_clf"):
        raw_row = {**seed_full, **touched}
        _, label = predict_classification(art, raw_row=raw_row)
        pred_label = label[0]
        band_color = BAND_COLORS.get(pred_label, ACCENT)
        band_icon  = BAND_ICONS.get(pred_label, "📌")
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,rgba(108,99,255,0.12),rgba(67,217,173,0.07));
                    border:2px solid {band_color};border-radius:16px;padding:28px 32px;text-align:center;margin-top:20px;">
          <div style="font-size:0.8rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{TEXT_MUTED};margin-bottom:8px;">PREDICTED ASSESSMENT BAND</div>
          <div style="font-size:3.5rem;line-height:1.1;">{band_icon}</div>
          <div style="font-size:2.8rem;font-weight:800;color:{band_color};font-family:'JetBrains Mono',monospace;margin-top:4px;">{pred_label}</div>
        </div>""", unsafe_allow_html=True)

def _classification_batch(art):
    st.markdown(f'<div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:12px;padding:16px 20px;margin-bottom:20px;font-size:0.84rem;color:{TEXT_MUTED};">CSV: raw pre-encoding columns (same as Manual Input). Preprocessing applied automatically.</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="clf_batch_upload")
    if uploaded is None: return
    try: df_in = pd.read_csv(uploaded)
    except Exception as e: st.error(f"Failed: {e}"); return
    c1,c2 = st.columns(2); c1.metric("Rows",f"{len(df_in):,}"); c2.metric("Cols",str(len(df_in.columns)))
    with st.expander("👁️ Preview"): st.dataframe(df_in.head(), use_container_width=True)
    if st.button("🚀 Run Batch Prediction", key="clf_batch_predict"):
        with st.spinner("Predicting…"): _enc, decoded = predict_classification_batch(art, df_in)
        df_out = df_in.copy(); df_out["predicted_class"] = decoded
        errors = int((decoded=="ERROR").sum())
        st.success(f"✅ {len(df_out)-errors:,} predictions." + (f" ⚠️ {errors} errors." if errors else ""))
        band_counts = pd.Series(decoded).value_counts()
        fig = go.Figure(go.Bar(
            x=list(band_counts.index), y=list(band_counts.values),
            marker=dict(color=[BAND_COLORS.get(b,ACCENT) for b in band_counts.index], line=dict(width=0)),
            text=[f"{v:,}" for v in band_counts.values], textposition="outside"))
        _plotly_layout(fig, "Prediction Band Distribution", 300); st.plotly_chart(fig, use_container_width=True)
        st.dataframe(df_out.head(20), use_container_width=True)
        st.download_button("⬇️ Download CSV", df_out.to_csv(index=False).encode(), "classification_predictions.csv", "text/csv")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE: About
# ──────────────────────────────────────────────────────────────────────────────
def page_about():
    page_header("About This Platform", "Project structure, methodology, and usage guide", "ℹ️")
    c1,c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(f"""
        <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:14px;padding:20px 24px;margin-bottom:16px;">
          <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:{ACCENT};margin-bottom:10px;">PROJECT STRUCTURE</div>
          <div style="font-size:0.85rem;color:{TEXT_MUTED};line-height:2;font-family:'JetBrains Mono',monospace;">
            app/main.py &nbsp;&nbsp;&nbsp;&nbsp; → Streamlit UI<br>
            src/train_*.py &nbsp;&nbsp; → Training pipelines<br>
            src/predict.py &nbsp;&nbsp; → Inference helpers<br>
            models/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; → Persisted pickles<br>
            images/ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; → Notebook charts
          </div>
        </div>
        <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:14px;padding:20px 24px;">
          <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:{ACCENT2};margin-bottom:10px;">HOW TO RETRAIN</div>
          <div style="font-size:0.85rem;color:{TEXT_MUTED};line-height:2;font-family:'JetBrains Mono',monospace;">
            pip install -r requirements.txt<br>
            python src/train_regression.py<br>
            python src/train_classification.py<br>
            streamlit run app/main.py
          </div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:14px;padding:20px 24px;margin-bottom:16px;">
          <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:{ACCENT3};margin-bottom:10px;">METHODOLOGY</div>
          <div style="font-size:0.85rem;color:{TEXT_MUTED};line-height:1.9;">
            <b style="color:{TEXT_MAIN};">Milestone 1 — Regression</b><br>
            Score from engagement, demographics, submission timing. Five models: Ridge, Lasso, DT, RF, GBM.<br><br>
            <b style="color:{TEXT_MAIN};">Milestone 2 — Classification</b><br>
            Band from demo + behavioural features + engineered late-submission flags. Four models: LR, DT, RF, GBM.
          </div>
        </div>
        <div style="background:{CARD_BG};border:1px solid {BORDER};border-radius:14px;padding:20px 24px;">
          <div style="font-size:0.7rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:{ACCENT4};margin-bottom:10px;">DATASET</div>
          <div style="font-size:0.85rem;color:{TEXT_MUTED};line-height:1.9;">
            <b style="color:{TEXT_MAIN};">Open University Learning Analytics Dataset (OULAD)</b><br>
            ~170k student-assessment records across 7 modules. Features span demographics, VLE logs, and assessment outcomes.
          </div>
        </div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="OULAD ML Dashboard", page_icon="🎓",
                       layout="wide", initial_sidebar_state="expanded")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    art = artifacts()

    with st.sidebar:
        st.markdown(f"""
        <div style="padding:12px 0 20px;text-align:center;">
          <div style="font-size:2rem;">🎓</div>
          <div style="font-size:1.05rem;font-weight:700;color:{TEXT_MAIN};letter-spacing:-0.02em;">OULAD ML Platform</div>
          <div style="font-size:0.72rem;color:{TEXT_MUTED};margin-top:3px;">Analytics · Prediction · Insights</div>
        </div>""", unsafe_allow_html=True)
        st.divider()
        st.markdown(f'<div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{TEXT_MUTED};margin-bottom:6px;">ML TASK</div>', unsafe_allow_html=True)
        task = st.radio("task_select", ["Regression","Classification"], label_visibility="collapsed")
        st.divider()
        st.markdown(f'<div style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;color:{TEXT_MUTED};margin-bottom:6px;">PAGE</div>', unsafe_allow_html=True)
        page = st.radio("page_select",
            ["🏠 Overview","📊 Model Analytics","🔬 Data Insights","🔮 Prediction","ℹ️ About"],
            label_visibility="collapsed")
        st.divider()
        st.markdown(f'<div style="font-size:0.75rem;color:{TEXT_MUTED};line-height:1.7;padding-top:4px;"><span class="status-pill">🟢 LIVE</span><br>Models from models/.<br>Re-train to refresh.</div>', unsafe_allow_html=True)

    is_reg = task == "Regression"
    if page.startswith("🏠"):    page_overview(art)
    elif page.startswith("📊"):  page_regression_analytics(art) if is_reg else page_classification_analytics(art)
    elif page.startswith("🔬"):  page_data_insights(art, task)
    elif page.startswith("🔮"):  page_regression_predict(art) if is_reg else page_classification_predict(art)
    else:                        page_about()

if __name__ == "__main__":
    main()
