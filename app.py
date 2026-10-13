"""
app.py
------
Streamlit dashboard for the Student Performance Prediction System.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

# Allow imports from src/ regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st

matplotlib.use("Agg")   # non-interactive backend for Streamlit

from src.data_loader import (
    load_raw_dataframe,
    get_train_test_split,
    preprocess,
    CATEGORICAL_COLS,
    NUMERICAL_COLS,
    FEATURE_LABELS,
    TARGET_COL,
)
from src.model import train_model, evaluate_model, get_feature_importance, predict_single

# ── page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS overrides ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #1f4e79; }
    .metric-label { font-size: 0.85rem; color: #555; margin-top: 4px; }
    .disclaimer {
        background: #fef9e7;
        border-left: 4px solid #d4a017;
        border-radius: 6px;
        padding: 12px 16px;
        font-size: 0.9rem;
        color: #5a3e00;
        line-height: 1.5;
    }
    @media (prefers-color-scheme: dark) {
        .disclaimer { background: #3d2e00; border-left-color: #f0b429; color: #fde9a2; }
    }
    [data-theme="dark"] .disclaimer { background: #3d2e00; border-left-color: #f0b429; color: #fde9a2; }
    .prediction-box {
        background: #e8f4fd;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        border: 2px solid #1f4e79;
    }
    .prediction-grade { font-size: 4rem; font-weight: 800; color: #1f4e79; }
    /* Helper text for sliders / dropdowns */
    .field-hint {
        font-size: 0.78rem;
        color: #666;
        margin-top: -8px;
        margin-bottom: 6px;
        line-height: 1.4;
    }
    @media (prefers-color-scheme: dark) {
        .field-hint { color: #aab4c2; }
    }
    /* Streamlit dark-theme override (data-theme attribute set by Streamlit) */
    [data-theme="dark"] .field-hint { color: #aab4c2; }
</style>
""", unsafe_allow_html=True)


# ── value-mapping helpers ─────────────────────────────────────────────────────

_EDU_MAP = {
    0: "None",
    1: "Primary education (4th grade)",
    2: "5th–9th grade",
    3: "Secondary education",
    4: "Higher education",
}

_STUDYTIME_MAP = {
    1: "Less than 2 hours / week",
    2: "2–5 hours / week",
    3: "5–10 hours / week",
    4: "More than 10 hours / week",
}

_TRAVELTIME_MAP = {
    1: "Less than 15 minutes",
    2: "15–30 minutes",
    3: "30 minutes – 1 hour",
    4: "More than 1 hour",
}

_FAMREL_MAP = {
    1: "Very bad",
    2: "Bad",
    3: "Average",
    4: "Good",
    5: "Excellent",
}

_HEALTH_MAP = {
    1: "Very bad",
    2: "Bad",
    3: "Average",
    4: "Good",
    5: "Very good",
}

_ALC_MAP = {
    1: "Very low",
    2: "Low",
    3: "Moderate",
    4: "High",
    5: "Very high",
}

_GOOUT_MAP = {
    1: "Very rarely",
    2: "Rarely",
    3: "Sometimes",
    4: "Often",
    5: "Very often",
}

_FREETIME_MAP = {
    1: "Very little",
    2: "Little",
    3: "Average",
    4: "Plenty",
    5: "A lot",
}


def _hint(text: str) -> None:
    """Render a small helper line beneath a widget."""
    st.markdown(f"<div class='field-hint'>ℹ️ {text}</div>", unsafe_allow_html=True)


# ── helpers ──────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Training and comparing models… (first run only)")
def load_and_train():
    """
    Load data, run cross-validation comparison, select best model,
    evaluate on test set.  Cached so re-runs are instant.

    Returns
    -------
    model, metrics, cv_results, best_name, feature_names, preprocessor,
    raw_df, X_test, y_test
    """
    X_train, X_test, y_train, y_test, feature_names, preprocessor, raw_df = (
        get_train_test_split(test_size=0.2, random_state=42)
    )
    model, cv_results, best_name = train_model(X_train, y_train, random_state=42)
    metrics = evaluate_model(model, X_test, y_test)
    return (
        model, metrics, cv_results, best_name,
        feature_names, preprocessor, raw_df, X_test, y_test,
    )


def grade_color(grade: float) -> str:
    if grade >= 16:
        return "#1a7a3c"
    if grade >= 10:
        return "#1f4e79"
    return "#c0392b"


def grade_label(grade: float) -> str:
    if grade >= 16:
        return "Excellent"
    if grade >= 14:
        return "Good"
    if grade >= 10:
        return "Satisfactory"
    return "At Risk"


# ── load data ────────────────────────────────────────────────────────────────

try:
    (
        model, metrics, cv_results, best_name,
        feature_names, preprocessor, raw_df, X_test, y_test,
    ) = load_and_train()
except ConnectionError as e:
    st.error("**Dataset download failed.**\n\n" + str(e))
    st.stop()
except Exception as e:
    st.error(f"**Unexpected error during initialisation:**\n\n{e}")
    st.stop()

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Inline SVG graduation cap — no external URL, no file dependency
    _CAP_SVG = (
        "data:image/svg+xml;charset=utf-8,"
        "%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 80'%3E"
        "%3Cpolygon points='50,5 95,28 50,51 5,28' fill='%231f4e79'/%3E"
        "%3Crect x='74' y='28' width='5' height='22' fill='%231f4e79'/%3E"
        "%3Cellipse cx='76.5' cy='52' rx='7' ry='4' fill='%231f4e79'/%3E"
        "%3Cpolygon points='50,51 85,34 85,54 50,71 15,54 15,34' fill='%235b9bd5'/%3E"
        "%3C/svg%3E"
    )
    st.image(_CAP_SVG, width=60)
    st.title("🎓 Student\nPerformance\nPredictor")
    st.caption("UCI Student Performance Dataset — Math course")
    st.markdown("---")
    st.markdown(f"""
**About this tool**

Enter a student's profile on the **Predict** tab and the model will
estimate their final grade (**G3**, 0–20 scale).

Features used: study habits, demographics, family background, and
school support — *not* prior term grades (G1/G2).

**Selected model:** {best_name}
""")
    st.markdown("---")
    st.markdown("**Dataset stats**")
    st.write(f"• {len(raw_df)} students")
    st.write(f"• {len(feature_names)} features")
    st.write(f"• Target: G3 (0–20)")
    st.markdown("---")
    st.markdown("""
<div class='disclaimer'>
⚠️ <b>Disclaimer:</b> Predictions are statistical estimates based on
historical patterns.  They are <em>not</em> guarantees of a student's
actual achievement or potential.
</div>
""", unsafe_allow_html=True)


# ── main tabs ─────────────────────────────────────────────────────────────────
tab_predict, tab_metrics, tab_data, tab_about = st.tabs(
    ["🔮 Predict", "📊 Model Metrics", "📂 Dataset", "ℹ️ About"]
)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.header("Predict Student Final Grade (G3)")
    st.markdown(
        "Fill in the student profile below and click **Predict** to get an estimated final grade."
    )

    # Gather current category values from the dataset for dropdowns
    raw_cats: dict = {}
    for col in CATEGORICAL_COLS:
        raw_cats[col] = sorted(raw_df[col].unique().tolist())

    with st.form("predict_form"):
        st.subheader("📋 Student Profile")

        col1, col2, col3 = st.columns(3)

        # ── demographics ──
        with col1:
            st.markdown("**Demographics**")

            school = st.selectbox(FEATURE_LABELS["school"], raw_cats["school"])
            _hint("GP = Gabriel Pereira  ·  MS = Mousinho da Silveira")

            sex = st.selectbox(FEATURE_LABELS["sex"], raw_cats["sex"])

            age = st.slider(FEATURE_LABELS["age"], int(raw_df["age"].min()), int(raw_df["age"].max()), 17)
            _hint(f"Student's current age: {age} years old")

            address = st.selectbox(FEATURE_LABELS["address"], raw_cats["address"])
            _hint("U = Urban address  ·  R = Rural address")

            famsize = st.selectbox(FEATURE_LABELS["famsize"], raw_cats["famsize"])
            _hint("LE3 = 3 or fewer family members  ·  GT3 = More than 3 family members")

            Pstatus = st.selectbox(FEATURE_LABELS["Pstatus"], raw_cats["Pstatus"])
            _hint("T = Parents live together  ·  A = Parents live apart")

            guardian = st.selectbox(FEATURE_LABELS["guardian"], raw_cats["guardian"])

        # ── family & school ──
        with col2:
            st.markdown("**Family & School**")

            Medu = st.slider(FEATURE_LABELS["Medu"], 0, 4, 2)
            _hint(f"Level {Medu}: {_EDU_MAP[Medu]}")

            Fedu = st.slider(FEATURE_LABELS["Fedu"], 0, 4, 2)
            _hint(f"Level {Fedu}: {_EDU_MAP[Fedu]}")

            Mjob = st.selectbox(FEATURE_LABELS["Mjob"], raw_cats["Mjob"])
            Fjob = st.selectbox(FEATURE_LABELS["Fjob"], raw_cats["Fjob"])
            reason = st.selectbox(FEATURE_LABELS["reason"], raw_cats["reason"])
            schoolsup = st.selectbox(FEATURE_LABELS["schoolsup"], raw_cats["schoolsup"])
            famsup = st.selectbox(FEATURE_LABELS["famsup"], raw_cats["famsup"])
            paid = st.selectbox(FEATURE_LABELS["paid"], raw_cats["paid"])

        # ── lifestyle ──
        with col3:
            st.markdown("**Lifestyle & Habits**")

            studytime = st.slider(FEATURE_LABELS["studytime"], 1, 4, 2)
            _hint(f"Value {studytime}: {_STUDYTIME_MAP[studytime]}")

            traveltime = st.slider(FEATURE_LABELS["traveltime"], 1, 4, 1)
            _hint(f"Value {traveltime}: {_TRAVELTIME_MAP[traveltime]}")

            failures = st.slider(FEATURE_LABELS["failures"], 0, 3, 0)
            if failures == 0:
                _hint("0 past class failures")
            elif failures == 1:
                _hint("1 past class failure")
            else:
                _hint(f"{failures} past class failures")

            activities = st.selectbox(FEATURE_LABELS["activities"], raw_cats["activities"])
            nursery = st.selectbox(FEATURE_LABELS["nursery"], raw_cats["nursery"])
            higher = st.selectbox(FEATURE_LABELS["higher"], raw_cats["higher"])
            internet = st.selectbox(FEATURE_LABELS["internet"], raw_cats["internet"])
            romantic = st.selectbox(FEATURE_LABELS["romantic"], raw_cats["romantic"])

            famrel = st.slider(FEATURE_LABELS["famrel"], 1, 5, 4)
            _hint(f"Value {famrel}: {_FAMREL_MAP[famrel]}")

            freetime = st.slider(FEATURE_LABELS["freetime"], 1, 5, 3)
            _hint(f"Value {freetime}: {_FREETIME_MAP[freetime]} free time after school")

            goout = st.slider(FEATURE_LABELS["goout"], 1, 5, 3)
            _hint(f"Value {goout}: {_GOOUT_MAP[goout]} goes out with friends")

            Dalc = st.slider(FEATURE_LABELS["Dalc"], 1, 5, 1)
            _hint(f"Value {Dalc}: {_ALC_MAP[Dalc]} workday alcohol consumption")

            Walc = st.slider(FEATURE_LABELS["Walc"], 1, 5, 1)
            _hint(f"Value {Walc}: {_ALC_MAP[Walc]} weekend alcohol consumption")

            health = st.slider(FEATURE_LABELS["health"], 1, 5, 3)
            _hint(f"Value {health}: {_HEALTH_MAP[health]} health status")

            absences = st.slider(FEATURE_LABELS["absences"], 0, 93, 4)
            _hint(f"{absences} school absence{'s' if absences != 1 else ''} recorded")

        submitted = st.form_submit_button("🔮 Predict Grade", use_container_width=True, type="primary")

    if submitted:
        # Build a raw single-row DataFrame matching feature order
        raw_input = {
            "school": school, "sex": sex, "address": address,
            "famsize": famsize, "Pstatus": Pstatus,
            "Mjob": Mjob, "Fjob": Fjob, "reason": reason, "guardian": guardian,
            "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
            "activities": activities, "nursery": nursery, "higher": higher,
            "internet": internet, "romantic": romantic,
            # numerical
            "age": age, "Medu": Medu, "Fedu": Fedu,
            "traveltime": traveltime, "studytime": studytime, "failures": failures,
            "famrel": famrel, "freetime": freetime, "goout": goout,
            "Dalc": Dalc, "Walc": Walc, "health": health, "absences": absences,
        }

        input_raw_df = pd.DataFrame([raw_input])

        # Transform using the fitted ColumnTransformer (same as training)
        try:
            input_transformed = preprocessor.transform(
                input_raw_df[CATEGORICAL_COLS + NUMERICAL_COLS]
            )
        except Exception as e:
            st.error(f"Input encoding failed: {e}")
            st.stop()

        try:
            predicted_grade = predict_single(model, input_transformed)
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        color = grade_color(predicted_grade)
        label = grade_label(predicted_grade)
        mae   = metrics["mae"]

        st.markdown("---")
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.markdown(f"""
<div class='prediction-box'>
    <div style='font-size:1rem; color:#555; margin-bottom:8px;'>Estimated Final Grade</div>
    <div class='prediction-grade' style='color:{color};'>{predicted_grade:.1f}</div>
    <div style='font-size:1.2rem; font-weight:600; color:{color}; margin-top:4px;'>{label}</div>
    <div style='font-size:0.85rem; color:#777; margin-top:8px;'>out of 20</div>
    <div style='font-size:0.78rem; color:#999; margin-top:6px;'>
        Typical error: ±{mae:.1f} grade points (MAE on test set)
    </div>
</div>
""", unsafe_allow_html=True)

        with res_col2:
            # Grade scale bar
            fig, ax = plt.subplots(figsize=(5, 1.2))
            ax.barh(0, 20, color="#e0e0e0", height=0.4)
            ax.barh(0, predicted_grade, color=color, height=0.4)
            ax.set_xlim(0, 20)
            ax.set_yticks([])
            ax.set_xlabel("Grade (0–20)")
            ax.axvline(10, color="#888", linewidth=1, linestyle="--")
            ax.set_title("Grade Scale", fontsize=10)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown(f"""
<div class='disclaimer'>
⚠️ <b>Note:</b> This prediction is a <em>statistical estimate</em> based on historical patterns
from the UCI Student Performance dataset.  The model's typical error on unseen students is
approximately <b>±{mae:.1f} grade points</b> (MAE).  It is <em>not</em> a guarantee of a
student's actual performance, potential, or future achievement.
Educators should use this as one data point among many, never as a final judgment.
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.header("📊 Model Evaluation Metrics")
    st.markdown(
        f"Metrics computed on the **held-out test set** (20% of data, {len(y_test)} students). "
        f"Algorithm selected: **{best_name}** (lowest cross-validation MAE across 3 candidates). "
        "The model was trained with `random_state=42` for full reproducibility."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""
<div class='metric-card'>
    <div class='metric-value'>{metrics['mae']:.3f}</div>
    <div class='metric-label'>Mean Absolute Error (MAE)</div>
    <div style='font-size:0.75rem;color:#888;margin-top:6px;'>avg. grade-point error</div>
</div>
""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
<div class='metric-card'>
    <div class='metric-value'>{metrics['rmse']:.3f}</div>
    <div class='metric-label'>Root Mean Squared Error (RMSE)</div>
    <div style='font-size:0.75rem;color:#888;margin-top:6px;'>penalises large errors more</div>
</div>
""", unsafe_allow_html=True)
    with c3:
        r2_pct = metrics['r2'] * 100
        st.markdown(f"""
<div class='metric-card'>
    <div class='metric-value'>{metrics['r2']:.3f}</div>
    <div class='metric-label'>R² Score</div>
    <div style='font-size:0.75rem;color:#888;margin-top:6px;'>
        model explains {r2_pct:.1f}% of variance
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ── cross-validation comparison table ──
    st.subheader("Model Comparison (5-fold Cross-Validation on Training Set)")
    st.markdown(
        "Three regressors were compared using identical folds. "
        "The test set was kept separate throughout. "
        "Lower MAE is better."
    )
    cv_rows = []
    for alg, res in cv_results.items():
        cv_rows.append({
            "Algorithm": alg,
            "CV MAE (mean)": f"{res['mean_mae']:.4f}",
            "CV MAE (±std)": f"{res['std_mae']:.4f}",
            "CV RMSE (mean)": f"{res['mean_rmse']:.4f}",
            "CV RMSE (±std)": f"{res['std_rmse']:.4f}",
            "Selected": "✅" if alg == best_name else "",
        })
    st.dataframe(pd.DataFrame(cv_rows).set_index("Algorithm"), use_container_width=True)

    st.markdown("---")

    chart_col1, chart_col2 = st.columns(2)

    # ── actual vs predicted scatter ──
    with chart_col1:
        st.subheader("Actual vs. Predicted Grades")
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(y_test, metrics["y_pred"], alpha=0.6, color="#1f4e79", edgecolors="white", s=55)
        lo, hi = 0, 20
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.5, label="Perfect prediction")
        ax.set_xlabel("Actual G3")
        ax.set_ylabel("Predicted G3")
        ax.set_title(f"Actual vs. Predicted (test set) — {best_name}")
        ax.legend(fontsize=8)
        ax.set_xlim(lo - 0.5, hi + 0.5)
        ax.set_ylim(lo - 0.5, hi + 0.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # ── residuals ──
    with chart_col2:
        st.subheader("Residuals Distribution")
        residuals = np.array(y_test) - metrics["y_pred"]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.hist(residuals, bins=20, color="#1f4e79", edgecolor="white", alpha=0.8)
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5)
        ax.set_xlabel("Residual (Actual − Predicted)")
        ax.set_ylabel("Count")
        ax.set_title("Residual Distribution")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")

    # ── feature importance ──
    st.subheader("Top 15 Feature Importances")
    st.markdown(
        "Feature importance scores indicate which variables the model relies on most "
        "to split data — they do **not** imply causation and should not be used to "
        "make claims about individual students."
    )
    importance = get_feature_importance(model, feature_names)
    top15 = dict(list(importance.items())[:15])
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [FEATURE_LABELS.get(k, k) for k in top15.keys()]
    values = list(top15.values())
    colors = ["#1f4e79" if v > np.mean(values) else "#5b9bd5" for v in values]
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Feature Importance ({best_name})")
    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("""
> **Limitations of feature importance:**
> Scores reflect correlation patterns in historical data from two Portuguese schools.
> They may not generalise to other contexts, and should never be used to label or
> rank individual students.
""")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATASET EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_data:
    st.header("📂 Dataset Explorer")
    st.markdown(
        f"**UCI Student Performance (Math course)**  •  {len(raw_df)} rows × {len(raw_df.columns)} columns"
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Total students", len(raw_df))
        st.metric("Features used by model", len(feature_names))
    with col_b:
        st.metric("Missing values", int(raw_df.isnull().sum().sum()))
        st.metric("Mean G3 (final grade)", f"{raw_df[TARGET_COL].mean():.2f}")

    st.markdown("---")
    st.subheader("Grade Distribution (G3)")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.hist(raw_df[TARGET_COL], bins=range(0, 22), color="#1f4e79", edgecolor="white", alpha=0.85)
    ax.set_xlabel("G3 (Final Grade)")
    ax.set_ylabel("Number of Students")
    ax.set_title("Distribution of Final Grades")
    ax.axvline(raw_df[TARGET_COL].mean(), color="red", linestyle="--",
               label=f"Mean = {raw_df[TARGET_COL].mean():.1f}")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Raw Data Preview")
    st.dataframe(raw_df.head(50), use_container_width=True)

    st.subheader("Summary Statistics")
    st.dataframe(raw_df.describe(), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.header("ℹ️ About This Project")
    st.markdown(f"""
### Student Performance Prediction System

This dashboard was built as a machine-learning internship demonstration project.
It predicts the **final mathematics grade (G3)** of secondary-school students
using the **UCI Student Performance dataset**.

---

#### 🔬 Methodology

| Step | Detail |
|------|--------|
| Dataset | UCI Student Performance (Math) — 395 students, 33 columns |
| Target | G3 — final period grade (0–20) |
| Excluded | G1 and G2 (earlier period grades — not available at prediction time) |
| Features | 30 demographic, family, and study-habit variables |
| Preprocessing | scikit-learn `ColumnTransformer`: `OrdinalEncoder` (categorical) + `StandardScaler` (numerical) |
| Leakage prevention | Preprocessor fitted on training set only; test set transformed with same fitted object |
| Unknown categories | `OrdinalEncoder` maps unseen values to −1 (no silent fallback to 0) |
| Split | 80% train / 20% test (`random_state=42`) |
| Model selection | 5-fold CV on training set across RandomForest, ExtraTrees, GradientBoosting |
| Selected algorithm | **{best_name}** (lowest mean CV MAE) |
| Test-set evaluation | Single evaluation after model selection (no re-use) |

---

#### 📊 Measured Performance (held-out test set, {len(y_test)} students)

| Metric | Value |
|--------|-------|
| MAE    | {metrics['mae']:.4f} grade points |
| RMSE   | {metrics['rmse']:.4f} grade points |
| R²     | {metrics['r2']:.4f} |

> R² is modest because G1/G2 are deliberately excluded.
> Predictions from study habits and demographics alone carry inherent variance.
> These numbers are the actual measured values — not estimates or targets.

---

#### 📚 Data Source

P. Cortez and A. Silva. *Using Data Mining to Predict Secondary School Student Performance.*
In A. Brito and J. Teixeira (eds.), Proceedings of the 5th Future Business Technology Conference
(FUBUTEC 2008), pp. 5–12, Porto, Portugal, April 2008.

Dataset: [UCI Machine Learning Repository #320](https://archive.ics.uci.edu/dataset/320/student+performance)

---

#### ⚠️ Important Disclaimer

> Predictions generated by this tool are **statistical estimates** derived from
> historical patterns in a specific dataset (two Portuguese schools, 2008).
> They do **not** account for individual circumstances, teacher quality, personal
> effort changes, or any factor not present in the data.  This tool should be
> used as a supplementary reference only — **not** as a definitive assessment
> of any student's ability or future performance.  Do not use predictions to
> limit any student's opportunities.

---

#### 🛠️ Technology Stack

- **Python 3.x** · **pandas** · **NumPy**
- **scikit-learn** (ColumnTransformer, OrdinalEncoder, StandardScaler, RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor)
- **Streamlit** (dashboard)
- **Matplotlib** (charts)
    """)
