from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.base import clone

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "global_supply_chain_risk.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
BUNDLE_PATH = MODEL_DIR / "model_bundle.joblib"

st.set_page_config(page_title="Supply Chain Risk Intelligence", page_icon="🚢", layout="wide")

st.title("🚢 AI-Powered Supply Chain Bottleneck & Anomaly Detection")
st.caption("Classification predicts disruption risk; anomaly detectors flag unusual shipment patterns. These are decision-support signals, not guarantees.")

TARGET = "Disruption_Occurred"
ID_COLS = ["Shipment_ID"]
POST_OUTCOME_OR_SUSPECT = ["Lead_Time_Days"]
NUMERIC = [
    "Distance_km", "Weight_MT", "Fuel_Price_Index",
    "Geopolitical_Risk_Score", "Carrier_Reliability_Score",
    "Shipment_Month", "Shipment_DayOfWeek"
]
CATEGORICAL = [
    "Origin_Port", "Destination_Port", "Transport_Mode",
    "Product_Category", "Weather_Condition"
]
FEATURES = NUMERIC + CATEGORICAL
PORTS = ["Antwerp", "Busan", "Dubai", "Hamburg", "Los Angeles", "Marseille", "Rotterdam", "Shanghai", "Singapore"]
MODES = ["Air", "Rail", "Road", "Sea"]
PRODUCTS = ["Automotive", "Electronics", "Perishables", "Pharmaceuticals", "Textiles"]
WEATHER = ["Clear", "Fog", "Hurricane", "Rain", "Storm"]


@st.cache_data
def read_data():
    data = pd.read_csv(DATA_PATH)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Shipment_Month"] = data["Date"].dt.month
    data["Shipment_DayOfWeek"] = data["Date"].dt.dayofweek
    return data


def build_preprocessor():
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC),
        ("cat", categorical_pipe, CATEGORICAL)
    ])


def make_models():
    models = {}
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=250, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            objective="binary:logistic", eval_metric="logloss",
            random_state=42, n_jobs=4
        )
    except Exception:
        pass
    models["Random Forest (baseline)"] = RandomForestClassifier(
        n_estimators=250, min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1
    )
    return models


def fit_bundle(data):
    usable = data.dropna(subset=[TARGET]).copy()
    X = usable[FEATURES].copy()
    y = usable[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    model_rows, fitted_models = [], {}
    for name, estimator in make_models().items():
        estimator.fit(X_train_t, y_train)
        pred = estimator.predict(X_test_t)
        try:
            prob = estimator.predict_proba(X_test_t)[:, list(estimator.classes_).index(1)]
            auc = roc_auc_score(y_test, prob)
        except Exception:
            auc = np.nan
        model_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "ROC-AUC": auc
        })
        fitted_models[name] = estimator

    metrics = pd.DataFrame(model_rows).sort_values("F1", ascending=False)
    best_name = str(metrics.iloc[0]["Model"])
    # Keep the preprocessing object and fitted estimators together for reproducible inference.
    bundle = {
        "preprocessor": preprocessor,
        "models": fitted_models,
        "best_model_name": best_name,
        "metrics": metrics,
        "feature_names": FEATURES,
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "random_state": 42,
    }

    # Fit unsupervised detectors on training data only (no target used).
    bundle["isolation_forest"] = IsolationForest(
        n_estimators=250, contamination="auto", random_state=42
    ).fit(X_train_t)
    bundle["one_class_svm"] = OneClassSVM(kernel="rbf", nu=0.05, gamma="scale").fit(X_train_t)

    # Leakage audit summary: explicit exclusions and temporal split limitation.
    bundle["leakage_audit"] = {
        "target": TARGET,
        "excluded_columns": ID_COLS + POST_OUTCOME_OR_SUSPECT + ["Date"],
        "split": "Stratified 80/20 holdout before fitting preprocessing; random_state=42",
        "preprocessing": "Imputers, scaler, and one-hot encoder fit on training features only",
        "note": (
            "This is a leakage-risk audit, not proof of zero leakage. Lead_Time_Days is excluded "
            "because it may be known only after shipment progression/outcome. Shipment_ID is an identifier. "
            "Date is converted to calendar features. Weather_Condition has 100% disruption in Hurricane "
            "rows in this dataset; investigate possible synthetic target rules or proxy leakage."
        )
    }
    joblib.dump(bundle, BUNDLE_PATH)
    return bundle


@st.cache_resource
def load_or_train_bundle():
    # Re-train from the included dataset to avoid relying on an incompatible old pickle.
    return fit_bundle(read_data())


def risk_probability(model, X_transformed):
    if not hasattr(model, "predict_proba"):
        return None
    classes = list(getattr(model, "classes_", [0, 1]))
    probabilities = model.predict_proba(X_transformed)[0]
    return float(probabilities[classes.index(1)]) if 1 in classes else float(probabilities[-1])


def leakage_page(data, bundle):
    st.header("🔎 Data leakage & data quality audit")
    st.write("This page documents the checks used in this project. It is a risk audit—not a mathematical guarantee that no leakage exists.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(data):,}")
    c2.metric("Columns", len(data.columns))
    c3.metric("Missing cells", f"{int(data.isna().sum().sum()):,}")

    st.subheader("Columns deliberately excluded from model features")
    st.dataframe(pd.DataFrame([
        {"Column": "Shipment_ID", "Reason": "Identifier; not a predictive shipment attribute."},
        {"Column": "Lead_Time_Days", "Reason": "Potential post-outcome variable; excluded to reduce target leakage risk."},
        {"Column": "Date", "Reason": "Raw date excluded; month and weekday derived from it."},
        {"Column": TARGET, "Reason": "Label only; never included in X."},
    ]), use_container_width=True, hide_index=True)

    st.subheader("Leakage prevention steps")
    st.markdown("""
- Split into train/test sets **before** fitting imputation, scaling, and encoding.
- Fit preprocessing only on the training subset; transform the test subset afterward.
- Keep the target `Disruption_Occurred` separate from the feature matrix.
- Exclude identifiers and potentially post-outcome `Lead_Time_Days`.
- Report holdout metrics separately from training.
    """)

    hurricane = data[data["Weather_Condition"].astype(str).str.lower().eq("hurricane")]
    if len(hurricane):
        hurricane_rate = float(hurricane[TARGET].mean())
        st.warning(
            f"Dataset warning: {len(hurricane)} rows have Weather_Condition = Hurricane, "
            f"and the observed disruption rate is {hurricane_rate:.0%}. "
            "That perfect association can indicate a synthetic rule or target proxy. "
            "Check how the dataset was generated before claiming real-world generalization."
        )
    st.subheader("Model validation metrics (held-out test set)")
    st.dataframe(bundle["metrics"].style.format({
        "Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
        "F1": "{:.3f}", "ROC-AUC": "{:.3f}"
    }), use_container_width=True, hide_index=True)
    st.caption("Random stratified holdout is useful for a demo. For future-date deployment, add a time-based split and external validation.")


data = read_data()
with st.sidebar:
    st.header("Navigation")
    page = st.radio("Choose page", ["Shipment analysis", "Model evaluation", "Leakage audit"])
    threshold = st.slider("Disruption alert threshold", 0.10, 0.90, 0.50, 0.05)
    st.caption("Models train from the bundled CSV when the app first starts.")
    if st.button("Retrain models"):
        st.cache_resource.clear()
        st.cache_data.clear()
        st.rerun()

try:
    bundle = load_or_train_bundle()
except Exception as exc:
    st.error("Model training failed. Confirm requirements are installed and the CSV is present in data/.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.stop()

if page == "Shipment analysis":
    st.header("📦 Shipment analysis: anomaly detection → disruption classification")
    st.write("Enter shipment details once. Isolation Forest and One-Class SVM check for unusual patterns first; then the selected classifier estimates disruption risk.")
    with st.form("shipment_form"):
        a, b, c = st.columns(3)
        with a:
            planned_date = st.date_input("Planned shipment date", value=pd.Timestamp.today().date())
            origin = st.selectbox("Origin port", PORTS, index=6)
            destination = st.selectbox("Destination port", [p for p in PORTS if p != origin])
            mode = st.selectbox("Transport mode", MODES, index=3)
        with b:
            product = st.selectbox("Product category", PRODUCTS, index=1)
            weather = st.selectbox("Weather condition", WEATHER, index=0)
            distance = st.number_input("Distance (km)", 500.0, 15000.0, 7700.0, 100.0)
            weight = st.number_input("Weight (MT)", 1.0, 500.0, 245.0, 5.0)
        with c:
            fuel = st.number_input("Fuel price index", 1.2, 4.5, 2.85, 0.05)
            geo = st.slider("Geopolitical risk score", 0.0, 10.0, 5.0, 0.1)
            reliability = st.slider("Carrier reliability score", 0.5, 1.0, 0.75, 0.01)
        submitted = st.form_submit_button("🔍 Detect anomaly and predict disruption", type="primary", use_container_width=True)

    if submitted:
        raw = pd.DataFrame([{
            "Distance_km": distance, "Weight_MT": weight, "Fuel_Price_Index": fuel,
            "Geopolitical_Risk_Score": geo, "Carrier_Reliability_Score": reliability,
            "Shipment_Month": planned_date.month, "Shipment_DayOfWeek": planned_date.weekday(),
            "Origin_Port": origin, "Destination_Port": destination, "Transport_Mode": mode,
            "Product_Category": product, "Weather_Condition": weather
        }])[FEATURES]
        Xt = bundle["preprocessor"].transform(raw)

        # Stage 1: unsupervised anomaly detection; these detectors do not use target labels.
        st.divider()
        st.subheader("Stage 1 — Anomaly detection")
        iso = bundle["isolation_forest"]
        svm = bundle["one_class_svm"]
        iso_label = int(iso.predict(Xt)[0])
        svm_label = int(svm.predict(Xt)[0])
        iso_score = float(iso.decision_function(Xt)[0])
        svm_score = float(svm.decision_function(Xt)[0])
        m1, m2 = st.columns(2)
        m1.metric("Isolation Forest", "Possible anomaly" if iso_label == -1 else "Typical pattern", f"Decision score: {iso_score:.4f}")
        m2.metric("One-Class SVM", "Possible anomaly" if svm_label == -1 else "Typical pattern", f"Decision score: {svm_score:.4f}")
        anomaly_flag = (iso_label == -1 or svm_label == -1)
        if anomaly_flag:
            st.warning("At least one detector flags this shipment as unusual. Classification will still run; consider manual review.")
        else:
            st.success("Neither detector flags this shipment as unusual. This does not guarantee that disruption will not occur.")
        st.caption("The two anomaly score scales are different and should not be compared directly.")

        # Stage 2: supervised classification always runs, including for flagged anomalies.
        st.subheader("Stage 2 — Disruption classification")
        model_name = bundle["best_model_name"]
        model = bundle["models"][model_name]
        prob = risk_probability(model, Xt)
        pred = int(prob >= threshold) if prob is not None else int(model.predict(Xt)[0])
        x, y = st.columns(2)
        x.metric("Selected classifier", model_name)
        y.metric("Estimated disruption probability", f"{prob:.1%}" if prob is not None else "Not available")
        if prob is not None:
            st.progress(float(np.clip(prob, 0, 1)))
        if pred:
            st.error("⚠️ Disruption alert: the model score is at or above the selected threshold.")
        else:
            st.success("✅ Below the selected alert threshold. Disruption is still possible.")
        if anomaly_flag:
            st.warning("Combined interpretation: unusual pattern detected. Prioritize manual review even if the classifier score is low.")
        with st.expander("View model input"):
            st.dataframe(raw, use_container_width=True, hide_index=True)

elif page == "Model evaluation":
    st.header("📊 Model evaluation")
    st.write("Metrics below are computed on a held-out test split; preprocessing was fit only on the training split.")
    st.dataframe(bundle["metrics"].style.format({
        "Accuracy": "{:.3f}", "Precision": "{:.3f}", "Recall": "{:.3f}",
        "F1": "{:.3f}", "ROC-AUC": "{:.3f}"
    }), use_container_width=True, hide_index=True)
    st.info(f"Selected model by holdout F1: **{bundle['best_model_name']}**")
    st.caption(f"Training rows: {bundle['train_rows']:,} | Test rows: {bundle['test_rows']:,} | Random state: {bundle['random_state']}")
    st.write("Model selection should consider false negatives and false positives—not accuracy alone.")

else:
    leakage_page(data, bundle)

with st.expander("Project notes"):
    st.markdown("""
- **XGBoost** is a supervised classifier used for disruption prediction.
- **Isolation Forest** and **One-Class SVM** are unsupervised detectors for unusual shipment profiles.
- `Shipment_ID` is excluded as an identifier.
- `Lead_Time_Days` is excluded because it may be post-outcome information.
- Missing values are imputed inside the preprocessing pipeline fitted on training data only.
- The included data has a strong association between Hurricane weather and disruption; investigate whether this reflects a synthetic data-generation rule.
- These models are educational prototypes and need external/time-based validation before operational use.
    """)
