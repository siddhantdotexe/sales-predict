# app.py - Universal Sales Predictor (No server-side save; improved UI)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px
import streamlit.components.v1 as components

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# -----------------------
# ASSET: uploaded file path (dev note: this will be transformed by infra)
# -----------------------
ASSET_URL = "/mnt/data/logs-siddhantdotexe-sales-predict-main-app.py-2025-11-20T15_47_54.591Z.txt"

# -----------------------
# Page / theme tweaks
# -----------------------
st.set_page_config(page_title="Universal Sales Predictor", page_icon="🛒", layout="wide")
primary = "#0f766e"   # teal
accent = "#0369a1"    # blue

# small CSS for aesthetics
st.markdown(
    f"""
    <style>
    .main-card {{
        background: linear-gradient(135deg, rgba(15,118,110,0.08), rgba(3,105,161,0.04));
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 6px 24px rgba(16,24,40,0.06);
    }}
    .header-title {{
        font-size:28px;
        font-weight:700;
        color: {primary};
        margin: 0;
    }}
    .muted {{
        color: #6b7280;
    }}
    .pill {{
        display:inline-block;padding:6px 12px;border-radius:999px;background:#eef2ff;color:#3730a3;font-weight:600;font-size:12px;
    }}
    </style>
    """, unsafe_allow_html=True
)

# Header
with st.container():
    c1, c2 = st.columns([4,1])
    with c1:
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.markdown('<div style="display:flex;align-items:center;gap:14px;">', unsafe_allow_html=True)
        st.markdown(f'<div><h1 class="header-title">🛒 Universal Sales Predictor</h1><div class="muted">Upload any sales dataset, train quickly, and predict — downloads only (no server saves).</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with c2:
        # show the uploaded file link (developer: path will be converted to URL)
        try:
            # if it is an image the platform will serve; otherwise show as link
            st.markdown(f'<a class="pill" href="{ASSET_URL}" target="_blank">View uploaded asset</a>', unsafe_allow_html=True)
        except Exception:
            pass

st.write("")  # spacing

# -------------------------
# helper functions
# -------------------------
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def build_preprocessor(df, feature_cols):
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # scikit-learn >=1.2 uses sparse_output
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe)
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, cat_cols)
    ], remainder="drop", sparse_threshold=0)

    return preprocessor, numeric_cols, cat_cols

def train_model(pipeline, X_train, y_train, X_val=None, y_val=None):
    pipeline.fit(X_train, y_train)
    train_preds = pipeline.predict(X_train)
    stats = {
        "r2_train": r2_score(y_train, train_preds),
        "rmse_train": rmse(y_train, train_preds),
        "mae_train": mean_absolute_error(y_train, train_preds)
    }
    if X_val is not None and y_val is not None:
        val_preds = pipeline.predict(X_val)
        stats.update({
            "r2_val": r2_score(y_val, val_preds),
            "rmse_val": rmse(y_val, val_preds),
            "mae_val": mean_absolute_error(y_val, val_preds)
        })
    return pipeline, stats

def make_download_button_bytes(obj_bytes: bytes, filename: str, label: str):
    return st.download_button(label=label, data=obj_bytes, file_name=filename, mime="application/octet-stream")

# -------------------------
# Data upload / sample
# -------------------------
st.sidebar.header("Data & Model")
uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"], help="CSV with numeric target (sales).")
use_sample = st.sidebar.checkbox("Use sample dataset (toy)", value=False)

if use_sample:
    df = px.data.tips().rename(columns={"total_bill":"Item_MRP", "tip":"Item_Outlet_Sales"})
    df["Outlet_Type"] = np.random.choice(["Supermarket Type1","Grocery Store","Supermarket Type2"], size=len(df))
    st.sidebar.success("Sample dataset loaded")
elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        # clear any previous pipeline if dataset changes
        if "pipeline" in st.session_state:
            del st.session_state["pipeline"]
            st.sidebar.info("Previous model cleared — please retrain for this dataset.")
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")
        st.stop()
else:
    st.info("Upload a CSV to start or toggle 'Use sample dataset' in the sidebar.")
    st.stop()

# main two-column layout: left for train/predict, right for analytics
left, right = st.columns([2,1])

# -------------------------
# Left: Training + Prediction
# -------------------------
with left:
    st.subheader("1) Configure dataset & train")
    st.markdown("**Preview**")
    st.dataframe(df.head())

    st.markdown("**Choose target column (numeric)**")
    target_col = st.selectbox("Target column", options=df.columns.tolist(), index=len(df.columns)-1)

    if not pd.api.types.is_numeric_dtype(df[target_col]):
        st.warning("Target column is not numeric. We will attempt to coerce to numeric (bad rows -> NaN).")
        try:
            df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
            if df[target_col].isna().all():
                st.error("Target conversion failed — supply a numeric target column.")
                st.stop()
            else:
                st.info("Target coerced to numeric, NaNs introduced for non-numeric rows.")
        except Exception:
            st.stop()

    st.markdown("**Features (predictors)**")
    default_features = [c for c in df.columns if c != target_col]
    # remove accidental duplication
    if target_col in default_features:
        default_features.remove(target_col)

    feature_cols = st.multiselect("Select features", options=default_features, default=default_features)

    if len(feature_cols) == 0:
        st.error("Pick at least one feature.")
        st.stop()

    drop_na_target = st.checkbox("Drop rows with missing target", value=True)
    if drop_na_target:
        df = df.dropna(subset=[target_col])

    st.markdown("**Train / validation split**")
    test_size = st.slider("Validation fraction", 0.05, 0.5, 0.2, step=0.05)

    st.markdown("**Model selection**")
    model_choice = st.selectbox("Regressor", options=["XGBoost (if available)" if XGBOOST_AVAILABLE else "XGBoost (unavailable)", "RandomForest", "Ridge (linear)"])
    n_estimators = st.number_input("n_estimators (trees)", 50, 1000, value=100, step=10)

    train_button = st.button("Train model", key="train_btn")
    if train_button:
        X = df[feature_cols]
        y = df[target_col]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

        preprocessor, num_cols, cat_cols = build_preprocessor(df, feature_cols)

        if "XGBoost" in model_choice and XGBOOST_AVAILABLE:
            reg = XGBRegressor(n_estimators=int(n_estimators), verbosity=0, n_jobs=-1, random_state=42)
        elif "RandomForest" in model_choice:
            reg = RandomForestRegressor(n_estimators=int(n_estimators), n_jobs=-1, random_state=42)
        else:
            reg = Ridge()

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", reg)
        ])

        with st.spinner("Training..."):
            pipeline, stats = train_model(pipeline, X_train, y_train, X_val, y_val)

        st.success("Model trained")
        # store pipeline in session so user can predict
        st.session_state["pipeline"] = pipeline
        st.session_state["feature_cols"] = feature_cols
        st.session_state["target_col"] = target_col
        st.session_state["model_stats"] = stats

        # show metrics as cards
        cols = st.columns(3)
        cols[0].metric("Train R²", f"{stats['r2_train']:.4f}")
        cols[1].metric("Val R²", f"{stats.get('r2_val', 'N/A') if stats.get('r2_val') is not None else 'N/A'}")
        cols[2].metric("Val RMSE", f"{stats.get('rmse_val', 0):,.2f}")

        # provide pipeline download (no server write)
        buf = BytesIO()
        joblib.dump(pipeline, buf)
        buf.seek(0)
        make_download_button_bytes(buf.read(), "pipeline.joblib", "⬇️ Download pipeline (joblib)")

    # Prediction UI (single row)
    if "pipeline" in st.session_state:
        st.markdown("---")
        st.subheader("2) Single-row prediction")
        pipeline = st.session_state["pipeline"]
        feat_cols = st.session_state["feature_cols"]

        # use a sample row to provide sensible defaults
        try:
            example_row = df[feat_cols].dropna().sample(1, random_state=42).iloc[0]
        except Exception:
            example_row = None

        single_inputs = {}
        cols_input = st.columns(3)
        for i, feat in enumerate(feat_cols):
            with cols_input[i % 3]:
                if pd.api.types.is_numeric_dtype(df[feat]):
                    default = float(example_row[feat]) if example_row is not None else float(df[feat].dropna().median())
                    single_inputs[feat] = st.number_input(feat, value=default)
                else:
                    opts = df[feat].dropna().unique().tolist()
                    default_idx = 0
                    if example_row is not None:
                        try:
                            default_idx = list(opts).index(example_row[feat])
                        except Exception:
                            default_idx = 0
                    single_inputs[feat] = st.selectbox(feat, options=opts, index=default_idx)

        if st.button("Predict single row"):
            single_df = pd.DataFrame([single_inputs])
            try:
                pred = pipeline.predict(single_df)[0]
                st.success(f"Predicted {st.session_state['target_col']}: {pred:,.2f}")

                # show encoded vector (debug / transparency)
                preproc = pipeline.named_steps.get("preprocessor", None)
                if preproc is not None:
                    try:
                        encoded = preproc.transform(single_df)
                        st.caption("Encoded input vector (first 10 values shown):")
                        st.write(np.array(encoded).ravel()[:10].tolist())
                    except Exception:
                        pass
            except Exception as e:
                st.error(f"Prediction failed: {e}")

        # Batch predictions (download only)
        st.markdown("---")
        st.subheader("Batch predictions (upload CSV with same feature columns)")
        batch_upload = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch_pred")
        if batch_upload is not None:
            try:
                batch_df = pd.read_csv(batch_upload)
                missing = [c for c in feat_cols if c not in batch_df.columns]
                if missing:
                    st.error(f"Uploaded file missing required features: {missing}")
                else:
                    X_batch = batch_df[feat_cols]
                    preds = pipeline.predict(X_batch)
                    batch_df["Predicted_" + st.session_state["target_col"]] = preds
                    st.dataframe(batch_df.head(20))
                    csv_bytes = batch_df.to_csv(index=False).encode()
                    st.download_button("⬇️ Download batch predictions CSV", csv_bytes, file_name="batch_predictions.csv", mime="text/csv")
            except Exception as e:
                st.error(f"Could not process batch file: {e}")

# -------------------------
# Right: Analytics & explainers
# -------------------------
with right:
    st.subheader("Quick Analytics")
    st.markdown("Interactive charts provide fast insight into your dataset.")

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        st.markdown("**Numeric distributions**")
        for col in numeric_cols[:6]:
            fig = px.histogram(df, x=col, nbins=40, title=col, marginal="box")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns found.")

    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if cat_cols:
        st.markdown("**Categorical counts (top 10)**")
        for col in cat_cols[:6]:
            vc = df[col].value_counts().nlargest(10).reset_index()
            vc.columns = [col, "count"]
            fig = px.bar(vc, x=col, y="count", title=col)
            st.plotly_chart(fig, use_container_width=True)

    # Correlation
    if len(numeric_cols) >= 2:
        st.markdown("**Correlation matrix**")
        corr = df[numeric_cols].corr()
        fig_corr = px.imshow(corr, text_auto=True, title="Correlation")
        st.plotly_chart(fig_corr, use_container_width=True)

# -------------------------
# Footer / About
# -------------------------
st.markdown("---")
with st.container():
    c1, c2 = st.columns([3,1])
    with c1:
        st.markdown("**Notes:** This app **does not** persist saves to server. Use download buttons to export models or predictions. For persistent storage, connect S3/GDrive or a DB.")
    with c2:
        st.markdown(f'<small class="muted">Version • no-save</small>', unsafe_allow_html=True)
