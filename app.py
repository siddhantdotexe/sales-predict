# app.py - Universal Sales Predictor (categorical comparison & relationship analysis removed)
import streamlit as st
import pandas as pd
import numpy as np
import os
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
import plotly.graph_objects as go

# Optional XGBoost
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

st.set_page_config(page_title="ML Based Revenue Predictor", layout="wide", page_icon="🛒")

# -------------------------
# Helper functions
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessor(df, feature_cols):
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", make_ohe())
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, cat_cols)
    ], remainder="drop")

    return preprocessor, numeric_cols, cat_cols

def train_model(pipeline, X_train, y_train, X_val=None, y_val=None):
    pipeline.fit(X_train, y_train)
    train_preds = pipeline.predict(X_train)
    r2_tr = r2_score(y_train, train_preds)
    rmse_tr = rmse(y_train, train_preds)
    mae_tr = mean_absolute_error(y_train, train_preds)

    stats = {"r2_train": r2_tr, "rmse_train": rmse_tr, "mae_train": mae_tr}
    if X_val is not None and y_val is not None:
        val_preds = pipeline.predict(X_val)
        stats.update({
            "r2_val": r2_score(y_val, val_preds),
            "rmse_val": rmse(y_val, val_preds),
            "mae_val": mean_absolute_error(y_val, val_preds)
        })
    return pipeline, stats

def download_link_fileobj(obj_bytes, filename, mime="text/csv"):
    return st.download_button(label=f"⬇️ Download {filename}", data=obj_bytes, file_name=filename, mime=mime)

def sanitize_numeric(df, numeric_cols=None):
    df = df.copy()
    if numeric_cols is None:
        return df
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(r'[^\d\.\-eE]', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# -------------------------
# UI Layout
st.title("🛒 ML Based Revenue Predictor")
st.markdown("Upload any sales dataset (CSV), select a numeric target, train a model, and predict. Categorical comparisons & relationship analyses have been removed (per request).")

# Sidebar: Upload dataset or use sample
st.sidebar.header("Data / Model")
uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample dataset (toy)", value=False)

if use_sample:
    # generate a small sample if no file
    df = px.data.tips()  # small dataset; we'll adapt it to a 'sales' like example
    # create a synthetic sales target from tips dataset
    df = df.rename(columns={"total_bill":"Item_MRP", "tip":"Item_Outlet_Sales"})
    df["Outlet_Type"] = np.random.choice(["Supermarket Type1","Grocery Store","Supermarket Type2"], size=len(df))
    st.sidebar.success("Loaded sample dataset (tips -> adapted).")
elif uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")
        st.stop()
else:
    st.info("Upload a CSV to start or check 'Use sample dataset' in the sidebar.")
    st.stop()

st.subheader("Preview of uploaded data")
st.dataframe(df.head())

# Choose target
st.subheader("Choose target column (the column we will predict)")
target_col = st.selectbox("Select target (numeric)", options=df.columns.tolist(), index=len(df.columns)-1 if len(df.columns)>0 else 0)

# Validate target numeric
if not pd.api.types.is_numeric_dtype(df[target_col]):
    st.warning("Target column is not numeric. The app requires numeric target for regression. Please upload dataset with numeric sales column.")
    # try to coerce
    try:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        if df[target_col].isna().all():
            st.stop()
        else:
            st.info("Converted target column to numeric with coercion (NaN introduced for bad rows).")
    except Exception:
        st.stop()

# Features selection
st.subheader("Select feature columns (predictors)")
default_features = [c for c in df.columns if c != target_col]
feature_cols = st.multiselect("Features to use (choose one or more)", options=default_features, default=default_features)

if len(feature_cols) == 0:
    st.error("Pick at least one feature column.")
    st.stop()

# Option to drop rows with missing target
if st.checkbox("Drop rows with missing target values", value=True):
    df = df.dropna(subset=[target_col])

# Train/test split ratio
st.subheader("Train / Validation settings")
test_size = st.slider("Validation set fraction", min_value=0.05, max_value=0.5, value=0.2, step=0.05)

# Model selection
st.subheader("Model selection")
model_choice = st.selectbox("Choose regressor", options=["XGBoost (fast, powerful)" if XGBOOST_AVAILABLE else "XGBoost (unavailable)",
                                                         "RandomForest", "Ridge (linear)"])
if model_choice.startswith("XGBoost") and not XGBOOST_AVAILABLE:
    st.warning("XGBoost not available in this environment; pick RandomForest or Ridge.")

n_estimators = st.number_input("n_estimators (for tree models)", value=100, min_value=10, max_value=2000, step=10)

# Train button
if st.button("Train model"):
    with st.spinner("Building preprocessor and training model..."):
        # split
        X = df[feature_cols]
        y = df[target_col]
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

        # build preprocessor
        preprocessor, num_cols, cat_cols = build_preprocessor(df, feature_cols)

        # build regressor
        if "XGBoost" in model_choice and XGBOOST_AVAILABLE:
            reg = XGBRegressor(n_estimators=int(n_estimators), verbosity=0, n_jobs=-1, random_state=42)
        elif "RandomForest" in model_choice:
            reg = RandomForestRegressor(n_estimators=int(n_estimators), n_jobs=-1, random_state=42)
        else:
            reg = Ridge()

        # full pipeline
        pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", reg)])

        # train
        pipeline, stats = train_model(pipeline, X_train, y_train, X_val, y_val)

        st.success("Training complete!")
        st.metric("Train R²", f"{stats['r2_train']:.4f}")
        st.metric("Val R²", f"{stats['r2_val']:.4f}" if "r2_val" in stats else "N/A")
        st.write(f"Train RMSE: {stats['rmse_train']:.4f} — Val RMSE: {stats.get('rmse_val','N/A'):.4f}" if "rmse_val" in stats else f"Train RMSE: {stats['rmse_train']:.4f}")

        # show residual plot for validation
        if "r2_val" in stats:
            preds_val = pipeline.predict(X_val)
            resid = y_val - preds_val
            fig = px.scatter(x=preds_val, y=resid, labels={"x":"Predicted", "y":"Residual"}, title="Residuals (val set)")
            st.plotly_chart(fig, use_container_width=True)

        # store pipeline in session state
        st.session_state["pipeline"] = pipeline
        st.session_state["feature_cols"] = feature_cols
        st.session_state["target_col"] = target_col
        st.session_state["model_stats"] = stats
        st.session_state["num_cols"] = num_cols
        st.session_state["cat_cols"] = cat_cols

# If a pipeline is present, show prediction UI
if "pipeline" in st.session_state:
    pipeline = st.session_state["pipeline"]
    st.sidebar.success("Model ready ✅")
    st.subheader("Single-row prediction (use current feature names)")

    # build form for single-row
    single_vals = {}
    cols_widgets = st.columns(3)
    for i, feat in enumerate(st.session_state["feature_cols"]):
        # safe: widget based on dataset if column exists; otherwise text input
        if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
            single_vals[feat] = cols_widgets[i % 3].number_input(f"{feat}", value=float(df[feat].dropna().median()) if not df[feat].dropna().empty else 0.0, key=f"sv_{feat}")
        elif feat in df.columns:
            opts = df[feat].dropna().unique().tolist()
            if len(opts) > 0:
                single_vals[feat] = cols_widgets[i % 3].selectbox(f"{feat}", options=opts, index=0, key=f"sv_{feat}")
            else:
                single_vals[feat] = cols_widgets[i % 3].text_input(f"{feat}", value="", key=f"sv_{feat}")
        else:
            # feature missing in current df (user trained on another dataset) -> allow manual entry
            single_vals[feat] = cols_widgets[i % 3].text_input(f"{feat} (missing in dataset)", value="", key=f"sv_missing_{feat}")

    if st.button("Predict single row"):
        single_df = pd.DataFrame([single_vals])
        # sanitize numeric columns recorded during training
        num_cols_stored = st.session_state.get("num_cols", [])
        try:
            single_df = sanitize_numeric(single_df, numeric_cols=num_cols_stored if num_cols_stored else None)
        except Exception:
            pass

        try:
            pred = pipeline.predict(single_df)[0]
            st.success(f"Predicted {st.session_state['target_col']}: {pred:,.2f}")
            st.download_button("Download prediction", single_df.assign(Prediction=pred).to_csv(index=False).encode(), "single_prediction.csv", "text/csv")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Batch predictions: upload new CSV with same feature columns
    st.subheader("Batch predictions (upload CSV with same feature columns)")
    batch_file = st.file_uploader("Upload CSV for batch prediction", type=["csv"], key="batch_pred")
    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            missing = [c for c in st.session_state["feature_cols"] if c not in batch_df.columns]
            if missing:
                st.error(f"Uploaded file is missing required feature columns: {missing}")
            else:
                X_batch = batch_df[st.session_state["feature_cols"]]
                X_batch = sanitize_numeric(X_batch, numeric_cols=st.session_state.get("num_cols", []))
                preds = pipeline.predict(X_batch)
                batch_df["Predicted_" + st.session_state["target_col"]] = preds
                st.dataframe(batch_df.head(30))
                csv_bytes = batch_df.to_csv(index=False).encode()
                download_link_fileobj(csv_bytes, "batch_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Could not process batch file: {e}")

# Analytics section (numeric-only, no categorical comparisons or relationship analysis)
st.sidebar.markdown("---")
st.sidebar.subheader("Analytics")
if st.sidebar.button("Show analytics"):
    st.header("Automatic Analytics for uploaded dataset (numeric only)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Numeric distributions
    st.subheader("Numeric distributions")
    for col in numeric_cols:
        fig = px.histogram(df, x=col, nbins=50, title=f"Distribution: {col}")
        st.plotly_chart(fig, use_container_width=True)

    # correlation heatmap
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto=True, title="Correlation matrix (numeric features)")
        st.plotly_chart(fig, use_container_width=True)

    # time-series detection & plot (if date-like column present)
    date_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            date_cols.append(col)
        elif df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(200)
            parsed = 0
            for s in sample:
                try:
                    _ = pd.to_datetime(s)
                    parsed += 1
                except Exception:
                    pass
            if len(sample) > 0 and parsed / max(1, len(sample)) > 0.6:
                date_cols.append(col)

    if len(date_cols) > 0:
        st.subheader("Detected date-like columns (time series)")
        chosen_date = st.selectbox("Choose date column for time-series", options=date_cols)
        if chosen_date:
            try:
                ts = df[[chosen_date, target_col]].copy()
                ts[chosen_date] = pd.to_datetime(ts[chosen_date], errors="coerce")
                ts = ts.dropna(subset=[chosen_date, target_col])
                if not ts.empty:
                    ts_agg = ts.set_index(chosen_date).resample("W")[target_col].mean().reset_index()
                    fig_ts = px.line(ts_agg, x=chosen_date, y=target_col, title=f"Weekly avg {target_col}")
                    st.plotly_chart(fig_ts, use_container_width=True)
                else:
                    st.info("No valid date/target data for time-series aggregation.")
            except Exception as e:
                st.info(f"Could not make time-series plot: {e}")


# Save & load pipeline controls removed per request (no saving to disk in this version)
st.sidebar.markdown("---")
st.sidebar.write("This app trains in-memory and provides single/batch predictions + numeric analytics only.")
Claude
