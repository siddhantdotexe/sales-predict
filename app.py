# app.py - Universal Sales Predictor (UPDATED: avoid CV deadlock / nested parallelism)
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px

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


def build_preprocessor(X_data, feature_cols):
    """Build preprocessor based on actual data to be transformed"""
    numeric_cols = [c for c in feature_cols if c in X_data.columns and pd.api.types.is_numeric_dtype(X_data[c])]
    cat_cols = [c for c in feature_cols if c in X_data.columns and not pd.api.types.is_numeric_dtype(X_data[c])]

    transformers = []
    
    if numeric_cols:
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])
        transformers.append(("num", numeric_transformer, numeric_cols))
    
    if cat_cols:
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", make_ohe())
        ])
        transformers.append(("cat", categorical_transformer, cat_cols))

    if not transformers:
        st.error("No valid feature columns found!")
        st.stop()

    preprocessor = ColumnTransformer(transformers, remainder="drop")
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
            "mae_val": mean_absolute_error(y_val, val_preds),
            "val_preds": val_preds
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
st.markdown("Upload any sales dataset (CSV), select a numeric target, train a model with proper baseline comparison.")

# Sidebar: Upload dataset or use sample
st.sidebar.header("Data / Model")
uploaded_file = st.sidebar.file_uploader("Upload CSV dataset", type=["csv"])
use_sample = st.sidebar.checkbox("Use sample dataset (toy)", value=False)

if use_sample:
    df = px.data.tips()
    df = df.rename(columns={"total_bill": "Item_MRP", "tip": "Item_Outlet_Sales"})
    df["Outlet_Type"] = np.random.choice(["Supermarket Type1", "Grocery Store", "Supermarket Type2"], size=len(df))
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
target_col = st.selectbox("Select target (numeric)", options=df.columns.tolist(), 
                          index=len(df.columns) - 1 if len(df.columns) > 0 else 0)

# Validate target numeric
if not pd.api.types.is_numeric_dtype(df[target_col]):
    st.warning("Target column is not numeric. Attempting to convert...")
    try:
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
        if df[target_col].isna().all():
            st.error("Could not convert target to numeric. Please check your data.")
            st.stop()
        else:
            st.info("Converted target column to numeric with coercion (NaN introduced for bad rows).")
    except Exception:
        st.error("Failed to convert target column.")
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

if len(df) == 0:
    st.error("No data left after dropping missing target values!")
    st.stop()

# Train/test split ratio
st.subheader("Train / Validation settings")
test_size = st.slider("Validation set fraction", min_value=0.05, max_value=0.5, value=0.2, step=0.05)

# Model selection
st.subheader("Model selection")
model_options = ["RandomForest", "Ridge (linear)"]
if XGBOOST_AVAILABLE:
    model_options.insert(0, "XGBoost (fast, powerful)")
else:
    model_options.insert(0, "XGBoost (unavailable)")

model_choice = st.selectbox("Choose regressor", options=model_options)

if model_choice.startswith("XGBoost") and not XGBOOST_AVAILABLE:
    st.warning("XGBoost not available in this environment; pick RandomForest or Ridge.")

n_estimators = st.number_input("n_estimators (for tree models)", value=100, min_value=10, max_value=2000, step=10)
use_multiple_cores_for_final_fit = st.checkbox("Use multiple cores for final fit (if available)", value=True)

# ---------- Train button block ----------
if st.button("Train model"):
    with st.spinner("Building preprocessor and training model..."):
        try:
            # Split data FIRST
            X = df[feature_cols].copy()
            y = df[target_col].copy()
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

            st.info(f"Training on {len(X_train)} samples, validating on {len(X_val)} samples")

            # Build preprocessor based on TRAINING data only (prevent data leakage)
            preprocessor, num_cols, cat_cols = build_preprocessor(X_train, feature_cols)

            # Build regressor (final regressor - may use multiple cores for final fit)
            if "XGBoost" in model_choice and XGBOOST_AVAILABLE:
                final_n_jobs = -1 if use_multiple_cores_for_final_fit else 1
                reg_final = XGBRegressor(n_estimators=int(n_estimators), verbosity=0, n_jobs=final_n_jobs, random_state=42)
            elif "RandomForest" in model_choice:
                final_n_jobs = -1 if use_multiple_cores_for_final_fit else 1
                reg_final = RandomForestRegressor(n_estimators=int(n_estimators), n_jobs=final_n_jobs, random_state=42)
            else:
                reg_final = Ridge()

            # Full pipeline for actual final model
            pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", reg_final)])

            # ---- 1) Cross-validation comparison with proper baseline ----
            st.subheader("📊 Cross-validation (5-fold) comparison")
            cv = KFold(n_splits=5, shuffle=True, random_state=42)

            # Create baseline pipeline with same preprocessing
            baseline_pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("regressor", DummyRegressor(strategy="mean"))
            ])

            try:
                with st.spinner("Running cross-validation (single-threaded to avoid nested-parallel issues)..."):
                    # NOTE: run CV single-threaded to avoid nested parallelism deadlocks
                    cv_n_jobs = 1

                    # Baseline CV scores
                    baseline_rmse_cv = -cross_val_score(
                        baseline_pipeline, X_train, y_train,
                        scoring="neg_root_mean_squared_error", cv=cv, n_jobs=cv_n_jobs
                    )
                    baseline_r2_cv = cross_val_score(
                        baseline_pipeline, X_train, y_train,
                        scoring="r2", cv=cv, n_jobs=cv_n_jobs
                    )

                    # For CV make sure tree regressors use n_jobs=1 to avoid nested parallelism
                    if isinstance(reg_final, RandomForestRegressor):
                        reg_cv = RandomForestRegressor(n_estimators=int(n_estimators), n_jobs=1, random_state=42)
                    elif XGBOOST_AVAILABLE and isinstance(reg_final, XGBRegressor):
                        # create XGB for CV with single thread
                        reg_cv = XGBRegressor(n_estimators=int(n_estimators), verbosity=0, n_jobs=1, random_state=42)
                    elif isinstance(reg_final, Ridge):
                        reg_cv = Ridge()
                    else:
                        reg_cv = reg_final

                    pipeline_cv = Pipeline([("preprocessor", preprocessor), ("regressor", reg_cv)])

                    model_rmse_cv = -cross_val_score(
                        pipeline_cv, X_train, y_train,
                        scoring="neg_root_mean_squared_error", cv=cv, n_jobs=cv_n_jobs
                    )
                    model_r2_cv = cross_val_score(
                        pipeline_cv, X_train, y_train,
                        scoring="r2", cv=cv, n_jobs=cv_n_jobs
                    )

                    # Display comparison
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Baseline (Dummy Mean Predictor)**")
                        st.metric("RMSE", f"{baseline_rmse_cv.mean():.3f} ± {baseline_rmse_cv.std():.3f}")
                        st.metric("R²", f"{baseline_r2_cv.mean():.3f} ± {baseline_r2_cv.std():.3f}")
                    
                    with col2:
                        st.markdown(f"**Your Model ({model_choice})**")
                        st.metric("RMSE", f"{model_rmse_cv.mean():.3f} ± {model_rmse_cv.std():.3f}",
                                 delta=f"{baseline_rmse_cv.mean() - model_rmse_cv.mean():.3f}",
                                 delta_color="normal")
                        st.metric("R²", f"{model_r2_cv.mean():.3f} ± {model_r2_cv.std():.3f}",
                                 delta=f"{model_r2_cv.mean() - baseline_r2_cv.mean():.3f}",
                                 delta_color="normal")

                    # Interpretation
                    if model_r2_cv.mean() > baseline_r2_cv.mean() + 0.01:
                        st.success("✅ Your model significantly outperforms the baseline!")
                    elif model_r2_cv.mean() > baseline_r2_cv.mean():
                        st.info("ℹ️ Your model slightly outperforms the baseline.")
                    else:
                        st.warning("⚠️ Your model is not better than baseline. Consider feature engineering or different approach.")

            except Exception as e:
                st.warning(f"CV comparison could not be completed: {e}")

            # ---- 2) Final train on full training set, evaluate on validation ----
            st.subheader("📈 Final model training")
            pipeline, stats = train_model(pipeline, X_train, y_train, X_val, y_val)

            st.success("Training complete!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Train R²", f"{stats['r2_train']:.4f}")
            col2.metric("Val R²", f"{stats['r2_val']:.4f}")
            col3.metric("Val RMSE", f"{stats['rmse_val']:.4f}")

            # Show residual plot for validation
            if "val_preds" in stats:
                try:
                    preds_val = stats["val_preds"]
                    resid = y_val.values - preds_val
                    
                    fig = px.scatter(x=preds_val, y=resid, 
                                   labels={"x": "Predicted", "y": "Residual"}, 
                                   title="Residual Plot (Validation Set)")
                    fig.add_hline(y=0, line_dash="dash", line_color="red")
                    st.plotly_chart(fig, use_container_width=True)

                    resid_mean = np.mean(resid)
                    resid_std = np.std(resid, ddof=1)
                    st.write(f"Validation residuals: mean={resid_mean:.3f}, std={resid_std:.3f}")
                except Exception as e:
                    st.info(f"Could not compute residuals: {e}")

            # Store pipeline and metadata in session state
            st.session_state["pipeline"] = pipeline
            st.session_state["feature_cols"] = feature_cols
            st.session_state["target_col"] = target_col
            st.session_state["model_stats"] = stats
            st.session_state["num_cols"] = num_cols
            st.session_state["cat_cols"] = cat_cols

        except Exception as e:
            st.error(f"Training failed: {e}")
            import traceback
            st.code(traceback.format_exc())
# ---------------------------------------------------------------------------


# If a pipeline is present, show prediction UI
if "pipeline" in st.session_state:
    pipeline = st.session_state["pipeline"]
    st.sidebar.success("Model ready ✅")
    
    st.markdown("---")
    st.subheader("🔮 Make Prediction")
    st.markdown("Enter values for each feature:")
    
    # Build form for single-row
    single_vals = {}
    cols_widgets = st.columns(3)
    for i, feat in enumerate(st.session_state["feature_cols"]):
        if feat in df.columns and pd.api.types.is_numeric_dtype(df[feat]):
            median_val = float(df[feat].dropna().median()) if not df[feat].dropna().empty else 0.0
            single_vals[feat] = cols_widgets[i % 3].number_input(
                f"{feat}",
                value=median_val,
                key=f"sv_{feat}"
            )
        elif feat in df.columns:
            opts = df[feat].dropna().unique().tolist()
            if len(opts) > 0:
                single_vals[feat] = cols_widgets[i % 3].selectbox(
                    f"{feat}", 
                    options=opts, 
                    index=0, 
                    key=f"sv_{feat}"
                )
            else:
                single_vals[feat] = cols_widgets[i % 3].text_input(
                    f"{feat}", 
                    value="", 
                    key=f"sv_{feat}"
                )
        else:
            single_vals[feat] = cols_widgets[i % 3].text_input(
                f"{feat} (not in training data)", 
                value="", 
                key=f"sv_missing_{feat}"
            )

    if st.button("Predict"):
        try:
            single_df = pd.DataFrame([single_vals])
            num_cols_stored = st.session_state.get("num_cols", [])
            single_df = sanitize_numeric(single_df, numeric_cols=num_cols_stored)
            
            pred = pipeline.predict(single_df)[0]
            st.success(f"**Predicted {st.session_state['target_col']}: {pred:,.2f}**")
            
            # Download option
            result_df = single_df.copy()
            result_df["Prediction"] = pred
            csv_data = result_df.to_csv(index=False).encode()
            st.download_button(
                "Download prediction", 
                csv_data, 
                "single_prediction.csv", 
                "text/csv"
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")


# Analytics section
st.sidebar.markdown("---")
st.sidebar.subheader("Analytics")
if st.sidebar.button("Show analytics"):
    st.markdown("---")
    st.header("📊 Automatic Analytics")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Numeric distributions
    if numeric_cols:
        st.subheader("Numeric Feature Distributions")
        for col in numeric_cols:
            fig = px.histogram(df, x=col, nbins=50, title=f"Distribution: {col}")
            st.plotly_chart(fig, use_container_width=True)

        # Correlation heatmap
        if len(numeric_cols) >= 2:
            st.subheader("Correlation Matrix")
            corr = df[numeric_cols].corr()
            fig = px.imshow(corr, text_auto=".2f", title="Correlation matrix (numeric features)",
                          color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric columns found for distribution analysis.")

    # Time-series detection & plot
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
        st.subheader("Time Series Analysis")
        chosen_date = st.selectbox("Choose date column for time-series", options=date_cols)
        if chosen_date:
            try:
                ts = df[[chosen_date, target_col]].copy()
                ts[chosen_date] = pd.to_datetime(ts[chosen_date], errors="coerce")
                ts = ts.dropna(subset=[chosen_date, target_col])
                if not ts.empty:
                    ts_agg = ts.set_index(chosen_date).resample("W")[target_col].mean().reset_index()
                    fig_ts = px.line(ts_agg, x=chosen_date, y=target_col, 
                                    title=f"Weekly average {target_col}")
                    st.plotly_chart(fig_ts, use_container_width=True)
                else:
                    st.info("No valid date/target data for time-series aggregation.")
            except Exception as e:
                st.info(f"Could not make time-series plot: {e}")

st.sidebar.markdown("---")
st.sidebar.write("✨ This app provides ML-based predictions with proper baseline comparison and analytics.")
