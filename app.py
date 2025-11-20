# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
import base64

# ---------------------------
# Helpers
# ---------------------------
def load_logo():
    for f in ("logo.png", "logo.jpg", "logo.jpeg"):
        if os.path.exists(f):
            return f
    return None

def load_model_and_data(model_path="model.pkl", x_path="X_data.pkl", csv_path="Train.csv"):
    model = None
    X_train = None
    full_df = None

    # load model
    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            st.error(f"Could not load model from {model_path}: {e}")
            model = None

    # load X_data.pkl if exists
    if os.path.exists(x_path):
        try:
            with open(x_path, "rb") as f:
                X_train = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load X_data.pkl: {e}")
            X_train = None

    # try to load full Train.csv (the original dataset) if present
    if os.path.exists(csv_path):
        try:
            full_df = pd.read_csv(csv_path)
        except Exception as e:
            st.warning(f"Could not read {csv_path}: {e}")
            full_df = None

    # if user saved full dataframe as pickle name
    if full_df is None:
        for p in ("big_mart_data.pkl", "bigmart_full.pkl", "big_mart.pkl"):
            if os.path.exists(p):
                try:
                    full_df = pd.read_pickle(p)
                    break
                except Exception as e:
                    st.warning(f"Could not read {p}: {e}")
                    full_df = None

    return model, X_train, full_df

def encode_input(df: pd.DataFrame, X_train: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # manual encoding consistent with training notebook
    df.replace({"Item_Fat_Content": {"Low Fat": 0, "Regular": 1}}, inplace=True)
    df.replace({"Outlet_Size": {"Small": 2, "Medium": 1, "High": 0}}, inplace=True)
    df.replace({"Outlet_Location_Type": {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}}, inplace=True)
    df.replace({"Outlet_Type": {
        "Grocery Store": 0,
        "Supermarket Type1": 1,
        "Supermarket Type2": 2,
        "Supermarket Type3": 3
    }}, inplace=True)

    # fill or map high-card columns using X_train's mode if present
    if X_train is not None:
        for col in ("Item_Identifier", "Item_Type", "Outlet_Identifier"):
            if col in X_train.columns:
                df[col] = X_train[col].mode()[0]

        # ensure order matches X_train
        for col in X_train.columns:
            if col not in df.columns:
                df[col] = 0
        df = df[X_train.columns]

    return df

def save_predictions_local(df: pd.DataFrame, filename="predictions_saved.csv"):
    mode = "a" if os.path.exists(filename) else "w"
    header = not os.path.exists(filename)
    df.to_csv(filename, mode=mode, header=header, index=False)
    return filename

def render_lottie_from_url(url: str, height: int = 300):
    html = f"""
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    <lottie-player src="{url}" background="transparent" speed="1" style="width:100%; height:{height}px;" loop autoplay></lottie-player>
    """
    components.html(html, height=height + 20)

# ---------------------------
# Load assets
# ---------------------------
st.set_page_config(page_title="Big Mart Sales Prediction", page_icon="🛒", layout="wide")
logo_path = load_logo()

with st.spinner("Loading model and metadata..."):
    model, X_train, full_df = load_model_and_data(model_path="model.pkl", x_path="X_data.pkl", csv_path="Train.csv")

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    if logo_path:
        st.image(logo_path, width=160)
    else:
        st.markdown("### 🛒 Big Mart Dashboard")
        st.write("_Add `logo.png` to the folder to brand the app_")
    page = st.radio("Navigation", ["Predict (Single)", "Batch Predictions", "Analytics", "About & Save"])
    st.markdown("---")
    st.write("Tip: upload a CSV in Batch Predictions to predict many rows at once.")

# Top header + hero animation
col1, col2 = st.columns([3,1])
with col1:
    st.title("🛒 Big Mart Sales Prediction — Enhanced Dashboard")
    st.write("Polished UI, CSV upload/download, and analytics.")
with col2:
    try:
        render_lottie_from_url("https://assets7.lottiefiles.com/packages/lf20_jbrw3hcz.json", height=120)
    except Exception:
        pass

# ---------------------------
# PREDICTION (SINGLE)
# ---------------------------
if page == "Predict (Single)":
    st.header("🔮 Single Prediction")
    with st.form("single_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            Item_Identifier = st.text_input("Item Identifier", "FDG33")
            Item_Weight = st.number_input("Item Weight", 0.0, 100.0, 15.5, step=0.1)
            Item_Fat_Content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
        with c2:
            Item_Visibility = st.number_input("Item Visibility", 0.0, 1.0, 0.05, step=0.01)
            Item_Type = st.text_input("Item Type", "Seafood")
            Item_MRP = st.number_input("Item MRP", 0.0, 1000.0, 225.5, step=0.5)
        with c3:
            Outlet_Identifier = st.text_input("Outlet Identifier", "OUT027")
            Outlet_Establishment_Year = st.number_input("Outlet Establishment Year", 1900, 2030, 1985)
            Outlet_Size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
            Outlet_Location_Type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
            Outlet_Type = st.selectbox("Outlet Type", ["Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"])

        submit = st.form_submit_button("Predict")

    if submit:
        raw = pd.DataFrame([{
            "Item_Identifier": Item_Identifier,
            "Item_Weight": Item_Weight,
            "Item_Fat_Content": Item_Fat_Content,
            "Item_Visibility": Item_Visibility,
            "Item_Type": Item_Type,
            "Item_MRP": Item_MRP,
            "Outlet_Identifier": Outlet_Identifier,
            "Outlet_Establishment_Year": Outlet_Establishment_Year,
            "Outlet_Size": Outlet_Size,
            "Outlet_Location_Type": Outlet_Location_Type,
            "Outlet_Type": Outlet_Type
        }])
        st.subheader("Input (preview)")
        st.dataframe(raw.T)

        if model is None or X_train is None:
            st.error("Model or training metadata missing. Make sure model.pkl and X_data.pkl are present.")
        else:
            try:
                enc = encode_input(raw, X_train)
                pred = model.predict(enc)[0]
                out_df = raw.copy()
                out_df["Predicted_Sales"] = pred
                st.markdown(f"### 💰 Predicted Sales: ${pred:,.2f}")
                st.download_button("Download single prediction (CSV)", out_df.to_csv(index=False).encode(), "single_prediction.csv", "text/csv")
                if st.button("Save prediction to server"):
                    saved = save_predictions_local(out_df, filename="predictions_saved.csv")
                    st.success(f"Saved to `{saved}`")
            except Exception as e:
                st.error(f"Prediction error: {e}")

# ---------------------------
# BATCH PREDICTIONS
# ---------------------------
elif page == "Batch Predictions":
    st.header("📁 Batch Predictions (CSV Upload)")
    st.markdown("Upload CSV with columns similar to training data. App will encode & predict.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write(f"Uploaded {uploaded.name} — rows: {df.shape[0]}")
            st.dataframe(df.head())

            if model is None or X_train is None:
                st.error("Model or X_data.pkl missing — cannot predict.")
            else:
                enc = encode_input(df, X_train)
                preds = model.predict(enc)
                df["Predicted_Sales"] = preds
                st.dataframe(df.head(30))

                st.download_button("Download predictions CSV", df.to_csv(index=False).encode(), "batch_predictions.csv", "text/csv")
                if st.button("Save predictions to server (append)"):
                    saved_path = save_predictions_local(df, filename="predictions_saved.csv")
                    st.success(f"Saved to {saved_path}")

                # quick distribution chart
                fig = px.histogram(df, x="Predicted_Sales", nbins=40, title="Predicted Sales Distribution")
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Could not process file: {e}")

# ---------------------------
# ANALYTICS (ENHANCED)
# ---------------------------
elif page == "Analytics":
    st.header("📊 Analytics — Dataset Visualizations (from your training data)")

    # Prefer full_df (Train.csv) if available because it contains Item_Outlet_Sales
    data_for_viz = None
    if full_df is not None:
        data_for_viz = full_df.copy()
        st.info("Using original Train.csv for analytics (contains Item_Outlet_Sales).")
    elif X_train is not None:
        data_for_viz = X_train.copy()
        st.info("Using X_data.pkl (features only) for analytics.")
    else:
        st.error("No dataset available for analytics. Add Train.csv or X_data.pkl to the folder.")
        st.stop()

    # Replace encoded integers with readable labels where possible (best-effort)
    def try_map_fat_content(series):
        # If strings already, return as-is; else map common encodings to labels
        if series.dtype == object:
            return series
        mapping = {0: "Low Fat", 1: "Regular", 2: "Other"}
        return series.map(lambda x: mapping.get(x, str(x)))

    # NUMERIC DISTRIBUTIONS (hist + KDE-like using plotly density)
    st.subheader("Numeric distributions (histogram + density)")

    # helper to show histogram + density
    def hist_with_density(df, col, title):
        fig = px.histogram(df, x=col, nbins=50, marginal="rug", title=title)
        try:
            fig.update_traces(marker=dict(line=dict(width=0)))
        except Exception:
            pass
        st.plotly_chart(fig, use_container_width=True)

    for col, title in [
        ("Item_Weight", "Item Weight Distribution"),
        ("Item_Visibility", "Item Visibility Distribution"),
        ("Item_MRP", "Item MRP Distribution")
    ]:
        if col in data_for_viz.columns:
            hist_with_density(data_for_viz, col, title)
        else:
            st.info(f"{col} not found in dataset.")

    # Item_Outlet_Sales only exists in full_df (Train.csv)
    if "Item_Outlet_Sales" in data_for_viz.columns:
        hist_with_density(data_for_viz, "Item_Outlet_Sales", "Item Outlet Sales Distribution")

    st.markdown("---")
    st.subheader("Categorical counts (countplots)")

    # Outlet_Establishment_Year
    if "Outlet_Establishment_Year" in data_for_viz.columns:
        fig_year = px.histogram(data_for_viz, x="Outlet_Establishment_Year",
                                title="Outlet Establishment Year Counts")
        st.plotly_chart(fig_year, use_container_width=True)

    # Item_Fat_Content
    if "Item_Fat_Content" in data_for_viz.columns:
        fat_series = try_map_fat_content(data_for_viz["Item_Fat_Content"])
        fat_counts = fat_series.value_counts().reset_index()
        fat_counts.columns = ["Item_Fat_Content", "count"]
        fig_fat = px.bar(fat_counts, x="Item_Fat_Content", y="count", title="Item Fat Content Counts")
        st.plotly_chart(fig_fat, use_container_width=True)

    # Outlet_Size
    if "Outlet_Size" in data_for_viz.columns:
        size_counts = data_for_viz["Outlet_Size"].value_counts().reset_index()
        size_counts.columns = ["Outlet_Size", "count"]
        fig_size = px.bar(size_counts, x="Outlet_Size", y="count", title="Outlet Size Counts")
        st.plotly_chart(fig_size, use_container_width=True)

    # Item_Type (may have many categories) — show top 20
    if "Item_Type" in data_for_viz.columns:
        st.markdown("### Top Item Types (top 20)")
        top_types = data_for_viz["Item_Type"].value_counts().nlargest(20).reset_index()
        top_types.columns = ["Item_Type", "count"]
        top_types["Item_Type"] = top_types["Item_Type"].astype(str)
        fig_type = px.bar(top_types, x="count", y="Item_Type", orientation="h", title="Top Item Types (top 20)")
        st.plotly_chart(fig_type, use_container_width=True)

    st.markdown("---")
    st.subheader("Additional analytics")

    # Boxplot of Item_Outlet_Sales by Outlet_Type (if sales exist)
    if "Item_Outlet_Sales" in data_for_viz.columns and "Outlet_Type" in data_for_viz.columns:
        try:
            df_box = data_for_viz[["Item_Outlet_Sales", "Outlet_Type"]].copy()
            df_box["Outlet_Type"] = df_box["Outlet_Type"].astype(str)
            fig_box = px.box(df_box, x="Outlet_Type", y="Item_Outlet_Sales", title="Sales by Outlet Type")
            st.plotly_chart(fig_box, use_container_width=True)
        except Exception:
            st.info("Could not render boxplot for Sales by Outlet_Type.")

    # Correlation heatmap for numeric features (small)
    numeric_cols = data_for_viz.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) >= 2:
        corr = data_for_viz[numeric_cols].corr()
        if len(numeric_cols) > 12:
            topnums = data_for_viz[numeric_cols].var().nlargest(12).index.tolist()
            corr = data_for_viz[topnums].corr()
        fig_corr = px.imshow(corr, text_auto=True, title="Correlation matrix (numeric features)")
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")
    st.subheader("Model Feature Importance")
    if model is not None and X_train is not None:
        try:
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                fi = pd.DataFrame({"feature": X_train.columns, "importance": imp})
                fi = fi.sort_values("importance", ascending=True)
                fig_fi = px.bar(fi, x="importance", y="feature", orientation="h", title="Feature Importance (model)")
                st.plotly_chart(fig_fi, use_container_width=True)
            else:
                st.info("Model does not expose `feature_importances_`.")
        except Exception as e:
            st.error(f"Could not compute feature importance: {e}")
    else:
        st.info("Model or training features missing — cannot compute feature importance.")

# ---------------------------
# ABOUT & SAVE
# ---------------------------
elif page == "About & Save":
    st.header("ℹ️ About & Saved Predictions")
    st.write("""
        This app predicts Big Mart sales using a trained XGBoost model.
        - Single predictions, batch uploads, analytics, and download/save functionality.
        - Place `logo.png` to brand the app.
    """)
    if os.path.exists("predictions_saved.csv"):
        saved = pd.read_csv("predictions_saved.csv")
        st.success("Found saved predictions (predictions_saved.csv).")
        st.dataframe(saved.tail(20))
        st.download_button("Download saved predictions", saved.to_csv(index=False).encode(), "predictions_saved.csv", "text/csv")
    else:
        st.info("No saved predictions found. Use Save buttons to create `predictions_saved.csv`.")
