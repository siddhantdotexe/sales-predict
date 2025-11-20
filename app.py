# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components
from collections import Counter
from sklearn.preprocessing import LabelEncoder

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
    raw_df = None

    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            st.warning(f"Failed to load model.pkl: {e}")

    if os.path.exists(x_path):
        try:
            with open(x_path, "rb") as f:
                X_train = pickle.load(f)
        except Exception as e:
            st.warning(f"Failed to load X_data.pkl: {e}")

    if os.path.exists(csv_path):
        try:
            raw_df = pd.read_csv(csv_path)
        except Exception as e:
            st.warning(f"Failed to read Train.csv: {e}")

    return model, X_train, raw_df

# Build mapping between raw categorical values and numeric encodings
def build_mappings(raw_df: pd.DataFrame, X_train: pd.DataFrame):
    """
    Returns dict: {col_name: {raw_value: encoded_value, ...}, ...}
    Two modes:
    - if X_train is provided (preferred): use row-wise relationships to infer mapping
    - else: fit LabelEncoder on raw_df[col] and produce mapping raw->label
    """
    mappings = {}
    if raw_df is None:
        return mappings

    # define categorical columns we expect from raw (based on your Train.csv)
    cat_cols = [
        "Item_Identifier", "Item_Fat_Content", "Item_Type",
        "Outlet_Identifier", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type"
    ]

    # normalize fat content variations
    if "Item_Fat_Content" in raw_df.columns:
        raw_df["Item_Fat_Content"] = raw_df["Item_Fat_Content"].replace({
            "LF": "Low Fat",
            "low fat": "Low Fat",
            "lowfat": "Low Fat",
            "reg": "Regular"
        })

    if X_train is not None:
        # prefer using X_train to infer raw->encoded relationships
        # align lengths if possible
        if len(raw_df) == len(X_train):
            for col in cat_cols:
                if col in raw_df.columns and col in X_train.columns:
                    # create a DataFrame with raw and encoded side-by-side
                    temp = pd.DataFrame({
                        "raw": raw_df[col].astype(str),
                        "enc": X_train[col]
                    })
                    # For each raw value, pick the most common encoded value
                    grp = temp.groupby("raw")["enc"].agg(lambda s: Counter(s).most_common(1)[0][0])
                    mappings[col] = grp.to_dict()
        else:
            # fallback: try to map by unique value order (less robust but may work)
            for col in cat_cols:
                if col in raw_df.columns and col in X_train.columns:
                    raw_unique = list(pd.Series(raw_df[col].astype(str).unique()))
                    enc_unique = list(pd.Series(X_train[col].unique()))
                    # if lengths mismatch, use mode-based mapping below instead
                    if len(raw_unique) == len(enc_unique):
                        mappings[col] = dict(zip(raw_unique, enc_unique))

    # For any columns still unmapped, fit a LabelEncoder on raw_df
    for col in cat_cols:
        if col in raw_df.columns and col not in mappings:
            le = LabelEncoder()
            arr = raw_df[col].astype(str).values
            le.fit(arr)
            mapping = {val: int(le.transform([val])[0]) for val in le.classes_}
            mappings[col] = mapping

    return mappings

def encode_input_single(raw_row: pd.DataFrame, mappings: dict, X_train: pd.DataFrame = None):
    """
    raw_row: single-row DataFrame with raw user inputs (string values)
    mappings: dict produced by build_mappings
    X_train: if provided, used to ensure final columns and column order
    """
    df = raw_row.copy()

    # normalize Item_Fat_Content common variants
    if "Item_Fat_Content" in df.columns:
        df["Item_Fat_Content"] = df["Item_Fat_Content"].replace({
            "LF": "Low Fat",
            "low fat": "Low Fat",
            "lowfat": "Low Fat",
            "reg": "Regular"
        })

    # Apply categorical mappings
    for col, mapping in mappings.items():
        if col in df.columns:
            raw_val = str(df.at[0, col])
            if raw_val in mapping:
                df.at[0, col] = mapping[raw_val]
            else:
                # fallback to most common mapping (mode) if unseen
                # pick the encoded value that occurs most across mapping values
                most_common_enc = Counter(mapping.values()).most_common(1)[0][0]
                df.at[0, col] = most_common_enc
        else:
            # column missing in input: use the most common encoded value as fallback
            most_common_enc = Counter(mapping.values()).most_common(1)[0][0]
            df.at[0, col] = most_common_enc

    # Ensure we have every column X_train had (if available)
    if X_train is not None:
        for col in X_train.columns:
            if col not in df.columns:
                df[col] = 0

        # reorder to match training order
        df = df[X_train.columns]
    else:
        # ensure numeric dtypes where possible
        for c in df.columns:
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                pass

    # final check: convert numeric-like columns to numeric
    for c in df.columns:
        if df[c].dtype == object:
            # try safe conversion
            try:
                df[c] = pd.to_numeric(df[c])
            except Exception:
                # leave as-is (shouldn't happen for categorical cols because we mapped them)
                pass

    return df

def save_predictions_local(df: pd.DataFrame, filename="predictions_saved.csv"):
    mode = "a" if os.path.exists(filename) else "w"
    df.to_csv(filename, mode=mode, header=not os.path.exists(filename), index=False)
    return filename

def render_lottie_from_url(url: str, height: int = 300):
    html = f"""
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    <lottie-player src="{url}" background="transparent" speed="1"
    style="width:100%; height:{height}px;" loop autoplay></lottie-player>
    """
    components.html(html, height=height + 20)

# ---------------------------
# Load assets
# ---------------------------
st.set_page_config(page_title="Big Mart Sales Prediction", page_icon="🛒", layout="wide")
logo_path = load_logo()

with st.spinner("Loading model and metadata..."):
    model, X_train, raw_df = load_model_and_data()

# Build mappings (string -> encoded integer)
mappings = build_mappings(raw_df, X_train)

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    if logo_path:
        st.image(logo_path, width=160)
    else:
        st.markdown("### 🛒 Big Mart Dashboard")

    page = st.radio("Navigation", ["Predict (Single)", "Analytics", "About & Save", "Debug"])
    st.markdown("---")

# ---------------------------
# Header
# ---------------------------
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🛒 Big Mart Sales Prediction — Dashboard")
with col2:
    try:
        render_lottie_from_url("https://assets7.lottiefiles.com/packages/lf20_jbrw3hcz.json", height=120)
    except:
        pass

# ---------------------------
# PREDICT (SINGLE)
# ---------------------------
if page == "Predict (Single)":
    st.header("🔮 Single Prediction")

    with st.form("single_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            Item_Identifier = st.text_input("Item Identifier", "FDG33")
            Item_Weight = st.number_input("Item Weight", 0.0, 200.0, 10.0)
            Item_Fat_Content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])

        with c2:
            Item_Visibility = st.number_input("Item Visibility", 0.0, 1.0, 0.05, format="%.5f")
            Item_Type = st.text_input("Item Type", "Seafood")
            Item_MRP = st.number_input("Item MRP", 0.0, 10000.0, 200.0)

        with c3:
            Outlet_Identifier = st.text_input("Outlet Identifier", "OUT027")
            Outlet_Establishment_Year = st.number_input("Outlet Year", 1900, 2030, 1990)
            Outlet_Size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
            Outlet_Location_Type = st.selectbox("Outlet Location", ["Tier 1", "Tier 2", "Tier 3"])
            Outlet_Type = st.selectbox("Outlet Type", [
                "Grocery Store", "Supermarket Type1", "Supermarket Type2", "Supermarket Type3"
            ])

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

        st.subheader("Input (Preview)")
        st.dataframe(raw.T)

        if model is None:
            st.error("Model (model.pkl) is missing or failed to load.")
        else:
            try:
                enc = encode_input_single(raw, mappings, X_train)

                # Validate dtypes are numeric
                bad_cols = [c for c in enc.columns if enc[c].dtype == object]
                if bad_cols:
                    st.error(f"Encoding failed: columns still object dtype: {bad_cols}")
                else:
                    # predict
                    pred = model.predict(enc)[0]
                    st.success(f"### 💰 Predicted Sales: {pred:,.2f}")

                    out_df = raw.copy()
                    out_df["Predicted_Sales"] = pred

                    st.download_button(
                        "Download Prediction CSV",
                        out_df.to_csv(index=False).encode(),
                        "single_prediction.csv",
                        "text/csv"
                    )

                    if st.button("Save Prediction"):
                        save_predictions_local(out_df)
                        st.success("Saved to predictions_saved.csv")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ---------------------------
# ANALYTICS
# ---------------------------
elif page == "Analytics":
    st.header("📊 Analytics")

    data_for_viz = raw_df if raw_df is not None else X_train

    if data_for_viz is None:
        st.error("No dataset found for analytics.")
    else:
        st.subheader("Numeric Distributions")
        num_cols = data_for_viz.select_dtypes(include=[np.number]).columns

        for col in num_cols:
            fig = px.histogram(data_for_viz, x=col, nbins=50, title=f"{col} Distribution")
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Categorical Counts")
        cat_cols = data_for_viz.select_dtypes(exclude=[np.number]).columns

        for col in cat_cols:
            fig = px.histogram(data_for_viz, x=col, title=f"{col} Counts")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# ABOUT & SAVE
# ---------------------------
elif page == "About & Save":
    st.header("ℹ About / Saved Predictions")

    st.write("""
    This app predicts Big Mart sales using a trained model (model.pkl).
    The app attempts to recreate the original string→numeric mappings
    by reading Train.csv (raw data) and X_data.pkl (if available).
    """)

    if os.path.exists("predictions_saved.csv"):
        df = pd.read_csv("predictions_saved.csv")
        st.success("Saved predictions found.")
        st.dataframe(df)
        st.download_button("Download saved predictions", df.to_csv(index=False).encode(), "predictions_saved.csv")
    else:
        st.info("No saved predictions available.")

# ---------------------------
# DEBUG
# ---------------------------
elif page == "Debug":
    st.header("🐞 Debug / Internals")

    st.subheader("Loaded files")
    st.write({
        "model_loaded": model is not None,
        "X_train_loaded": X_train is not None,
        "Train_csv_loaded": raw_df is not None
    })

    if X_train is not None:
        st.subheader("X_train sample")
        st.write(X_train.head())
        st.write(X_train.dtypes)
        st.write(list(X_train.columns))

    if raw_df is not None:
        st.subheader("Train.csv (raw) sample")
        st.write(raw_df.head())
        st.write(raw_df.dtypes)
        st.write(list(raw_df.columns))

    st.subheader("Mappings (sample)")
    st.write({k: (dict(list(v.items())[:10])) for k, v in mappings.items()})
