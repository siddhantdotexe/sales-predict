# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

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

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model = pickle.load(f)

    if os.path.exists(x_path):
        with open(x_path, "rb") as f:
            X_train = pickle.load(f)

    if os.path.exists(csv_path):
        try:
            full_df = pd.read_csv(csv_path)
        except Exception:
            full_df = None

    if full_df is None:
        for p in ("big_mart_data.pkl", "bigmart_full.pkl", "big_mart.pkl"):
            if os.path.exists(p):
                try:
                    full_df = pd.read_pickle(p)
                    break
                except Exception:
                    full_df = None

    return model, X_train, full_df


def encode_input(df: pd.DataFrame, X_train: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.replace({"Item_Fat_Content": {"Low Fat": 0, "Regular": 1}}, inplace=True)
    df.replace({"Outlet_Size": {"Small": 2, "Medium": 1, "High": 0}}, inplace=True)
    df.replace({"Outlet_Location_Type": {"Tier 1": 0, "Tier 2": 1, "Tier 3": 2}}, inplace=True)
    df.replace({"Outlet_Type": {
        "Grocery Store": 0,
        "Supermarket Type1": 1,
        "Supermarket Type2": 2,
        "Supermarket Type3": 3
    }}, inplace=True)

    if X_train is not None:
        for col in ("Item_Identifier", "Item_Type", "Outlet_Identifier"):
            if col in X_train.columns:
                df[col] = X_train[col].mode()[0]

        for col in X_train.columns:
            if col not in df.columns:
                df[col] = 0

        df = df[X_train.columns]

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
    model, X_train, full_df = load_model_and_data()


# ---------------------------
# Sidebar (Batch Removed)
# ---------------------------
with st.sidebar:
    if logo_path:
        st.image(logo_path, width=160)
    else:
        st.markdown("### 🛒 Big Mart Dashboard")

    page = st.radio("Navigation", ["Predict (Single)", "Analytics", "About & Save"])
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
            Item_Visibility = st.number_input("Item Visibility", 0.0, 1.0, 0.05)
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

        if model is None or X_train is None:
            st.error("Model or X_data.pkl missing.")
        else:
            try:
                enc = encode_input(raw, X_train)
                pred = model.predict(enc)[0]

                st.success(f"### 💰 Predicted Sales: ${pred:,.2f}")

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

    data_for_viz = full_df if full_df is not None else X_train

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
    st.header("ℹ️ About / Saved Predictions")

    st.write("""
    This app predicts Big Mart sales using a trained XGBoost model.
    Only **Single Prediction** and **Analytics** modes are enabled.
    """)

    if os.path.exists("predictions_saved.csv"):
        df = pd.read_csv("predictions_saved.csv")
        st.success("Saved predictions found.")
        st.dataframe(df)
        st.download_button("Download saved predictions", df.to_csv(index=False).encode(), "predictions_saved.csv")
    else:
        st.info("No saved predictions available.")
