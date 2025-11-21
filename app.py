# app.py — Big Mart Sales Predictor (Enhanced UI, fixed preprocessing/prediction)
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import streamlit.components.v1 as components

# ------------------------
# Asset path (developer-provided)
ASSET_URL = "/mnt/data/logs-siddhantdotexe-sales-predict-main-app.py-2025-11-20T15_47_54.591Z.txt"
# ------------------------

# ------------------------
# Enhanced UI / Theme (unchanged)
# ------------------------
st.set_page_config(page_title="ML Based Revenue Evaluator", page_icon="🛒", layout="wide")

# Modern color palette
PRIMARY = "#0f766e"
SECONDARY = "#0369a1"
ACCENT = "#f59e0b"
SUCCESS = "#10b981"
BACKGROUND = "#f8fafc"

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    * {{
        font-family: 'Inter', sans-serif;
    }}

    .main {{
        background: linear-gradient(135deg, #f8fafc 0%, #e0f2fe 100%);
    }}

    .card {{
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(15, 118, 110, 0.1);
        transition: all 0.3s ease;
    }}

    .card:hover {{
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        transform: translateY(-2px);
    }}

    .gradient-card {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {SECONDARY} 100%);
        color: white;
        border-radius: 16px;
        padding: 32px;
        box-shadow: 0 10px 25px -5px rgba(15, 118, 110, 0.3);
    }}

    .title {{
        font-size: 36px;
        font-weight: 700;
        background: linear-gradient(135deg, {PRIMARY} 0%, {SECONDARY} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }}

    .subtitle {{
        font-size: 16px;
        color: #64748b;
        font-weight: 400;
        line-height: 1.6;
    }}

    .metric-card {{
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border-left: 4px solid {PRIMARY};
    }}

    .metric-value {{
        font-size: 28px;
        font-weight: 700;
        color: {PRIMARY};
        margin: 8px 0;
    }}

    .metric-label {{
        font-size: 13px;
        color: #64748b;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    .pill {{
        display: inline-block;
        padding: 8px 16px;
        border-radius: 999px;
        background: linear-gradient(135deg, {PRIMARY}, {SECONDARY});
        color: white;
        font-weight: 600;
        font-size: 13px;
        text-decoration: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(15, 118, 110, 0.3);
    }}

    .pill:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(15, 118, 110, 0.4);
    }}

    .status-badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
    }}

    .status-success {{
        background: #d1fae5;
        color: #065f46;
    }}

    .status-error {{
        background: #fee2e2;
        color: #991b1b;
    }}

    .stButton>button {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {SECONDARY} 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 32px;
        font-weight: 600;
        font-size: 15px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px -1px rgba(15, 118, 110, 0.3);
    }}

    .stButton>button:hover {{
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(15, 118, 110, 0.4);
    }}

    .sidebar .element-container {{
        transition: all 0.3s ease;
    }}

    div[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, white 0%, #f8fafc 100%);
        border-right: 1px solid #e2e8f0;
    }}

    .info-box {{
        background: #eff6ff;
        border-left: 4px solid {SECONDARY};
        border-radius: 8px;
        color: black;
        padding: 16px;
        margin: 12px 0;
    }}

    .success-box {{
        background: #d1fae5;
        border-left: 4px solid {SUCCESS};
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
    }}

    h1, h2, h3 {{
        color: #0f172a;
    }}

    .stDataFrame {{
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }}

    .icon-text {{
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 14px;
        color: #475569;
        margin: 8px 0;
    }}
    </style>
""", unsafe_allow_html=True)


def render_lottie(url, height=100):
    html = f"""
    <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    <lottie-player src="{url}" background="transparent" speed="1" 
                   style="width:100%; height:{height}px;" loop autoplay></lottie-player>
    """
    components.html(html, height=height + 20)


# ------------------------
# Helpers (improved correctness)
# ------------------------
def load_logo():
    for f in ("logo.png", "logo.jpg", "logo.jpeg"):
        if os.path.exists(f):
            return f
    return None


def load_model_and_data(model_path="model.pkl", x_path="X_data.pkl", csv_path="Train.csv"):
    """
    Load model (pickle), X_train metadata (pickled DataFrame) and full Train.csv (if present).
    Returns (model, X_train, full_df) — any may be None with warnings shown later.
    """
    model = None
    X_train = None
    full_df = None

    if os.path.exists(model_path):
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        except Exception as e:
            # do not crash UI on bad pickle; show warning later
            st.warning(f"Could not load '{model_path}': {e}")

    if os.path.exists(x_path):
        try:
            with open(x_path, "rb") as f:
                X_train = pickle.load(f)
            # ensure it's a DataFrame
            if not isinstance(X_train, pd.DataFrame):
                st.warning(f"X_data.pkl loaded but is not a pandas DataFrame. Ignoring X_data.pkl.")
                X_train = None
        except Exception as e:
            st.warning(f"Could not load '{x_path}': {e}")
            X_train = None

    if os.path.exists(csv_path):
        try:
            full_df = pd.read_csv(csv_path)
        except Exception as e:
            st.warning(f"Could not read '{csv_path}': {e}")
            full_df = None

    # try common alternate pickles if Train.csv is not present
    if full_df is None:
        for alt in ("big_mart_data.pkl", "bigmart_full.pkl", "big_mart.pkl"):
            if os.path.exists(alt):
                try:
                    full_df = pd.read_pickle(alt)
                    break
                except Exception:
                    full_df = None

    return model, X_train, full_df


def encode_input(df: pd.DataFrame, X_train: pd.DataFrame) -> pd.DataFrame:
    """
    Robust encoding that:
    - Uses training data categories (if X_train provided) to map categorical columns to codes.
    - Does NOT overwrite user-provided values with training-mode.
    - Maps unseen categories -> -1 (distinct code).
    - Adds missing columns expected by X_train with value 0.
    - Reorders columns to match X_train.columns if provided.
    """
    df = df.copy()

    # Ensure all expected columns exist (do not overwrite)
    # Apply safe replacements for the columns that had consistent mapping in original notebook
    # but only if present in df (do not force them)
    if "Item_Fat_Content" in df.columns:
        df["Item_Fat_Content"] = df["Item_Fat_Content"].replace({"Low Fat": 0, "Regular": 1})
    if "Outlet_Size" in df.columns:
        df["Outlet_Size"] = df["Outlet_Size"].replace({"Small": 2, "Medium": 1, "High": 0})
    if "Outlet_Location_Type" in df.columns:
        df["Outlet_Location_Type"] = df["Outlet_Location_Type"].replace({"Tier 1": 0, "Tier 2": 1, "Tier 3": 2})
    if "Outlet_Type" in df.columns:
        df["Outlet_Type"] = df["Outlet_Type"].replace({
            "Grocery Store": 0,
            "Supermarket Type1": 1,
            "Supermarket Type2": 2,
            "Supermarket Type3": 3
        })

    # If we have X_train metadata, use it to produce stable mappings for high-card categorical columns
    if X_train is not None:
        # We'll handle columns that commonly existed in training
        cat_candidates = [c for c in ["Item_Identifier", "Item_Type", "Outlet_Identifier"] if c in X_train.columns]

        for col in cat_candidates:
            # Get training values (preserve order, unique)
            train_vals = X_train[col].astype(object).tolist()
            # We want unique preserving first-occurrence order
            seen = {}
            ordered_unique = []
            for v in train_vals:
                if v not in seen:
                    seen[v] = True
                    ordered_unique.append(v)

            # Build mapping: value -> code (0..n-1)
            mapping = {val: idx for idx, val in enumerate(ordered_unique)}

            # If user provided the column, map values; unseen -> -1
            if col in df.columns:
                # If df[col] is numeric already and training values are numeric, attempt direct usage
                if pd.api.types.is_numeric_dtype(df[col]) and all(pd.api.types.is_number(x) for x in ordered_unique):
                    # assume it's already encoded compatibly; attempt no-op
                    # but cast to numeric to be safe
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
                else:
                    # map strings/categories into codes; unseen -> -1
                    df[col] = df[col].map(lambda x: mapping.get(x, -1)).astype(int)
            else:
                # Column not provided by user; do not force overwrite — add default -1 (unknown)
                df[col] = -1

        # Ensure all training columns exist in df; add missing numeric columns with 0
        for col in X_train.columns:
            if col not in df.columns:
                # if training col is numeric, fill 0; else fill -1 for categorical
                if pd.api.types.is_numeric_dtype(X_train[col]):
                    df[col] = 0
                else:
                    df[col] = -1

        # Reorder columns to match X_train exactly (important for model.predict)
        df = df.reindex(columns=X_train.columns, fill_value=0)
    else:
        # No X_train available: do a minimal safe encoding for candidate categorical columns
        from sklearn.preprocessing import LabelEncoder
        cat_candidates = [c for c in ["Item_Identifier", "Item_Type", "Outlet_Identifier"] if c in df.columns]
        for col in cat_candidates:
            if df[col].dtype == object:
                try:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                except Exception:
                    # fallback to mapping by unique
                    uniques = df[col].astype(str).unique().tolist()
                    mapping = {v: i for i, v in enumerate(uniques)}
                    df[col] = df[col].map(lambda x: mapping.get(str(x), -1)).astype(int)

    # Final safety: ensure all columns are numeric to avoid unpicklable dtypes in model
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

    return df


def to_download_bytes(df: pd.DataFrame):
    return df.to_csv(index=False).encode()


# ------------------------
# Load assets (unchanged)
# ------------------------
logo_path = load_logo()
with st.spinner("🔄 Loading model & data..."):
    model, X_train, full_df = load_model_and_data()


# ------------------------
# Sidebar (unchanged)
# ------------------------
with st.sidebar:
    st.markdown('<div style="text-align: center; margin-bottom: 24px;">', unsafe_allow_html=True)
    if logo_path:
        st.image(logo_path, width=120)
    else:
        st.markdown("### 🛒 Sales")
        st.markdown('<p style="color: #64748b; font-size: 13px;">Sales Intelligence Platform</p>',
                    unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**📍 Navigation**")
    page = st.radio("", ["🔮 Single Prediction", "📊 Analytics", "ℹ️ About"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("**🔧 System Status**")

    model_status = "✅ Loaded" if model is not None else "❌ Missing"
    x_status = "✅ Loaded" if X_train is not None else "❌ Missing"
    data_status = "✅ Loaded" if full_df is not None else "❌ Missing"

    st.markdown(f'<div class="icon-text">🤖 Model: <strong>{model_status}</strong></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="icon-text">📋 Metadata: <strong>{x_status}</strong></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="icon-text">📊 Dataset: <strong>{data_status}</strong></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        f'<div style="text-align: center;"><a class="pill" href="{ASSET_URL}" target="_blank">📄 View Asset</a></div>',
        unsafe_allow_html=True)


# ------------------------
# Header (unchanged)
# ------------------------
col1, col2 = st.columns([4, 1])
with col1:
    st.markdown('<div class="gradient-card">', unsafe_allow_html=True)
    st.markdown('<h1 style="color: white; font-size: 42px; margin-bottom: 12px;">🛒 ML Based Revenue Evaluator</h1>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color: rgba(255,255,255,0.9); font-size: 16px; margin: 0;">Leverage AI-powered predictions to forecast sales performance with precision and confidence.</p>',
        unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    try:
        render_lottie("https://assets7.lottiefiles.com/packages/lf20_jbrw3hcz.json", height=140)
    except Exception:
        pass

st.markdown("<br>", unsafe_allow_html=True)


# ------------------------
# SINGLE PREDICTION (unchanged UI, improved backend)
# ------------------------
if "🔮 Single Prediction" in page:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🎯 Make a Prediction")
    st.markdown('<p class="subtitle">Enter item and outlet details to generate sales forecast</p>',
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    with st.form("single_form", clear_on_submit=False):
        st.markdown("#### 📦 Item Information")
        c1, c2, c3 = st.columns(3)
        with c1:
            Item_Identifier = st.text_input("Item Identifier", value="FDG33", help="Unique product ID")
            Item_Weight = st.number_input("Item Weight (kg)", min_value=0.0, max_value=100.0, value=15.5, step=0.1)
        with c2:
            Item_Fat_Content = st.selectbox("Fat Content", options=["Low Fat", "Regular"])
            Item_Visibility = st.number_input("Visibility", min_value=0.0, max_value=1.0, value=0.05, step=0.01,
                                              help="Display visibility metric")
        with c3:
            Item_Type = st.text_input("Item Type", value="Seafood")
            Item_MRP = st.number_input("MRP ($)", min_value=0.0, max_value=10000.0, value=225.5, step=0.5)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🏪 Outlet Information")
        c4, c5, c6 = st.columns(3)
        with c4:
            Outlet_Identifier = st.text_input("Outlet Identifier", value="OUT027")
            Outlet_Establishment_Year = st.number_input("Establishment Year", min_value=1900, max_value=2030,
                                                        value=1985)
        with c5:
            Outlet_Size = st.selectbox("Outlet Size", options=["Small", "Medium", "High"])
            Outlet_Location_Type = st.selectbox("Location Type", options=["Tier 1", "Tier 2", "Tier 3"])
        with c6:
            Outlet_Type = st.selectbox("Outlet Type",
                                       options=["Grocery Store", "Supermarket Type1", "Supermarket Type2",
                                                "Supermarket Type3"])

        st.markdown("<br>", unsafe_allow_html=True)
        submitted = st.form_submit_button("🚀 Generate Prediction", use_container_width=True)

    if submitted:
        st.markdown("<br>", unsafe_allow_html=True)

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

        if model is None or X_train is None:
            st.error("⚠️ Model or metadata missing. Ensure `model.pkl` and `X_data.pkl` are available.")
        else:
            try:
                enc = encode_input(raw, X_train)

                # Check shape consistency before predicting
                if enc.shape[1] != X_train.shape[1]:
                    st.warning(f"Encoded input has {enc.shape[1]} cols but model expects {X_train.shape[1]}. Attempting to align...")
                    enc = enc.reindex(columns=X_train.columns, fill_value=0)

                pred = model.predict(enc)[0]

                # Success display
                col_a, col_b, col_c = st.columns([1, 2, 1])
                with col_b:
                    st.markdown('<div class="success-box">', unsafe_allow_html=True)
                    st.markdown(f'<div style="text-align: center;">', unsafe_allow_html=True)
                    st.markdown(f'<p class="metric-label">Predicted Sales</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="metric-value" style="font-size: 48px; margin: 16px 0;">${pred:,.2f}</p>',
                                unsafe_allow_html=True)
                    st.markdown('</div></div>', unsafe_allow_html=True)

                # Download section
                st.markdown("<br>", unsafe_allow_html=True)
                out_df = raw.copy()
                out_df["Predicted_Sales"] = pred
                csv_bytes = out_df.to_csv(index=False).encode()

                col_x, col_y, col_z = st.columns([1, 2, 1])
                with col_y:
                    st.download_button("📥 Download Prediction Report", csv_bytes,
                                       file_name="single_prediction.csv", mime="text/csv",
                                       use_container_width=True)

                # Input preview
                with st.expander("📋 View Input Details"):
                    st.dataframe(raw.T, use_container_width=True)

            except Exception as e:
                st.error(f"❌ Prediction error: {e}")


# ------------------------
# ANALYTICS (unchanged UI, same visuals)
# ------------------------
elif "📊 Analytics" in page:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📈 Data Analytics & Insights")
    st.markdown('<p class="subtitle">Explore patterns and trends in your sales data</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    data_for_viz = full_df if full_df is not None else X_train

    if data_for_viz is None:
        st.error("⚠️ No dataset available. Place Train.csv or X_data.pkl in the app folder.")
    else:
        # Summary metrics
        if "Item_Outlet_Sales" in data_for_viz.columns:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-label">Total Records</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">{len(data_for_viz):,}</p>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-label">Avg Sales</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">${data_for_viz["Item_Outlet_Sales"].mean():,.0f}</p>',
                            unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-label">Max Sales</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">${data_for_viz["Item_Outlet_Sales"].max():,.0f}</p>',
                            unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col4:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-label">Min Sales</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="metric-value">${data_for_viz["Item_Outlet_Sales"].min():,.0f}</p>',
                            unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

        # Distributions
        st.markdown("#### 📊 Numeric Distributions")
        for col in ["Item_Weight", "Item_Visibility", "Item_MRP", "Item_Outlet_Sales"]:
            if col in data_for_viz.columns:
                fig = px.histogram(data_for_viz, x=col, nbins=50,
                                   color_discrete_sequence=[PRIMARY])
                fig.update_layout(
                    title=dict(
                        text=f"{col} Distribution",
                        font=dict(size=18, color='#0f172a', family="Inter")
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(family="Inter", size=12, color='#0f172a'),
                    xaxis=dict(
                        title=dict(text=col, font=dict(size=14, color='#0f172a')),
                        showgrid=True,
                        gridcolor='#e5e7eb',
                        gridwidth=1,
                        tickfont=dict(size=11, color='#475569')
                    ),
                    yaxis=dict(
                        title=dict(text='Count', font=dict(size=14, color='#0f172a')),
                        showgrid=True,
                        gridcolor='#e5e7eb',
                        gridwidth=1,
                        tickfont=dict(size=11, color='#475569')
                    ),
                    hovermode='x unified',
                    margin=dict(l=60, r=40, t=60, b=60)
                )
                fig.update_traces(hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>')
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📈 Categorical Analysis")

        for col in ["Outlet_Establishment_Year", "Item_Fat_Content", "Item_Type", "Outlet_Size"]:
            if col in data_for_viz.columns:
                if col == "Item_Type":
                    top = data_for_viz[col].value_counts().nlargest(20).reset_index()
                    top.columns = ["Item_Type", "count"]
                    fig = px.bar(top, x="count", y="Item_Type", orientation="h",
                                 color_discrete_sequence=[SECONDARY])
                    fig.update_layout(
                        title=dict(
                            text=f"Top 20 {col}",
                            font=dict(size=18, color='#0f172a', family="Inter")
                        ),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(family="Inter", size=12, color='#0f172a'),
                        xaxis=dict(
                            title=dict(text='Count', font=dict(size=14, color='#0f172a')),
                            showgrid=True,
                            gridcolor='#e5e7eb',
                            gridwidth=1,
                            tickfont=dict(size=11, color='#475569')
                        ),
                        yaxis=dict(
                            title=dict(text=col, font=dict(size=14, color='#0f172a')),
                            showgrid=True,
                            gridcolor='#e5e7eb',
                            gridwidth=1,
                            tickfont=dict(size=11, color='#475569')
                        ),
                        hovermode='closest',
                        margin=dict(l=120, r=40, t=60, b=60)
                    )
                    fig.update_traces(hovertemplate='<b>%{y}</b><br>Count: %{x}<extra></extra>')
                else:
                    fig = px.histogram(data_for_viz, x=col,
                                       color_discrete_sequence=[PRIMARY])
                    fig.update_layout(
                        title=dict(
                            text=f"{col} Distribution",
                            font=dict(size=18, color='#0f172a', family="Inter")
                        ),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(family="Inter", size=12, color='#0f172a'),
                        xaxis=dict(
                            title=dict(text=col, font=dict(size=14, color='#0f172a')),
                            showgrid=True,
                            gridcolor='#e5e7eb',
                            gridwidth=1,
                            tickfont=dict(size=11, color='#475569')
                        ),
                        yaxis=dict(
                            title=dict(text='Count', font=dict(size=14, color='#0f172a')),
                            showgrid=True,
                            gridcolor='#e5e7eb',
                            gridwidth=1,
                            tickfont=dict(size=11, color='#475569')
                        ),
                        hovermode='x unified',
                        margin=dict(l=60, r=40, t=60, b=60)
                    )
                    fig.update_traces(hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>')

                st.plotly_chart(fig, use_container_width=True)

        # Feature importance
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🎯 Model Feature Importance")
        if model is not None and X_train is not None:
            try:
                if hasattr(model, "feature_importances_"):
                    imp = model.feature_importances_
                    fi = pd.DataFrame({"feature": X_train.columns, "importance": imp})
                    fi = fi.sort_values("importance", ascending=True).tail(20)

                    fig_fi = px.bar(fi, x="importance", y="feature", orientation="h",
                                    color="importance",
                                    color_continuous_scale=[[0, PRIMARY], [1, SECONDARY]])
                    fig_fi.update_layout(
                        title=dict(
                            text="Top 20 Most Important Features",
                            font=dict(size=18, color='#0f172a', family="Inter")
                        ),
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(family="Inter", size=12, color='#0f172a'),
                        showlegend=False,
                        xaxis=dict(
                            title=dict(text='Importance', font=dict(size=14, color='#0f172a')),
                            showgrid=True,
                            gridcolor='#e5e7eb',
                            gridwidth=1,
                            tickfont=dict(size=11, color='#475569')
                        ),
                        yaxis=dict(
                            title=dict(text='Feature', font=dict(size=14, color='#0f172a')),
                            showgrid=True,
                            gridcolor='#e5e7eb',
                            gridwidth=1,
                            tickfont=dict(size=11, color='#475569')
                        ),
                        hovermode='closest',
                        margin=dict(l=120, r=40, t=60, b=60)
                    )
                    fig_fi.update_traces(hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>')
                    st.plotly_chart(fig_fi, use_container_width=True)
                else:
                    st.info("ℹ️ Model does not expose feature importances.")
            except Exception as e:
                st.error(f"❌ Could not compute feature importance: {e}")


# ------------------------
# ABOUT (unchanged)
# ------------------------
elif "ℹ️ About" in page:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ℹ️ About This Application")

    st.markdown("""
    <div class="info-box">
    <h4>🎯 Purpose</h4>
    <p>This Predictor leverages advanced machine learning (XGBoost) to forecast 
    product sales with high accuracy, helping retailers make data-driven inventory and pricing decisions.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <h4>🔒 Privacy & Security</h4>
    <p>No server-side data storage is performed. All predictions are generated locally and can be 
    downloaded for your records. Your data remains completely private.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box">
    <h4>📋 Requirements</h4>
    <p>For full functionality, ensure these files are in your app directory:</p>
    <ul>
        <li><code>model.pkl</code> - Trained XGBoost model</li>
        <li><code>X_data.pkl</code> - Feature metadata</li>
        <li><code>Train.csv</code> - Historical sales data (optional, for analytics)</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # System info
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 📁 File Status")

        files = {
            "model.pkl": os.path.exists("model.pkl"),
            "X_data.pkl": os.path.exists("X_data.pkl"),
            "Train.csv": os.path.exists("Train.csv")
        }

        for file, exists in files.items():
            status = "✅ Present" if exists else "❌ Missing"
            badge_class = "status-success" if exists else "status-error"
            st.markdown(
                f'<div class="icon-text">{file}: <span class="status-badge {badge_class}">{status}</span></div>',
                unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🛠️ Technical Stack")
        st.markdown("""
        - **ML Framework**: XGBoost
        - **UI Framework**: Streamlit
        - **Visualization**: Plotly
        - **Data Processing**: Pandas, NumPy
        """)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f'<div style="text-align: center;"><a class="pill" href="{ASSET_URL}" target="_blank">📄 View Application Asset</a></div>',
        unsafe_allow_html=True)
