# app.py — Big Mart Sales Predictor (final: dropdowns, upgraded metric cards, asset links removed, no texture)
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
# Enhanced UI / Theme
# ------------------------
st.set_page_config(page_title="ML Based Revenue Evaluator", page_icon="🛒", layout="wide")

PRIMARY = "#0f766e"
SECONDARY = "#0369a1"
ACCENT = "#f59e0b"
SUCCESS = "#10b981"
BACKGROUND = "#0b1220"  # dark background to make cards pop (change if you prefer light)

st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    * {{ font-family: 'Inter', sans-serif; }}
    body {{ background: {BACKGROUND}; }}

    .card {{
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 14px rgba(2,6,23,0.45);
        border: 1px solid rgba(255,255,255,0.02);
        transition: all 0.25s ease;
    }}

    .gradient-card {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {SECONDARY} 100%);
        color: white;
        border-radius: 16px;
        padding: 32px;
        box-shadow: 0 10px 25px -5px rgba(15, 118, 110, 0.3);
    }}

    .subtitle {{
        font-size: 16px;
        color: #94a3b8;
        font-weight: 400;
        line-height: 1.6;
    }}

    /* Upgraded metric card style (clean glass look, no texture) */
    .metric-card {{
        position: relative;
        overflow: hidden;
        border-radius: 14px;
        padding: 18px 20px;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        gap: 6px;
        box-shadow: 0 6px 18px rgba(2,6,23,0.45);
        border: 1px solid rgba(255,255,255,0.03);
        color: white;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
        background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.005));
        backdrop-filter: blur(6px);
        word-wrap: break-word;
    }}

    .metric-card:hover {{
        transform: translateY(-6px);
        box-shadow: 0 18px 35px rgba(2,6,23,0.6);
    }}

    .metric-card::before {{
        content: "";
        position: absolute;
        left: 0;
        top: 0;
        width: 6px;
        height: 100%;
        background: linear-gradient(180deg, {PRIMARY}, {SECONDARY});
        border-top-left-radius: 14px;
        border-bottom-left-radius: 14px;
    }}

    .metric-label {{
        font-size: 13px;
        color: rgba(255,255,255,0.95);
        font-weight: 700;
        letter-spacing: 0.6px;
        text-transform: uppercase;
    }}

    .metric-value {{
        font-size: 32px;
        font-weight: 800;
        color: white;
        margin-top: 6px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}

    .metric-sub {{
        font-size: 12px;
        color: rgba(255,255,255,0.85);
    }}

    .metric-icon {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 42px;
        height: 42px;
        border-radius: 10px;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.03);
        font-weight: 700;
        color: white;
        font-size: 16px;
    }}

    @media (max-width: 900px) {{
        .metric-value {{ font-size: 22px; }}
    }}

    .stDataFrame, .css-1v0mbdj.e1tzin5v1 {{ border-radius: 12px; overflow: hidden; }}
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
# Helpers
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
            st.warning(f"Could not load '{model_path}': {e}")

    if os.path.exists(x_path):
        try:
            with open(x_path, "rb") as f:
                X_train = pickle.load(f)
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

    if X_train is not None:
        cat_candidates = [c for c in ["Item_Identifier", "Item_Type", "Outlet_Identifier"] if c in X_train.columns]

        for col in cat_candidates:
            train_vals = X_train[col].astype(object).tolist()
            seen = {}
            ordered_unique = []
            for v in train_vals:
                if v not in seen:
                    seen[v] = True
                    ordered_unique.append(v)

            mapping = {val: idx for idx, val in enumerate(ordered_unique)}

            if col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]) and all(pd.api.types.is_number(x) for x in ordered_unique):
                    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(-1).astype(int)
                else:
                    df[col] = df[col].map(lambda x: mapping.get(x, -1)).astype(int)
            else:
                df[col] = -1

        for col in X_train.columns:
            if col not in df.columns:
                if pd.api.types.is_numeric_dtype(X_train[col]):
                    df[col] = 0
                else:
                    df[col] = -1

        df = df.reindex(columns=X_train.columns, fill_value=0)
    else:
        from sklearn.preprocessing import LabelEncoder
        cat_candidates = [c for c in ["Item_Identifier", "Item_Type", "Outlet_Identifier"] if c in df.columns]
        for col in cat_candidates:
            if df[col].dtype == object:
                try:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                except Exception:
                    uniques = df[col].astype(str).unique().tolist()
                    mapping = {v: i for i, v in enumerate(uniques)}
                    df[col] = df[col].map(lambda x: mapping.get(str(x), -1)).astype(int)

    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

    return df


def to_download_bytes(df: pd.DataFrame):
    return df.to_csv(index=False).encode()


# ------------------------
# Load assets
# ------------------------
logo_path = load_logo()
with st.spinner("🔄 Loading model & data..."):
    model, X_train, full_df = load_model_and_data()


# ------------------------
# Prepare dropdown options for Item/Outlet fields
# ------------------------
def safe_unique_list(df, col):
    try:
        if df is None:
            return []
        if col not in df.columns:
            return []
        vals = df[col].dropna().unique().tolist()
        vals = [str(v) for v in vals]
        vals_sorted = sorted(vals)
        return vals_sorted
    except Exception:
        return []

if full_df is not None and isinstance(full_df, pd.DataFrame):
    ITEM_ID_OPTIONS = safe_unique_list(full_df, "Item_Identifier")
    ITEM_TYPE_OPTIONS = safe_unique_list(full_df, "Item_Type")
    OUTLET_ID_OPTIONS = safe_unique_list(full_df, "Outlet_Identifier")
elif X_train is not None and isinstance(X_train, pd.DataFrame):
    ITEM_ID_OPTIONS = safe_unique_list(X_train, "Item_Identifier")
    ITEM_TYPE_OPTIONS = safe_unique_list(X_train, "Item_Type")
    OUTLET_ID_OPTIONS = safe_unique_list(X_train, "Outlet_Identifier")
else:
    ITEM_ID_OPTIONS = []
    ITEM_TYPE_OPTIONS = []
    OUTLET_ID_OPTIONS = []


def select_or_text(label, options, default_value):
    if options:
        try:
            index = options.index(default_value) if default_value in options else 0
        except Exception:
            index = 0
        return st.selectbox(label, options=options, index=index)
    else:
        return st.text_input(label, value=default_value)


# ------------------------
# Sidebar (cleaned — no asset link)
# ------------------------
with st.sidebar:
    st.markdown('<div style="text-align: center; margin-bottom: 24px;">', unsafe_allow_html=True)
    if logo_path:
        st.image(logo_path, width=120)
    else:
        st.markdown("### 🛒 Sales")
        st.markdown('<p style="color: #94a3b8; font-size: 13px;">Sales Intelligence Platform</p>',
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


# ------------------------
# Header
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
# SINGLE PREDICTION (unchanged UI, improved backend + dropdowns)
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
            Item_Identifier = select_or_text("Item Identifier", ITEM_ID_OPTIONS, default_value="FDG33")
            Item_Weight = st.number_input("Item Weight (kg)", min_value=0.0, max_value=100.0, value=15.5, step=0.1)
        with c2:
            Item_Fat_Content = st.selectbox("Fat Content", options=["Low Fat", "Regular"])
            Item_Visibility = st.number_input("Visibility", min_value=0.0, max_value=1.0, value=0.05, step=0.01,
                                              help="Display visibility metric")
        with c3:
            Item_Type = select_or_text("Item Type", ITEM_TYPE_OPTIONS, default_value="Seafood")
            Item_MRP = st.number_input("MRP ($)", min_value=0.0, max_value=10000.0, value=225.5, step=0.5)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 🏪 Outlet Information")
        c4, c5, c6 = st.columns(3)
        with c4:
            Outlet_Identifier = select_or_text("Outlet Identifier", OUTLET_ID_OPTIONS, default_value="OUT027")
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
# ANALYTICS (upgraded metric cards — labels inside cards)
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
            total = len(data_for_viz)
            avg_sales = data_for_viz["Item_Outlet_Sales"].mean()
            max_sales = data_for_viz["Item_Outlet_Sales"].max()
            min_sales = data_for_viz["Item_Outlet_Sales"].min()

            cols = st.columns(4, gap="large")
            vals = [
                ("TOTAL RECORDS", f"{total:,}", "📦"),
                ("AVG SALES", f"${avg_sales:,.0f}", "💲"),
                ("MAX SALES", f"${max_sales:,.0f}", "📈"),
                ("MIN SALES", f"${min_sales:,.0f}", "📉")
            ]

            for col, (label, value, icon) in zip(cols, vals):
                with col:
                    st.markdown(f'''
                        <div class="metric-card" style="padding:22px 22px; display:flex; flex-direction:column; justify-content:space-between; height:140px;">
                            <div style="display:flex; flex-direction:column; gap:6px;">
                                <div class="metric-label">{label}</div>
                                <div class="metric-value">{value}</div>
                            </div>
                            <div style="display:flex; justify-content:flex-end; align-items:center; gap:8px;">
                                <div class="metric-icon">{icon}</div>
                                <div class="metric-sub">Updated</div>
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)

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
# ABOUT (clean — no view asset)
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
