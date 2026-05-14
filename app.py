import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AutoVal · Car Price Intelligence",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  GLOBAL CSS  (dark luxury theme)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

/* ── Root palette ── */
:root {
  --bg:          #0d0f14;
  --surface:     #161920;
  --surface2:    #1e2230;
  --border:      #2a2f3f;
  --accent:      #e8a020;
  --accent2:     #f0c060;
  --text-hi:     #f0f2f8;
  --text-lo:     #7a8099;
  --danger:      #e05050;
  --success:     #38c98a;
  --radius:      14px;
}

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
  background: var(--bg) !important;
  font-family: 'DM Sans', sans-serif;
  color: var(--text-hi);
}

/* Hide default header */
header[data-testid="stHeader"] { display: none !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-family: 'DM Sans', sans-serif; }

/* ── Sidebar selectbox ── */
[data-testid="stSidebar"] .stSelectbox label { color: var(--text-lo) !important; font-size: 11px; letter-spacing: .08em; text-transform: uppercase; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px 22px !important;
}
[data-testid="stMetric"] label { color: var(--text-lo) !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: .09em; }
[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Syne', sans-serif; font-size: 28px !important; font-weight: 700; }

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border-radius: var(--radius); overflow: hidden; border: 1px solid var(--border); }

/* ── Buttons ── */
.stButton > button {
  background: var(--accent) !important;
  color: #0d0f14 !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  letter-spacing: .05em !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 12px 28px !important;
  transition: all .2s ease !important;
}
.stButton > button:hover { background: var(--accent2) !important; transform: translateY(-1px); }

/* ── Number inputs / selects ── */
[data-baseweb="input"], [data-baseweb="select"] {
  background: var(--surface2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 8px !important;
  color: var(--text-hi) !important;
}

/* ── Info / success / error boxes ── */
.stAlert { border-radius: var(--radius) !important; }

/* ── Slider ── */
[data-baseweb="slider"] [role="slider"] { background: var(--accent) !important; }

/* ── Tabs ── */
[data-baseweb="tab"] { font-family: 'Syne', sans-serif !important; }
[aria-selected="true"] { color: var(--accent) !important; border-bottom-color: var(--accent) !important; }

/* ── Section divider ── */
.section-title {
  font-family: 'Syne', sans-serif;
  font-size: 26px;
  font-weight: 800;
  color: var(--text-hi);
  margin-bottom: 4px;
}
.section-sub {
  font-family: 'DM Sans', sans-serif;
  font-size: 14px;
  color: var(--text-lo);
  margin-bottom: 28px;
}
.tag {
  display: inline-block;
  background: rgba(232,160,32,.15);
  color: var(--accent);
  border: 1px solid rgba(232,160,32,.35);
  border-radius: 20px;
  font-size: 11px;
  font-weight: 600;
  letter-spacing: .06em;
  padding: 3px 12px;
  margin-bottom: 10px;
  text-transform: uppercase;
}
hr.divider {
  border: none;
  border-top: 1px solid var(--border);
  margin: 28px 0;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOGO  (pure SVG — no external assets needed)
# ─────────────────────────────────────────────
LOGO_SVG = """
<svg width="220" height="54" viewBox="0 0 220 54" fill="none" xmlns="http://www.w3.org/2000/svg">
  <!-- Car silhouette icon -->
  <g transform="translate(2,8)">
    <!-- Body -->
    <path d="M4 24 L8 14 Q10 8 18 8 L32 8 Q40 8 44 14 L48 24 L52 24 Q54 24 54 27 L54 32 Q54 34 52 34 L48 34 Q47 38 43 38 Q39 38 38 34 L18 34 Q17 38 13 38 Q9 38 8 34 L4 34 Q2 34 2 32 L2 27 Q2 24 4 24 Z"
          fill="#e8a020"/>
    <!-- Windshield -->
    <path d="M14 24 L17 14 L37 14 L40 24 Z" fill="#0d0f14" opacity="0.6"/>
    <!-- Front wheel -->
    <circle cx="43" cy="36" r="4" fill="#0d0f14"/>
    <circle cx="43" cy="36" r="2" fill="#e8a020"/>
    <!-- Rear wheel -->
    <circle cx="13" cy="36" r="4" fill="#0d0f14"/>
    <circle cx="13" cy="36" r="2" fill="#e8a020"/>
    <!-- Headlight -->
    <rect x="47" y="26" width="5" height="3" rx="1.5" fill="#fff" opacity="0.9"/>
    <!-- Tail light -->
    <rect x="2" y="26" width="4" height="3" rx="1.5" fill="#e05050" opacity="0.9"/>
  </g>
  <!-- Word-mark -->
  <text x="70" y="30" font-family="Syne, sans-serif" font-weight="800" font-size="22"
        fill="#f0f2f8" letter-spacing="1">AUTO</text>
  <text x="119" y="30" font-family="Syne, sans-serif" font-weight="400" font-size="22"
        fill="#e8a020" letter-spacing="1">VAL</text>
  <!-- Tagline -->
  <text x="70" y="45" font-family="DM Sans, sans-serif" font-size="9.5"
        fill="#7a8099" letter-spacing="2.5">CAR PRICE INTELLIGENCE</text>
  <!-- Accent line -->
  <line x1="70" y1="34" x2="195" y2="34" stroke="#2a2f3f" stroke-width="1"/>
</svg>
"""


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(LOGO_SVG, unsafe_allow_html=True)
    st.markdown("<hr style='border:none;border-top:1px solid #2a2f3f;margin:20px 0'>", unsafe_allow_html=True)

    st.markdown("<p style='font-size:10px;letter-spacing:.12em;text-transform:uppercase;color:#7a8099;margin-bottom:10px'>Navigation</p>", unsafe_allow_html=True)

    pages = {
        "📊  Data Overview":           "Data Overview",
        "📈  Correlation Heatmap":     "Correlation Heatmap",
        "🤖  Model Evaluation":        "Model Evaluation",
        "🔮  Good Deal Analysis":      "Good Deal Analysis",
        "🧮  Price Calculator":        "Price Calculator",
    }
    app_mode = st.radio("", list(pages.keys()), label_visibility="collapsed")
    selected = pages[app_mode]

    st.markdown("<hr style='border:none;border-top:1px solid #2a2f3f;margin:20px 0'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:11px;color:#7a8099;line-height:1.7'>
      <b style='color:#f0f2f8'>AutoVal</b> uses Machine Learning to predict fair market values and identify good car deals.<br><br>
      Models: <span style='color:#e8a020'>Linear Regression</span> + <span style='color:#e8a020'>Random Forest</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  PLOTLY THEME HELPER
# ─────────────────────────────────────────────
def styled_fig(fig, height=400):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#161920",
        font=dict(family="DM Sans", color="#7a8099", size=12),
        height=height,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(gridcolor="#2a2f3f", linecolor="#2a2f3f"),
        yaxis=dict(gridcolor="#2a2f3f", linecolor="#2a2f3f"),
    )
    return fig


# ─────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("CAR DATA1.csv")
    df.columns = df.columns.str.strip()
    return df

df_raw = load_data()

# ─────────────────────────────────────────────
#  PREPROCESS
# ─────────────────────────────────────────────
df = df_raw.copy()
df["Car_Age"] = 2025 - df["Year"]
car_names = df_raw["Car_Name"].unique()
df_model = df.drop(["Year", "Car_Name"], axis=1)
df_model = pd.get_dummies(
    df_model,
    columns=["Fuel_Type", "Selling_type", "Transmission"],
    drop_first=True,
)

X = df_model.drop("Selling_Price", axis=1)
y = df_model["Selling_Price"]

# ─────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────
@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    lr = LinearRegression()
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    return lr, rf, X_train, X_test, y_train, y_test

lr, rf, X_train, X_test, y_train, y_test = train_models(X, y)

y_lr = lr.predict(X_test)
y_rf = rf.predict(X_test)

df["Predicted_LR"] = lr.predict(X)
df["Predicted_RF"] = rf.predict(X)
df["Good_Deal_LR"] = (df["Selling_Price"] < df["Predicted_LR"]).astype(int)
df["Good_Deal_RF"] = (df["Selling_Price"] < df["Predicted_RF"]).astype(int)

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


# ═══════════════════════════════════════════
#  PAGE: DATA OVERVIEW
# ═══════════════════════════════════════════
if selected == "Data Overview":
    st.markdown('<div class="tag">Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Data Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Explore the raw training data powering AutoVal\'s predictions.</div>', unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Records", f"{df_raw.shape[0]:,}")
    k2.metric("Features", str(df_raw.shape[1]))
    k3.metric("Avg. Selling Price", f"₹{df_raw['Selling_Price'].mean():.2f}L")
    k4.metric("Unique Cars", str(df_raw["Car_Name"].nunique()))

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    col_l, col_r = st.columns([3, 2])
    with col_l:
        st.markdown("#### 📋 Full Dataset")
        st.dataframe(df_raw, use_container_width=True, height=380)
    with col_r:
        st.markdown("#### 📐 Descriptive Statistics")
        st.dataframe(df_raw.describe().T.round(2), use_container_width=True, height=380)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    # Distribution charts
    st.markdown("#### 📊 Distribution Insights")
    c1, c2, c3 = st.columns(3)

    with c1:
        fuel_counts = df_raw["Fuel_Type"].value_counts().reset_index()
        fuel_counts.columns = ["Fuel_Type", "Count"]
        fig = px.pie(fuel_counts, values="Count", names="Fuel_Type",
                     title="Fuel Type Split",
                     color_discrete_sequence=["#e8a020", "#38c98a", "#7a8099"])
        st.plotly_chart(styled_fig(fig, 300), use_container_width=True)

    with c2:
        trans_counts = df_raw["Transmission"].value_counts().reset_index()
        trans_counts.columns = ["Transmission", "Count"]
        fig = px.bar(trans_counts, x="Transmission", y="Count",
                     title="Transmission Type",
                     color="Transmission",
                     color_discrete_sequence=["#e8a020", "#38c98a"])
        st.plotly_chart(styled_fig(fig, 300), use_container_width=True)

    with c3:
        fig = px.histogram(df_raw, x="Selling_Price", nbins=30,
                           title="Selling Price Distribution",
                           color_discrete_sequence=["#e8a020"])
        st.plotly_chart(styled_fig(fig, 300), use_container_width=True)


# ═══════════════════════════════════════════
#  PAGE: CORRELATION HEATMAP
# ═══════════════════════════════════════════
elif selected == "Correlation Heatmap":
    st.markdown('<div class="tag">Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Correlation Heatmap</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Understand how features relate to each other and to the target variable.</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📊 Raw Features", "🤖 Including Predictions"])

    for tab, label, drop_cols in [
        (tab1, "Raw Feature Correlations", ["Predicted_LR", "Predicted_RF"]),
        (tab2, "Correlation Including Predictions", []),
    ]:
        with tab:
            cols_to_drop = [c for c in drop_cols if c in df_model.columns]
            corr_data = df_model.drop(columns=cols_to_drop, errors="ignore").corr()
            fig, ax = plt.subplots(figsize=(13, 6))
            fig.patch.set_facecolor("#161920")
            ax.set_facecolor("#161920")
            sns.heatmap(
                corr_data, cmap="YlOrBr", annot=True, fmt=".2f",
                linewidths=0.5, linecolor="#2a2f3f",
                annot_kws={"size": 9, "color": "#0d0f14"},
                ax=ax,
            )
            ax.tick_params(colors="#7a8099", labelsize=9)
            plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
            ax.set_title(label, color="#f0f2f8", fontsize=14, pad=12)
            st.pyplot(fig)


# ═══════════════════════════════════════════
#  PAGE: MODEL EVALUATION
# ═══════════════════════════════════════════
elif selected == "Model Evaluation":
    st.markdown('<div class="tag">ML Models</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Model Evaluation & Comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Side-by-side performance of Linear Regression vs Random Forest on held-out test data.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    for col, label, y_pred, color in [
        (col1, "Linear Regression", y_lr, "#e8a020"),
        (col2, "Random Forest", y_rf, "#38c98a"),
    ]:
        with col:
            st.markdown(f"<h4 style='color:{color};font-family:Syne,sans-serif'>{label}</h4>", unsafe_allow_html=True)
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE",  f"{mean_absolute_error(y_test, y_pred):.2f}")
            m2.metric("RMSE", f"{rmse(y_test, y_pred):.2f}")
            m3.metric("R²",   f"{r2_score(y_test, y_pred):.3f}")

            fig = px.scatter(
                x=y_test, y=y_pred,
                labels={"x": "Actual Price (L)", "y": "Predicted Price (L)"},
                title="Actual vs Predicted",
                opacity=0.65,
            )
            fig.update_traces(marker=dict(color=color, size=7))
            # perfect-fit reference line
            mn, mx = float(y_test.min()), float(y_test.max())
            fig.add_trace(go.Scatter(
                x=[mn, mx], y=[mn, mx],
                mode="lines",
                line=dict(color="white", dash="dot", width=1.5),
                name="Perfect Fit",
            ))
            st.plotly_chart(styled_fig(fig), use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("#### 📋 Metrics Comparison Table")
    metrics_df = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R²"],
        "Linear Regression": [
            round(mean_absolute_error(y_test, y_lr), 3),
            round(rmse(y_test, y_lr), 3),
            round(r2_score(y_test, y_lr), 3),
        ],
        "Random Forest": [
            round(mean_absolute_error(y_test, y_rf), 3),
            round(rmse(y_test, y_rf), 3),
            round(r2_score(y_test, y_rf), 3),
        ],
    })
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    fig = px.bar(
        metrics_df, x="Metric",
        y=["Linear Regression", "Random Forest"],
        barmode="group",
        color_discrete_sequence=["#e8a020", "#38c98a"],
        title="Side-by-side Metric Comparison",
    )
    st.plotly_chart(styled_fig(fig), use_container_width=True)

    st.info("✅ Random Forest typically achieves higher R² and lower error — it is the primary engine for the Price Calculator.")


# ═══════════════════════════════════════════
#  PAGE: GOOD DEAL ANALYSIS
# ═══════════════════════════════════════════
elif selected == "Good Deal Analysis":
    st.markdown('<div class="tag">Deal Intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Good Deal Analysis</div>', unsafe_allow_html=True)
    st.markdown("<div class=\"section-sub\">Cars whose asking price is below the model&#39;s fair-value prediction are flagged as good deals.</div>", unsafe_allow_html=True)

    total = len(df)
    gd_lr = int(df["Good_Deal_LR"].sum())
    gd_rf = int(df["Good_Deal_RF"].sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Cars", total)
    k2.metric("Good Deals (LR)", gd_lr)
    k3.metric("Good Deals (RF)", gd_rf)
    k4.metric("Deal Rate (RF)", f"{gd_rf/total*100:.1f}%")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.pie(
            names=["Good Deal", "Overpriced / Fair"],
            values=[gd_rf, total - gd_rf],
            title="Deal Distribution (Random Forest)",
            color_discrete_sequence=["#38c98a", "#2a2f3f"],
            hole=0.45,
        )
        st.plotly_chart(styled_fig(fig, 360), use_container_width=True)

    with c2:
        # Scatter: Selling vs Predicted coloured by deal flag
        fig = px.scatter(
            df, x="Selling_Price", y="Predicted_RF",
            color=df["Good_Deal_RF"].map({1: "Good Deal", 0: "Not a Good Deal"}),
            color_discrete_map={"Good Deal": "#38c98a", "Not a Good Deal": "#e05050"},
            labels={"Selling_Price": "Asking Price (L)", "Predicted_RF": "Fair Value (L)"},
            title="Asking Price vs Fair Value",
            opacity=0.7,
        )
        mn2 = float(df["Selling_Price"].min())
        mx2 = float(df["Selling_Price"].max())
        fig.add_trace(go.Scatter(
            x=[mn2, mx2], y=[mn2, mx2],
            mode="lines",
            line=dict(color="white", dash="dot", width=1.5),
            name="Fair Price Line",
        ))
        st.plotly_chart(styled_fig(fig, 360), use_container_width=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("#### 🟢 Top Good Deals")
    top_deals = (
        df[df["Good_Deal_RF"] == 1]
        .copy()
        .assign(Saving=lambda d: (d["Predicted_RF"] - d["Selling_Price"]).round(2))
        .sort_values("Saving", ascending=False)
        [["Car_Name", "Year", "Selling_Price", "Predicted_RF", "Saving"]]
        .head(15)
        .rename(columns={"Predicted_RF": "Fair Value (L)", "Selling_Price": "Asking Price (L)", "Saving": "You Save (L)"})
    )
    st.dataframe(top_deals, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════
#  PAGE: PRICE CALCULATOR
# ═══════════════════════════════════════════
elif selected == "Price Calculator":
    st.markdown('<div class="tag">Live Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Car Price Calculator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Enter the details of any car to instantly get its fair market value and deal verdict.</div>', unsafe_allow_html=True)

    car_name = st.selectbox("🚗  Select Car Name", sorted(car_names))
    car_df = df_raw[df_raw["Car_Name"] == car_name]
    avg_year  = int(car_df["Year"].mean())
    avg_price = float(car_df["Selling_Price"].mean())

    kms_col = next((c for c in df_raw.columns if "kms" in c.lower() or "kilometer" in c.lower()), None)
    avg_kms = int(car_df[kms_col].mean()) if kms_col else 30000

    # Info bar
    st.markdown(f"""
    <div style='background:#1e2230;border:1px solid #2a2f3f;border-radius:12px;padding:14px 20px;margin-bottom:20px;display:flex;gap:40px;align-items:center'>
      <div><span style='font-size:10px;color:#7a8099;text-transform:uppercase;letter-spacing:.1em'>Historical Avg Price</span><br>
           <span style='font-family:Syne,sans-serif;font-size:22px;font-weight:700;color:#e8a020'>₹{avg_price:.2f}L</span></div>
      <div><span style='font-size:10px;color:#7a8099;text-transform:uppercase;letter-spacing:.1em'>Avg Year</span><br>
           <span style='font-family:Syne,sans-serif;font-size:22px;font-weight:700;color:#f0f2f8'>{avg_year}</span></div>
      <div><span style='font-size:10px;color:#7a8099;text-transform:uppercase;letter-spacing:.1em'>Avg Kms Driven</span><br>
           <span style='font-family:Syne,sans-serif;font-size:22px;font-weight:700;color:#f0f2f8'>{avg_kms:,}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("#### ✏️  Enter Car Details")

    col1, col2 = st.columns(2)
    with col1:
        asking_price = st.number_input("Asking / Present Price (Lakhs)", 0.0, 100.0, round(avg_price, 2), step=0.1)
        kms          = st.number_input("Kilometers Driven", 0, 500_000, avg_kms, step=1000)
        car_age      = st.slider("Car Age (Years)", 0, 25, 2025 - avg_year)
    with col2:
        owner        = st.selectbox("Previous Owners", [0, 1, 3])
        fuel         = st.selectbox("Fuel Type", ["Petrol", "Diesel"])
        seller       = st.selectbox("Seller Type", ["Dealer", "Individual"])
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("🔍  Predict Fair Value"):
        input_df = pd.DataFrame([{
            "Present_Price":           asking_price,
            "Driven_kms":              kms,
            "Owner":                   owner,
            "Car_Age":                 car_age,
            "Fuel_Type_Diesel":        1 if fuel == "Diesel" else 0,
            "Selling_type_Individual": 1 if seller == "Individual" else 0,
            "Transmission_Manual":     1 if transmission == "Manual" else 0,
        }])
        input_df = input_df.reindex(columns=X.columns, fill_value=0)
        predicted = rf.predict(input_df)[0]
        diff = asking_price - predicted
        pct  = abs(diff) / predicted * 100

        st.markdown("<hr class='divider'>", unsafe_allow_html=True)
        st.markdown("#### 📊  Prediction Results")

        r1, r2, r3 = st.columns(3)
        r1.metric("Asking Price",    f"₹{asking_price:.2f}L")
        r2.metric("Fair Value (RF)", f"₹{predicted:.2f}L")
        r3.metric("Difference",      f"{'▲' if diff > 0 else '▼'} ₹{abs(diff):.2f}L ({pct:.1f}%)")

        if asking_price <= predicted:
            st.success(f"🟢  **GOOD DEAL** — You're saving ₹{abs(diff):.2f}L ({pct:.1f}%) vs fair market value.")
        else:
            st.error(f"🔴  **OVERPRICED** — This car is listed ₹{abs(diff):.2f}L ({pct:.1f}%) above its fair value.")

        # Price comparison bar chart
        fig = go.Figure()
        bar_labels = ["Asking Price", "Predicted Fair Value", "Historical Avg"]
        bar_values = [asking_price, predicted, avg_price]
        bar_colors = [
            "#e05050" if asking_price > predicted else "#38c98a",
            "#38c98a",
            "#7a8099",
        ]
        fig.add_trace(go.Bar(
            x=bar_labels, y=bar_values,
            marker_color=bar_colors,
            text=[f"₹{v:.2f}L" for v in bar_values],
            textposition="outside",
            textfont=dict(color="#f0f2f8", size=13, family="Syne"),
        ))
        fig.update_layout(
            title=f"Price Breakdown — {car_name}",
            yaxis_title="Price (Lakhs)",
            showlegend=False,
        )
        st.plotly_chart(styled_fig(fig, 380), use_container_width=True)
