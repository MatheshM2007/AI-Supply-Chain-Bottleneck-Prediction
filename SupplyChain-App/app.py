from base64 import b64encode
from html import escape
from pathlib import Path
import warnings

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Constants
ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "global_supply_chain_risk.csv"
BACKGROUND = ROOT / "assets" / "bg.jpg"
TARGET = "Disruption_Occurred"
NUM = [
    "Distance_km", "Weight_MT", "Fuel_Price_Index",
    "Geopolitical_Risk_Score", "Carrier_Reliability_Score",
    "Shipment_Month", "Shipment_DayOfWeek",
]
CAT = [
    "Origin_Port", "Destination_Port", "Transport_Mode",
    "Product_Category", "Weather_Condition",
]
FEATURE_LABELS = {
    "Weather_Condition": "Weather",
    "Geopolitical_Risk_Score": "Political risk",
    "Carrier_Reliability_Score": "Carrier reliability",
    "Distance_km": "Distance",
    "Weight_MT": "Weight",
}

warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="ShipSafe | Know before it ships",
    page_icon="📦",
    layout="wide",
)

# Embed the local port image for offline use and provide a safe fallback.
if BACKGROUND.is_file():
    image_b64 = b64encode(BACKGROUND.read_bytes()).decode("ascii")
    image_uri = f"data:image/jpeg;base64,{image_b64}"
    hero_background = (
        "linear-gradient(90deg, rgba(10,31,68,.88) 0%, "
        "rgba(10,31,68,.55) 55%, rgba(10,31,68,.25) 100%), "
        f"url('{image_uri}')"
    )
    footer_background = f"url('{image_uri}')"
    footer_surface = "transparent"
else:
    hero_background = "none"
    footer_background = "none"
    footer_surface = "var(--navy)"

# Responsive styles for desktop and phone layouts.
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@700;800&display=swap');
:root {{
    color-scheme: light;
    --primary-color: #FFB020;
    --background-color: #F4F8FC;
    --secondary-background-color: #FFFFFF;
    --text-color: #0A1F44;
    --navy: #0A1F44;
    --amber: #FFB020;
    --ice: #F4F8FC;
    --slate: #4A5B73;
}}
html, body, .stApp, button, input, select, textarea {{
    font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
}}
html, body, .stApp {{ margin: 0; min-width: 320px; overflow-x: hidden; }}
.stApp {{ background: var(--ice); color: var(--navy); }}
#MainMenu, footer:not(.site-footer), header, [data-testid="stSidebar"], [data-testid="stToolbar"] {{
    display: none !important;
}}
.block-container {{ width: 100%; max-width: none; padding: 0 0 24px !important; }}
h1, h2, h3, h4 {{ font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important; }}
.site-nav {{ width: 100%; min-height: 76px; background: var(--navy); }}
.nav-inner {{
    max-width: 1180px; min-height: 76px; margin: auto; padding: 0 24px;
    display: flex; align-items: center; justify-content: space-between;
}}
.brand {{
    color: #FFFFFF !important; font: 800 25px 'Plus Jakarta Sans', sans-serif;
    letter-spacing: -.04em; text-decoration: none !important;
}}
.desktop-links {{ display: flex; align-items: center; gap: 38px; }}
.desktop-links a {{
    position: relative; color: #FFFFFF; font-size: 14px; font-weight: 600;
    letter-spacing: .08em; text-decoration: none; text-transform: uppercase;
    padding: 27px 0;
}}
.desktop-links a.active::after, .desktop-links a:hover::after {{
    position: absolute; right: 0; bottom: 17px; left: 0; height: 3px;
    background: var(--amber); content: '';
}}
.hero {{
    min-height: 85vh; width: 100%; display: flex; align-items: center;
    background-color: var(--navy); background-image: {hero_background};
    background-size: cover; background-position: center;
}}
.hero-inner {{ width: 100%; max-width: 1180px; margin: auto; padding: 64px 24px; }}
.eyebrow {{
    color: var(--amber); font-size: 13px; font-weight: 700;
    letter-spacing: .12em; text-transform: uppercase;
}}
.hero h1 {{
    max-width: 680px; margin: 18px 0 16px; color: #FFFFFF;
    font-size: 56px; font-weight: 800;
    letter-spacing: -.055em; line-height: 1.08;
}}
.hero p {{ margin: 0 0 32px; color: #FFFFFF; font-size: 20px; line-height: 1.6; }}
.hero-cta {{
    min-height: 52px; display: inline-flex; align-items: center; justify-content: center;
    padding: 0 26px; border-radius: 4px; background: var(--amber);
    color: var(--navy) !important; font-size: 16px; font-weight: 700;
    letter-spacing: .08em; text-decoration: none !important;
}}
.hero-cta:hover {{ background: #FFFFFF; }}
.st-key-check-section {{ width: 100%; background: var(--ice); }}
.st-key-check-content {{ max-width: 1180px; margin: auto; padding: 72px 24px 80px; }}
.section-heading {{ margin: 0 0 28px; }}
.section-heading .eyebrow {{ color: var(--slate); }}
.section-heading h2 {{
    margin: 9px 0 8px; color: var(--navy); font-size: 36px;
    font-weight: 800; letter-spacing: -.04em;
}}
.section-heading p {{ margin: 0; color: var(--slate); font-size: 16px; line-height: 1.6; }}
.st-key-form-fields {{ width: 100%; background: #FFFFFF; padding: 22px 24px; }}
[data-testid="stForm"] {{ border: 0; background: transparent; padding: 0; }}
div[data-testid="stHorizontalBlock"] {{ gap: 24px; }}
[data-testid="stWidgetLabel"] p, label {{
    color: var(--slate) !important; font-size: 13px !important;
    font-weight: 600 !important; letter-spacing: .08em !important;
    text-transform: uppercase !important;
}}
[data-testid="stSelectbox"] [role="group"],
[data-testid="stSelectbox"] [role="combobox"],
[data-testid="stDateInputField"] {{
    min-height: 48px; background: #FFFFFF !important; border: 0 !important;
    border-bottom: 2px solid var(--navy) !important; border-radius: 0 !important;
    box-shadow: none !important; color: var(--navy) !important;
    font-size: 16px !important;
}}
[data-testid="stSelectbox"] [role="group"]:focus-within,
[data-testid="stSelectbox"] [role="combobox"]:focus,
[data-testid="stDateInputField"]:focus-within {{
    border-bottom-color: var(--amber) !important;
}}
[data-testid="stDateInputField"] [role="spinbutton"],
[data-testid="stDateInputField"] [data-type="literal"] {{ color: var(--navy) !important; }}
[data-testid="stDateInputCalendarButton"] {{
    min-width: 44px; min-height: 44px; background: #FFFFFF !important;
    border: 0 !important; color: var(--navy) !important;
}}
[data-testid="stSlider"] p {{ color: var(--navy) !important; font-size: 14px !important; }}
[data-testid="stSlider"] [role="slider"] {{
    width: 24px !important; height: 24px !important;
    background: var(--amber) !important; border: 2px solid var(--amber) !important;
}}
[data-testid="stSlider"] [data-testid="stTickBar"] {{ background: var(--navy) !important; }}
[data-testid="stExpander"], [data-testid="stExpander"] details {{
    border: 0 !important; border-bottom: 1px solid rgba(74,91,115,.35) !important;
    border-radius: 0 !important; box-shadow: none !important; background: transparent !important;
}}
[data-testid="stExpander"] summary {{
    min-height: 48px; padding: 8px 0 !important; border: 0 !important;
    border-radius: 0 !important; color: var(--navy) !important;
}}
[data-testid="stExpander"] summary p {{ color: var(--navy) !important; font-size: 16px !important; }}
[data-testid="stFormSubmitButton"], [data-testid="stFormSubmitButton"] > button {{
    width: 100% !important;
}}
[data-testid="stFormSubmitButton"] > button {{
    min-height: 52px !important; border: 0 !important; border-radius: 4px !important;
    background: var(--navy) !important; color: #FFFFFF !important;
    font-size: 16px !important; font-weight: 700 !important;
    letter-spacing: .08em !important; text-transform: uppercase !important;
}}
[data-testid="stFormSubmitButton"] > button:hover,
[data-testid="stFormSubmitButton"] > button:disabled {{
    background: var(--navy) !important; color: #FFFFFF !important; opacity: 1 !important;
}}
[data-testid="stFormSubmitButton"] > button:hover {{
    background: var(--amber) !important; color: var(--navy) !important;
}}
.result-band {{
    width: 100%; box-sizing: border-box; padding: 42px 24px 36px;
    text-align: center; color: #FFFFFF;
}}
.result-on-time {{ background: #1FAF7A; }}
.result-at-risk {{ background: #F59E0B; color: var(--navy); }}
.result-critical {{ background: #D93A3A; }}
.result-probability {{
    font: 800 88px/1 'Plus Jakarta Sans', system-ui, sans-serif;
    letter-spacing: -.06em;
}}
.result-level {{ margin-top: 6px; font-size: 28px; font-weight: 700; }}
.result-advice {{ margin: 8px 0 24px; font-size: 18px; line-height: 1.5; }}
.gauge {{ position: relative; max-width: 760px; margin: 0 auto; }}
.gauge-track {{ height: 5px; background: rgba(255,255,255,.55); position: relative; }}
.result-at-risk .gauge-track {{ background: rgba(10,31,68,.28); }}
.gauge-fill {{ height: 100%; background: #FFFFFF; }}
.result-at-risk .gauge-fill {{ background: var(--navy); }}
.gauge-marker {{
    position: absolute; top: -5px; width: 2px; height: 15px;
    background: #FFFFFF;
}}
.result-at-risk .gauge-marker {{ background: var(--navy); }}
.gauge-value {{
    position: absolute; top: -26px; width: 2px; height: 26px;
    background: #FFFFFF; transform: translateX(-1px);
}}
.result-at-risk .gauge-value {{ background: var(--navy); }}
.gauge-labels {{
    display: flex; justify-content: space-between; margin-top: 12px;
    font-size: 12px; font-weight: 600;
}}
.st-key-insights-section {{ width: 100%; background: var(--ice); }}
.st-key-insights-content {{ max-width: 1180px; margin: auto; padding: 72px 24px; }}
.insight-columns {{ margin-top: 32px; }}
.insight-title {{
    margin: 0 0 22px; color: var(--navy); font-size: 23px;
    font-weight: 800; letter-spacing: -.03em;
}}
.bar-row {{
    display: grid; grid-template-columns: minmax(130px, 190px) 1fr 48px;
    align-items: center; gap: 14px; min-height: 48px;
    border-bottom: 1px solid rgba(74,91,115,.25); color: var(--slate); font-size: 16px;
}}
.bar-track {{ height: 7px; background: rgba(74,91,115,.18); }}
.bar-fill {{ height: 100%; background: var(--navy); }}
.route-fill {{ background: var(--amber); }}
.bar-value {{ color: var(--navy); text-align: right; font-size: 16px; font-weight: 600; }}
.stats {{
    display: grid; grid-template-columns: repeat(4, 1fr);
    margin-top: 54px; padding-top: 28px; border-top: 1px solid rgba(74,91,115,.35);
}}
.stat {{ padding: 4px 20px; border-right: 1px solid rgba(74,91,115,.35); }}
.stat:first-child {{ padding-left: 0; }}
.stat:last-child {{ border-right: 0; }}
.stat-number {{
    color: var(--navy); font: 800 42px/1 'Plus Jakarta Sans', system-ui, sans-serif;
    letter-spacing: -.05em;
}}
.stat-label {{
    margin-top: 9px; color: var(--slate); font-size: 13px;
    font-weight: 600; letter-spacing: .08em; text-transform: uppercase;
}}
.site-footer {{
    position: relative; isolation: isolate; overflow: hidden;
    width: 100%; min-height: 88px; display: flex; align-items: center;
    justify-content: center; background: {footer_surface}; color: #FFFFFF;
    font-size: 16px; font-weight: 600; letter-spacing: .04em;
}}
.footer-text {{ position: relative; z-index: 2; }}
.site-footer::before {{
    position: absolute; z-index: -2; inset: -10px; content: '';
    background-image: {footer_background}; background-size: cover;
    background-position: center 70%; filter: blur(4px) brightness(.42);
    transform: scale(1.04);
}}
.site-footer::after {{
    position: absolute; z-index: -1; inset: 0; content: '';
    background: rgba(10,31,68,.92);
}}
.mobile-nav {{ display: none; }}
@media (min-width: 769px) {{
    .desktop-links {{ display: flex; }}
}}
@media (max-width: 768px) {{
    html, body, .stApp {{ overflow-x: hidden !important; }}
    .block-container {{ padding-bottom: calc(84px + env(safe-area-inset-bottom)) !important; }}
    .nav-inner {{ min-height: 62px; padding: 0 16px; }}
    .brand {{ font-size: 22px; }}
    .desktop-links {{ display: none !important; }}
    .hero {{ min-height: 60vh; background-position: 58% center; }}
    .hero-inner {{ padding: 44px 16px; }}
    .hero h1 {{ max-width: 100%; margin: 14px 0; font-size: 36px; line-height: 1.15; }}
    .hero p {{ max-width: 330px; margin: 0 0 24px; font-size: 16px; }}
    .hero-cta {{ width: 100%; box-sizing: border-box; min-height: 52px; }}
    .st-key-check-content, .st-key-insights-content {{ padding: 48px 16px; }}
    .section-heading {{ margin-bottom: 22px; }}
    .section-heading h2 {{ font-size: 26px; }}
    .st-key-form-fields {{ padding: 16px; }}
    div[data-testid="stHorizontalBlock"] {{ flex-direction: column !important; gap: 8px !important; }}
    div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {{
        width: 100% !important; min-width: 100% !important; flex: 1 1 100% !important;
    }}
    [data-testid="stSelectbox"] [role="group"],
    [data-testid="stSelectbox"] [role="combobox"],
    [data-testid="stDateInputField"] {{ min-height: 52px; font-size: 16px !important; }}
    div[data-testid="stElementContainer"]:has([data-testid="stFormSubmitButton"]) {{
        position: sticky; bottom: calc(72px + env(safe-area-inset-bottom));
        z-index: 5; padding-top: 8px; background: #FFFFFF;
    }}
    [data-testid="stFormSubmitButton"] > button {{ min-height: 54px !important; }}
    .result-band {{ padding: 30px 16px 27px; }}
    .result-probability {{ font-size: 64px; }}
    .result-level {{ font-size: 24px; }}
    .result-advice {{ font-size: 16px; }}
    .st-key-insights-content {{ padding-bottom: 38px; }}
    .insight-columns {{ gap: 34px !important; }}
    .insight-title {{ font-size: 21px; }}
    .bar-row {{ grid-template-columns: minmax(105px, 1fr) minmax(65px, 1.2fr) 42px; gap: 9px; font-size: 16px; }}
    .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); row-gap: 22px; margin-top: 38px; }}
    .stat {{ padding: 4px 12px; }}
    .stat:nth-child(2) {{ border-right: 0; }}
    .stat:nth-child(3) {{ padding-left: 0; }}
    .stat-number {{ font-size: 34px; }}
    .stat-label {{ font-size: 10px; }}
    .site-footer {{ min-height: 76px; text-align: center; }}
    .mobile-nav {{
        position: fixed; z-index: 1000; right: 0; bottom: 0; left: 0;
        display: grid; grid-template-columns: repeat(3, 1fr);
        min-height: calc(64px + env(safe-area-inset-bottom));
        padding: 6px 12px env(safe-area-inset-bottom);
        background: var(--navy);
    }}
    .mobile-nav a {{
        display: flex; flex-direction: column; align-items: center; justify-content: center;
        gap: 3px; color: #FFFFFF; font-size: 11px; font-weight: 600;
        text-decoration: none;
    }}
    .mobile-nav a.active {{ color: var(--amber); }}
    .mobile-nav a {{ min-height: 48px; }}
    .mobile-nav svg {{ width: 21px; height: 21px; fill: currentColor; }}
    body:has(#check-shipment:target) .mobile-nav a[href="#home"],
    body:has(#insights:target) .mobile-nav a[href="#home"] {{ color: #FFFFFF; }}
    body:has(#check-shipment:target) .mobile-nav a[href="#check-shipment"],
    body:has(#insights:target) .mobile-nav a[href="#insights"] {{ color: var(--amber); }}
    body:has(#check-shipment:target) .desktop-links a[href="#home"]::after,
    body:has(#insights:target) .desktop-links a[href="#home"]::after {{ content: none; }}
    body:has(#check-shipment:target) .desktop-links a[href="#check-shipment"]::after,
    body:has(#insights:target) .desktop-links a[href="#insights"]::after {{ content: ''; }}
    }}
</style>
""", unsafe_allow_html=True)

# Model
@st.cache_resource(show_spinner="Preparing shipment predictions...")
def load_model():
    df = pd.read_csv(DATA)
    date = pd.to_datetime(df["Date"], errors="coerce")
    df["Shipment_Month"], df["Shipment_DayOfWeek"] = date.dt.month, date.dt.dayofweek
    X, y = df[NUM + CAT], df[TARGET].astype(int)
    prep = ColumnTransformer([
        ("n", Pipeline([
            ("i", SimpleImputer(strategy="median")),
            ("s", StandardScaler()),
        ]), NUM),
        ("c", Pipeline([
            ("i", SimpleImputer(strategy="most_frequent")),
            ("o", OneHotEncoder(handle_unknown="ignore")),
        ]), CAT),
    ])
    Xa, Xb, ya, yb = train_test_split(
        X, y, test_size=.2, random_state=42, stratify=y
    )
    Xa_t = prep.fit_transform(Xa)
    rf = RandomForestClassifier(
        n_estimators=150, min_samples_leaf=2, class_weight="balanced",
        random_state=42, n_jobs=-1,
    ).fit(Xa_t, ya)
    accuracy = accuracy_score(yb, rf.predict(prep.transform(Xb)))
    importance = {}
    for name, value in zip(prep.get_feature_names_out(), rf.feature_importances_):
        feature = next(item for item in NUM + CAT if name.split("__", 1)[1].startswith(item))
        importance[feature] = importance.get(feature, 0) + value
    return df, prep, rf, accuracy, pd.Series(importance).sort_values(ascending=False)


df, prep, rf, accuracy, importance = load_model()

# Navigation
st.markdown("""
<nav class="site-nav"><div class="nav-inner">
  <a class="brand" href="#home">ShipSafe</a>
  <div class="desktop-links">
    <a class="active" href="#home">Home</a>
    <a href="#check-shipment">Check Shipment</a>
    <a href="#insights">Insights</a>
  </div>
</div></nav>
""", unsafe_allow_html=True)

# Hero
st.markdown("""
<section class="hero" id="home"><div class="hero-inner">
  <div class="eyebrow">Look + Plan + Ship</div>
  <h1>Know before it ships.</h1>
  <p>Predict shipment delays before they happen.</p>
  <a class="hero-cta" href="#check-shipment">Check Shipment</a>
</div></section>
""", unsafe_allow_html=True)

# Check shipment
with st.container(key="check-section"):
    with st.container(key="check-content"):
        st.markdown("""
        <div class="section-heading" id="check-shipment">
          <div class="eyebrow">Shipment planning</div>
          <h2>Check shipment</h2>
          <p>Enter the shipment details to see its delay risk.</p>
        </div>
        """, unsafe_allow_html=True)
        with st.container(key="form-fields"):
            with st.form("prediction_form"):
                c1, c2, c3 = st.columns(3)
                origin = c1.selectbox("Origin port", sorted(df["Origin_Port"].dropna().unique()))
                destination = c2.selectbox(
                    "Destination port", sorted(df["Destination_Port"].dropna().unique()), index=1
                )
                transport = c3.selectbox(
                    "Transport mode", sorted(df["Transport_Mode"].dropna().unique())
                )
                c1, c2, c3 = st.columns(3)
                product = c1.selectbox(
                    "Product category", sorted(df["Product_Category"].dropna().unique())
                )
                weather = c2.selectbox(
                    "Weather", sorted(df["Weather_Condition"].dropna().unique())
                )
                ship_date = c3.date_input("Planned ship date")
                c1, c2, _ = st.columns(3)
                distance = c1.slider(
                    "Distance (km)", 100, int(df["Distance_km"].max()),
                    int(df["Distance_km"].median()),
                )
                weight = c2.slider(
                    "Weight (tonnes)", 1, int(df["Weight_MT"].max()),
                    int(df["Weight_MT"].median()),
                )
                with st.expander("More details"):
                    c1, c2, c3 = st.columns(3)
                    fuel = c1.slider(
                        "Fuel price index", float(df["Fuel_Price_Index"].min()),
                        float(df["Fuel_Price_Index"].max()), float(df["Fuel_Price_Index"].median()),
                    )
                    political_risk = c2.slider(
                        "Political risk", 0.0, 10.0,
                        float(df["Geopolitical_Risk_Score"].median()), .5,
                    )
                    carrier = c3.slider(
                        "Carrier reliability", 0.0, 1.0,
                        float(df["Carrier_Reliability_Score"].median()), .05,
                    )
                submitted = st.form_submit_button(
                    "Predict delay risk", use_container_width=True
                )

# Result
if submitted:
    row = pd.DataFrame([{
        "Distance_km": distance,
        "Weight_MT": weight,
        "Fuel_Price_Index": fuel,
        "Geopolitical_Risk_Score": political_risk,
        "Carrier_Reliability_Score": carrier,
        "Shipment_Month": ship_date.month,
        "Shipment_DayOfWeek": ship_date.weekday(),
        "Origin_Port": origin,
        "Destination_Port": destination,
        "Transport_Mode": transport,
        "Product_Category": product,
        "Weather_Condition": weather,
    }])
    st.session_state["shipment_probability"] = float(
        rf.predict_proba(prep.transform(row))[0][list(rf.classes_).index(1)]
    )

if "shipment_probability" in st.session_state:
    probability = st.session_state["shipment_probability"]
    if probability < .35:
        result_class, level, advice = "result-on-time", "On-Time", "Ship as planned."
    elif probability <= .65:
        result_class, level, advice = (
            "result-at-risk", "At-Risk", "Keep a backup carrier ready."
        )
    else:
        result_class, level, advice = (
            "result-critical", "Critical Delay",
            "Consider changing the route, date or carrier.",
        )
    st.markdown(f"""
    <section class="result-band {result_class}">
      <div class="result-probability">{probability:.0%}</div>
      <div class="result-level">{level}</div>
      <div class="result-advice">{advice}</div>
      <div class="gauge">
        <div class="gauge-track">
          <div class="gauge-fill" style="width:{probability * 100:.1f}%"></div>
          <span class="gauge-marker" style="left:35%"></span>
          <span class="gauge-marker" style="left:65%"></span>
          <span class="gauge-value" style="left:{probability * 100:.1f}%"></span>
        </div>
        <div class="gauge-labels"><span>0%</span><span>35%</span><span>65%</span><span>100%</span></div>
      </div>
    </section>
    """, unsafe_allow_html=True)

# Insights
with st.container(key="insights-section"):
    with st.container(key="insights-content"):
        st.markdown("""
        <div class="section-heading" id="insights">
          <div class="eyebrow">Shipment overview</div>
          <h2>Insights</h2>
        </div>
        """, unsafe_allow_html=True)
        left, right = st.columns(2, gap="large")
        featured = [
            "Weather_Condition", "Geopolitical_Risk_Score",
            "Carrier_Reliability_Score", "Distance_km", "Weight_MT",
        ]
        max_importance = max((importance.get(name, 0) for name in featured), default=1)
        with left:
            st.markdown('<h3 class="insight-title">What causes delays the most</h3>',
                        unsafe_allow_html=True)
            for feature in sorted(featured, key=lambda name: importance.get(name, 0), reverse=True):
                value = float(importance.get(feature, 0))
                width = value / max_importance * 100 if max_importance else 0
                st.markdown(
                    f'<div class="bar-row"><span>{escape(FEATURE_LABELS[feature])}</span>'
                    f'<div class="bar-track"><div class="bar-fill" style="width:{width:.1f}%"></div></div>'
                    f'<span class="bar-value">{value:.0%}</span></div>',
                    unsafe_allow_html=True,
                )
        with right:
            st.markdown('<h3 class="insight-title">Riskiest routes</h3>',
                        unsafe_allow_html=True)
            routes = df.groupby(["Origin_Port", "Destination_Port"])[TARGET].agg(
                ["mean", "size"]
            )
            routes = routes[routes["size"] >= 10].sort_values(
                "mean", ascending=False
            ).head(5)
            max_risk = float(routes["mean"].max()) if not routes.empty else 1
            for (from_port, to_port), route in routes.iterrows():
                risk = float(route["mean"])
                width = risk / max_risk * 100 if max_risk else 0
                label = escape(f"{from_port} → {to_port}")
                st.markdown(
                    f'<div class="bar-row"><span>{label}</span>'
                    f'<div class="bar-track"><div class="bar-fill route-fill" style="width:{width:.1f}%"></div></div>'
                    f'<span class="bar-value">{risk:.0%}</span></div>',
                    unsafe_allow_html=True,
                )
        port_count = pd.concat(
            [df["Origin_Port"], df["Destination_Port"]], ignore_index=True
        ).nunique()
        stats = [
            (f"{len(df):,}", "Shipments tracked"),
            (f"{df[TARGET].mean():.0%}", "Delay rate"),
            (f"{accuracy:.0%}", "Model accuracy"),
            (str(port_count), "Ports covered"),
        ]
        stat_html = "".join(
            f'<div class="stat"><div class="stat-number">{value}</div>'
            f'<div class="stat-label">{label}</div></div>'
            for value, label in stats
        )
        st.markdown(f'<div class="stats">{stat_html}</div>', unsafe_allow_html=True)

# Footer and phone navigation
st.markdown("""
<footer class="site-footer">
  <span class="footer-text">ShipSafe · Powered by Random Forest AI</span>
</footer>
<nav class="mobile-nav">
  <a class="active" href="#home">
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3 2 11h2v10h6v-6h4v6h6V11h2L12 3z"/></svg>
    <span>Home</span>
  </a>
  <a href="#check-shipment">
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 3h16v18H4V3zm3 4v2h10V7H7zm0 4v2h10v-2H7zm0 4v2h6v-2H7z"/></svg>
    <span>Check</span>
  </a>
  <a href="#insights">
    <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 19h16v2H2V3h2v16zm3-2V9h3v8H7zm5 0V4h3v13h-3zm5 0v-6h3v6h-3z"/></svg>
    <span>Insights</span>
  </a>
</nav>
""", unsafe_allow_html=True)
