import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ─────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Tanzania Fish Price Predictor",
    page_icon="🐟",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0277BD, #00695C);
        color: white; padding: 22px 25px;
        border-radius: 12px; margin-bottom: 25px; text-align: center;
    }
    .main-header h1 { font-size: 2rem; margin: 0; }
    .main-header p  { margin: 6px 0 0 0; opacity: 0.88; font-size: 1rem; }
    .result-box {
        background: linear-gradient(135deg, #E8F5E9, #C8E6C9);
        border: 2px solid #4CAF50; border-radius: 12px;
        padding: 22px; text-align: center; margin-top: 18px;
    }
    .result-box h2 { color: #2E7D32; font-size: 2.2rem; margin: 0; }
    .result-box p  { color: #388E3C; margin: 4px 0 0 0; font-size: 1rem; }
    .info-box {
        background: #E3F2FD; border-left: 5px solid #1976D2;
        border-radius: 6px; padding: 12px 16px; margin-bottom: 16px;
    }
    .tag-ocean {
        background: #E0F7FA; color: #00695C;
        padding: 2px 10px; border-radius: 20px;
        font-size: 0.8rem; font-weight: bold;
    }
    .tag-fresh {
        background: #E8EAF6; color: #283593;
        padding: 2px 10px; border-radius: 20px;
        font-size: 0.8rem; font-weight: bold;
    }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🐟 Tanzania Fish Price Predictor</h1>
    <p>Ocean, Sea & Freshwater Fish — ML-powered price estimation</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load Model & Encoders
# ─────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model.pkl",    "rb") as f: model    = pickle.load(f)
    with open("encoders.pkl", "rb") as f: encoders = pickle.load(f)
    return model, encoders

try:
    model, encoders = load_artifacts()
    st.sidebar.success("✅ Model loaded successfully")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# ─────────────────────────────────────────────
# Species grouped by source
# ─────────────────────────────────────────────
freshwater_species = ['Carp','Catfish','Dagaa','Electric Catfish',
                      'Lungfish','Nile Perch','Nile Tilapia','Sardine','Tilapia']
ocean_species      = ['Barracuda','Crab','Grouper','Kingfish','Lobster','Mackerel',
                      'Octopus','Parrotfish','Prawns','Red Mullet','Red Snapper',
                      'Sailfish','Sea Bass','Sea Bream','Squid','Swordfish',
                      'Tuna','Yellowfin Tuna']

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📖 About")
    st.markdown("""
This app predicts fish prices in **TZS** using a Decision Tree model
trained on **800 records** from Tanzania's fish markets.

**🌊 Ocean/Sea Species (18):**  
Tuna, Lobster, Prawns, Kingfish, Swordfish, Sailfish,
Grouper, Red Snapper, Barracuda, Octopus, Squid, Crab,
Mackerel, Sea Bass, Sea Bream, Parrotfish, Red Mullet,
Yellowfin Tuna

**🏞️ Freshwater Species (9):**  
Nile Perch, Tilapia, Nile Tilapia, Dagaa, Catfish,
Sardine, Lungfish, Electric Catfish, Carp
""")
    st.divider()
    st.caption("ML Project · Tanzania · 2026")

# ─────────────────────────────────────────────
# Info banner
# ─────────────────────────────────────────────
st.markdown("""
<div class="info-box">
    📌 Select a fish species and fill in the details below,
    then click <strong>Predict Price</strong>.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────
st.subheader("🔧 Enter Fish Details")

col1, col2 = st.columns(2)

with col1:
    source_type = st.radio(
        "🌊 Fish Source",
        options=["Ocean/Sea", "Freshwater"],
        horizontal=True
    )

    if source_type == "Ocean/Sea":
        species_options = ocean_species
    else:
        species_options = freshwater_species

    species = st.selectbox("🐠 Fish Species", options=sorted(species_options))

    region = st.selectbox("📍 Region", options=list(encoders['region'].classes_))
    season = st.selectbox("🌦 Season",  options=list(encoders['season'].classes_))
    market_type   = st.selectbox("🏪 Market Type",   options=list(encoders['market_type'].classes_))
    quality_grade = st.selectbox("⭐ Quality Grade", options=list(encoders['quality_grade'].classes_))

with col2:
    weight_kg = st.number_input(
        "⚖️ Weight (kg)", min_value=0.3, max_value=20.0, value=3.0, step=0.5)
    freshness_days = st.slider(
        "📅 Days Since Catch", min_value=0, max_value=7, value=1,
        help="0 = freshly caught, 7 = 7 days old")
    distance_to_market_km = st.number_input(
        "🚚 Distance to Market (km)", min_value=5, max_value=400, value=50, step=5)
    quantity_kg = st.number_input(
        "📦 Quantity (kg)", min_value=5.0, max_value=600.0, value=100.0, step=10.0)

    # Show expected price range hint
    st.divider()
    if source_type == "Ocean/Sea":
        st.info("🌊 Ocean/Sea fish generally command **higher prices** due to export demand.")
    else:
        st.info("🏞️ Freshwater fish are widely traded in **local & wholesale** markets.")

st.divider()

# ─────────────────────────────────────────────
# Predict Button
# ─────────────────────────────────────────────
if st.button("🔮 Predict Fish Price", use_container_width=True, type="primary"):

    input_dict = {
        'species':               encoders['species'].transform([species])[0],
        'source_type':           encoders['source_type'].transform([source_type])[0],
        'region':                encoders['region'].transform([region])[0],
        'season':                encoders['season'].transform([season])[0],
        'market_type':           encoders['market_type'].transform([market_type])[0],
        'quality_grade':         encoders['quality_grade'].transform([quality_grade])[0],
        'weight_kg':             weight_kg,
        'freshness_days':        freshness_days,
        'distance_to_market_km': distance_to_market_km,
        'quantity_kg':           quantity_kg,
    }

    input_df = pd.DataFrame([input_dict])
    predicted_price = max(1000, model.predict(input_df)[0])

    tag = "🌊 Ocean/Sea Fish" if source_type == "Ocean/Sea" else "🏞️ Freshwater Fish"

    st.markdown(f"""
    <div class="result-box">
        <p>💰 Estimated Fish Price &nbsp;·&nbsp; {tag}</p>
        <h2>TZS {predicted_price:,.2f}</h2>
        <p>≈ USD {predicted_price/2500:.2f} &nbsp;|&nbsp; ≈ EUR {predicted_price/2700:.2f}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Prediction Summary")
    summary_df = pd.DataFrame({
        "Feature": ["Species", "Source Type", "Region", "Season", "Market Type",
                    "Quality Grade", "Weight (kg)", "Freshness (days)",
                    "Distance (km)", "Quantity (kg)"],
        "Value":   [species, source_type, region, season, market_type,
                    quality_grade, f"{weight_kg} kg", f"{freshness_days} days",
                    f"{distance_to_market_km} km", f"{quantity_kg} kg"]
    })
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.info(
        f"📊 Estimated price range: "
        f"**TZS {predicted_price*0.9:,.0f}** – **TZS {predicted_price*1.1:,.0f}** "
        f"(±10% market variation)"
    )

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.divider()
st.caption("🐟 Tanzania Fish Price Predictor · 27 Species · Streamlit & Scikit-learn · 2026")
