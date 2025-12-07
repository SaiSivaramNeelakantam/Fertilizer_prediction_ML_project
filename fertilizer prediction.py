import pandas as pd
import streamlit as st
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ===============================
# Load Dataset
# ===============================
df = pd.read_csv("fertilizer_new.csv")
fert_ref = pd.read_csv(r"C:\Users\Sai sivaram\OneDrive\Desktop\fertilizer_ml_project\fertilizer_refference_new.csv")

# ===============================
# Fertilizer Model
# ===============================
X_fert = df.drop(columns=["Fertilizer"])
y_fert = df["Fertilizer"]

X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
    X_fert, y_fert, test_size=0.3, random_state=42, stratify=y_fert
)

categorical_cols_f = X_fert.select_dtypes(include="object").columns
numerical_cols_f = X_fert.select_dtypes(include=["float64", "int64"]).columns

num_transformer = Pipeline(steps=[("scaler", StandardScaler())])
cat_transformer = Pipeline(steps=[("encoder", OneHotEncoder(handle_unknown="ignore"))])

preprocessor_f = ColumnTransformer(
    transformers=[
        ("num", num_transformer, numerical_cols_f),
        ("cat", cat_transformer, categorical_cols_f)
    ]
)

fert_pipeline = Pipeline(steps=[("preprocessor", preprocessor_f),
                                ("classifier", RandomForestClassifier(random_state=42))])

fert_pipeline.fit(X_train_f, y_train_f)
fert_acc = accuracy_score(y_test_f, fert_pipeline.predict(X_test_f))

# ===============================
# Fertilizer Recommendations Dictionary
# ===============================
fert_recommendations = {
    "DAP": {
        "use": """🌱 Supplies both nitrogen (N) and phosphorus (P)  
🌱 Ideal for initial crop growth and root development  
🌱 Commonly used at sowing for cereals, pulses, and oilseeds""",
        "precaution": """⚠️ Do not place seeds directly in contact with DAP  
⚠️ Overuse can cause soil pH imbalance  
🧤 Wear gloves and 😷 masks while handling"""
    },
    "Urea": {
        "use": """🌿 High nitrogen (46%) for leafy growth  
🌿 Best for wheat, rice, maize  
🌿 Can also be used as foliar spray for quick greening""",
        "precaution": """☀️ Avoid applying under hot sun  
🌿 Apply in split doses  
🐛 Overuse makes plants weak and pest-prone  
📦 Store away from moisture"""
    },
    "Compost": {
        "use": """🌱 Improves soil structure and fertility  
🌱 Adds organic matter and microbes  
🌱 Suitable for all crops""",
        "precaution": """⚠️ Ensure compost is well-decomposed  
💧 Avoid overuse in waterlogged fields  
🧹 Use clean compost"""
    },
    "Organic Fertilizer": {
        "use": """🌿 Provides slow-release nutrients  
💧 Improves soil water-holding capacity  
🌱 Boosts microbial activity  
🌱 Great for sustainable farming""",
        "precaution": """⚠️ Nutrient concentration is lower than chemical fertilizers  
🧫 Ensure pathogen-free material  
🛠 Handle with clean tools"""
    },
    "Balanced NPK Fertilizer": {
        "use": """⚖️ Balanced nutrients (N, P, K)  
🌾 Good for cereals, vegetables, pulses, fruits""",
        "precaution": """🧪 Choose ratio based on soil test  
⚠️ Overuse can harm microbes and groundwater  
🌱 Avoid applying too close to roots"""
    },
    "Muriate of Potash": {
        "use": """🧂 Provides potassium for fruiting, flowering, and disease resistance  
🌾 Best for sugarcane, potato, cotton, fruits""",
        "precaution": """⚠️ Avoid overuse in chloride-sensitive crops (tobacco, grapes, citrus)  
🧪 Apply based on soil test"""
    },
    "Gypsum": {
        "use": """⚪ Supplies calcium and sulfur  
🌱 Improves soil structure and reduces salinity  
🌾 Used for groundnut, oilseeds, legumes""",
        "precaution": """⚠️ Apply only in recommended dose  
😷 Avoid inhaling dust  
🧤 Wear protective mask and gloves"""
    },
    "Lime": {
        "use": """🪨 Corrects soil acidity  
🌱 Improves nutrient availability and microbial activity  
🌾 Best for tea, paddy, maize""",
        "precaution": """🧪 Apply only based on soil pH test  
⚠️ Excess makes soil alkaline  
🧤 Handle with gloves"""
    },
    "General Purpose Fertilizer": {
        "use": """⚖️ Provides balanced nutrients for multiple crops  
🥕 Suitable for vegetables, gardens, and small farmers""",
        "precaution": """⚠️ Not crop-specific  
📏 Follow recommended dosage  
🌱 Overuse lowers soil fertility"""
    },
    "Water Retaining Fertilizer": {
        "use": """💧 Helps retain moisture in dry soils  
💦 Reduces irrigation frequency  
🌱 Best for drought-prone areas and horticultural crops""",
        "precaution": """⚠️ Mix properly into soil (not just surface)  
💧 Do not overdose → may cause waterlogging  
✅ Use certified products only"""
    }
}


# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="🌱 Fertilizer Recommendation", layout="wide")
st.title("🌾 Fertilizer Recommendation System")

st.sidebar.header("Enter Farm Details")

# User Inputs
Temperature = st.sidebar.number_input("🌡 Temperature (°C)", 10.0, 50.0, 25.0)
Moisture = st.sidebar.slider("💧 Moisture", 0.0, 1.0, 0.6)
Rainfall = st.sidebar.number_input("🌧 Rainfall (mm)", 0.0, 400.0, 100.0)
PH = st.sidebar.slider("⚗️ pH Value", 0.0, 14.0, 6.5)
Nitrogen = st.sidebar.number_input("🧪 Nitrogen", 0.0, 100.0, 60.0)
Phosphorous = st.sidebar.number_input("🧪 Phosphorous", 0.0, 200.0, 70.0)
Potassium = st.sidebar.number_input("🧪 Potassium", 0.0, 200.0, 70.0)
Carbon = st.sidebar.number_input("🌿 Carbon", 0.0, 5.0, 1.5)

Soil = st.sidebar.selectbox("🟤 Soil Type", df["Soil"].unique())
Crop = st.sidebar.selectbox("🌱 Crop (for Fertilizer Prediction)", df["Crop"].unique())

Acres = st.sidebar.number_input("🌾 Number of Acres", 1, 100, 1)

# ===============================
# Prediction + Store in Session
# ===============================
if st.sidebar.button("🔍 Predict Fertilizer"):
    fert_input = pd.DataFrame([{
        "Temperature": Temperature, "Moisture": Moisture, "Rainfall": Rainfall,
        "PH": PH, "Nitrogen": Nitrogen, "Phosphorous": Phosphorous,
        "Potassium": Potassium, "Carbon": Carbon,
        "Soil": Soil, "Crop": Crop
    }])

    fert_pred = fert_pipeline.predict(fert_input)[0]
    st.session_state["fert_pred"] = fert_pred  # store prediction

    st.subheader("✅ Fertilizer Recommendation")
    st.write(f"🌱 Recommended Fertilizer: **{fert_pred}**")
    st.info(f"Model Accuracy → Fertilizer: {fert_acc:.2f}")

# ===============================
# Cost & Guidelines Buttons
# ===============================
if "fert_pred" in st.session_state:
    fert_pred = st.session_state["fert_pred"]

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💰 Calculate Fertilizer Cost") or "show_cost" in st.session_state:
            st.session_state["show_cost"] = True

            fert_amount = fert_ref.loc[fert_ref["Fertilizer"] == fert_pred, "Amount_per_acre_kg"].values[0]
            if "Price_per_kg" in fert_ref.columns:
                default_price = fert_ref.loc[fert_ref["Fertilizer"] == fert_pred, "Price_per_kg"].values[0]
            else:
                default_price = fert_amount

            Price_per_kg = st.number_input("💰 Price per kg of Fertilizer",
                                           min_value=1.0, max_value=500.0,
                                           value=float(default_price))

            total_cost = fert_amount * Acres * Price_per_kg

            st.write(f"📦 Required Amount per Acre: **{fert_amount} kg**")
            st.write(f"💰 Estimated Cost for {Acres} acres: **₹ {total_cost:.2f}**")

    with col2:
        if st.button("📘 Fertilizer Guidelines"):
            if fert_pred in fert_recommendations:
                st.subheader(f"📘 Guidelines for {fert_pred}")
                st.markdown(f"**✅ Use Cases:** {fert_recommendations[fert_pred]['use']}")
                st.markdown(f"**⚠️ Precautions:** {fert_recommendations[fert_pred]['precaution']}")
            else:
                st.warning("No specific guidelines available for this fertilizer.")
