import streamlit as st

# ===============================
# 🔹 Background Image CSS
# ===============================
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://foodprint.org/wp-content/uploads/2018/10/GettyImages-907966126_optimized.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0); /* Transparent header */
}
[data-testid="stSidebar"] {
    background-color: rgba(140, 140, 140, 0.8); /* Semi-transparent sidebar */
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# ===============================
# About Section
# ===============================
st.title("Fertilizer Recommendation Model")

# 🌾 Problem Statement
st.header("🌾 Problem Statement")
st.markdown("""
Farmers face challenges in fertilizer management due to:  
1️⃣ ❌ Wrong fertilizer choice leading to **low crop yield**.  
2️⃣ 💸 Excess usage causing **higher input cost**.  
3️⃣ 🌍 Over-application harming **soil health & environment**.  
4️⃣ 🤔 Lack of awareness about **fertilizer-specific precautions**.  
""")

# 🎯 Objective
st.header("🎯 Objective")
st.markdown("""
The aim of this model is to:  
1️⃣ 🌱 **Predict the most suitable fertilizer** for given soil and crop.  
2️⃣ 📊 **Estimate required amount & cost** for selected acreage.  
3️⃣ ⚠️ Provide **safety guidelines** for handling fertilizers.  
4️⃣ 🌍 Support **sustainable farming practices**.  
""")

# ✅ What this Model Will Help
st.header("✅ What this Model Will Help")
st.markdown("""
1️⃣ 🌾 Recommend the **right fertilizer** for better yield.  
2️⃣ 💰 Help farmers **plan costs & optimize usage**.  
3️⃣ 🧑‍🌾 Share **practical precautions** for safer application.  
4️⃣ 🛡️ Protect **soil health & reduce environmental damage**.  
5️⃣ 📈 Empower farmers to make **data-driven decisions**.  
""")
