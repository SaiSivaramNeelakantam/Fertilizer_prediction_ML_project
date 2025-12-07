# 🌾 Fertilizer Recommendation System

A Machine Learning & Streamlit-based app that recommends the **best fertilizer**, estimates **cost**, and provides **usage guidelines** based on soil and environmental conditions.

---

## 🚀 Features

* Fertilizer prediction using Random Forest
* Cost calculation based on acres & price
* Fertilizer usage guidelines
* EDA module with plots (histogram, correlation heatmap, scatter plots)
* Clean Streamlit UI

---

## 📂 Project Files

```
Exploratory Data Analysis.py     → EDA Dashboard
fertilizer prediction.py         → ML Model + Recommendation App
Model_Info.py                    → About / Documentation Page
fertilizer_new.csv               → Training Dataset
fertilizer_refference_new.csv    → Cost & Reference Data
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

Run EDA App:

```bash
streamlit run "Exploratory Data Analysis.py"
```

Run Prediction App:

```bash
streamlit run "fertilizer prediction.py"
```

---

## 🧠 Model

* Preprocessing: StandardScaler + OneHotEncoder
* Algorithm: RandomForestClassifier
* Output: Recommended fertilizer label

---

## 🌱 Purpose

* Help farmers choose correct fertilizer
* Reduce cost & avoid overuse
* Support sustainable farming

---

## ⭐ Contribution

Pull requests are welcome!

---


