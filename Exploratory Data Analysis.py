# ==========================================
# Exploratory Data Analysis (EDA) App
# ==========================================
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import statsmodels.api as sm

# ==========================================
# Streamlit Page Config
# ==========================================
st.set_page_config(page_title="Exploratory Data Analysis", layout="wide")
st.title("📊 Exploratory Data Analysis App")

# ==========================================
# Load Dataset (Fixed CSV File)
# ==========================================
# Replace this with your dataset path
df = pd.read_csv("fertilizer_new.csv")

# Dataset Preview
st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())
st.write("🔢 Shape of dataset:", df.shape)

# Dataset Info
if st.checkbox("Show Dataset Info (df.info)"):
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [df[col].dtype for col in df.columns],
        "Non-Null Count": [df[col].count() for col in df.columns],
        "Missing Values": [df[col].isnull().sum() for col in df.columns]
    })
    st.dataframe(info_df)

# Identify numeric and categorical columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

# ==========================================
# Extra: Missing Values Heatmap
# ==========================================
if st.checkbox("Show Missing Values Heatmap"):
    fig = px.imshow(df.isnull(), aspect="auto", color_continuous_scale="Reds", title="Missing Values Heatmap")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# Tabs for Analysis
# ==========================================
tab1, tab2, tab3 = st.tabs(["🔵 Univariate Analysis", "🟠 Bivariate Analysis", "🟢 Multivariate Analysis"])

# --------------------------
# Univariate Analysis
# --------------------------
with tab1:
    st.header("🔵 Univariate Analysis")

    column = st.selectbox("Choose a column for Univariate Analysis", df.columns, key="uni_col")

    if column in numeric_cols:
        st.write("📈 Histogram (Distribution of Numeric Data)")
        fig = px.histogram(df, x=column, nbins=20, title=f"Histogram of {column}")
        st.plotly_chart(fig, use_container_width=True)

        st.write("📈 KDE Density Plot")
        fig3 = px.histogram(df, x=column, nbins=30, marginal="violin", histnorm="probability density",
                            title=f"KDE Density Plot of {column}")
        st.plotly_chart(fig3, use_container_width=True)

        st.write("📊 Box Plot (Outliers & Spread)")
        fig2 = px.box(df, y=column, title=f"Box Plot of {column}")
        st.plotly_chart(fig2, use_container_width=True)

        st.write("📑 Summary Statistics")
        st.dataframe(df[column].describe())

        st.markdown(f"""
        - **Mean:** {df[column].mean():.2f}  
        - **Median:** {df[column].median():.2f}  
        - **Minimum:** {df[column].min()}  
        - **Maximum:** {df[column].max()}  
        """)

    elif column in categorical_cols:
        df_counts = df[column].value_counts().reset_index()
        df_counts.columns = [column, 'count']

        st.write("📊 Bar Chart of Categories")
        fig = px.bar(df_counts, x=column, y='count', title=f"Bar Chart of {column}")
        st.plotly_chart(fig, use_container_width=True)

        st.write("📑 Category Counts")
        st.dataframe(df[column].value_counts())

    else:
        st.warning("⚠ Column type not supported for univariate analysis.")

# --------------------------
# Bivariate Analysis
# --------------------------
with tab2:
    st.header("🟠 Bivariate Analysis")

    col_x = st.selectbox("Choose X-axis column", df.columns, key="xcol")
    col_y = st.selectbox("Choose Y-axis column", df.columns, key="ycol")

    if col_x == col_y:
        st.warning("⚠ Choose different X and Y columns")

    # Case 1: Both Numeric
    elif col_x in numeric_cols and col_y in numeric_cols:
        st.write("📈 Scatter Plot (Numeric vs Numeric)")
        fig = px.scatter(df, x=col_x, y=col_y, trendline="ols", title=f"Scatter Plot of {col_x} vs {col_y}")
        st.plotly_chart(fig, use_container_width=True)

        st.write("🔗 Correlation")
        corr = df[[col_x, col_y]].corr().iloc[0, 1]
        st.write(f"Correlation between **{col_x}** and **{col_y}**: {corr:.2f}")

    # Case 2: Categorical vs Numeric
    elif col_x in categorical_cols and col_y in numeric_cols:
        st.write("📊 Box Plot (Categorical vs Numeric)")
        fig = px.box(df, x=col_x, y=col_y, title=f"{col_y} distribution across {col_x}")
        st.plotly_chart(fig, use_container_width=True)
        st.write("📑 Group Statistics")
        st.dataframe(df.groupby(col_x)[col_y].describe())

    elif col_y in categorical_cols and col_x in numeric_cols:
        st.write("📊 Box Plot (Categorical vs Numeric)")
        fig = px.box(df, x=col_y, y=col_x, title=f"{col_x} distribution across {col_y}")
        st.plotly_chart(fig, use_container_width=True)
        st.write("📑 Group Statistics")
        st.dataframe(df.groupby(col_y)[col_x].describe())

    # Case 3: Categorical vs Categorical
    elif col_x in categorical_cols and col_y in categorical_cols:
        st.write("📊 Crosstab (Categorical vs Categorical)")
        cross_tab = pd.crosstab(df[col_x], df[col_y])
        st.dataframe(cross_tab)

        st.write("📈 Heatmap of Crosstab")
        fig = px.imshow(cross_tab, text_auto=True, title=f"Heatmap of {col_x} vs {col_y}")
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.warning("⚠ Unsupported combination of columns.")

# --------------------------
# Multivariate Analysis
# --------------------------
with tab3:
    st.header("🟢 Multivariate Analysis")

    if len(numeric_cols) >= 2 and len(categorical_cols) >= 1:
        x_col = st.selectbox("Select X-axis (numeric)", numeric_cols, key="hue_x")
        y_col = st.selectbox("Select Y-axis (numeric)", numeric_cols, key="hue_y")
        hue_col = st.selectbox("Select Hue (categorical)", categorical_cols, key="hue_cat")

        fig_hue = px.scatter(
            df, x=x_col, y=y_col, color=hue_col, symbol=hue_col,
            title=f"Scatter Plot of {x_col} vs {y_col} colored by {hue_col}"
        )
        st.plotly_chart(fig_hue, use_container_width=True)
    else:
        st.warning("⚠ Need at least 2 numeric columns and 1 categorical column for multivariate analysis.")

# ==========================================
# Extra: Correlation Heatmap
# ==========================================
st.subheader("📌 Correlation Heatmap (Numeric Features)")
if len(numeric_cols) > 1:
    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# Download Option
# ==========================================
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("💾 Download Processed CSV", csv, "processed_data.csv", "text/csv")
