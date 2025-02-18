import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

# Load house price prediction model
model = joblib.load("models/house_price_model.pkl")

# Load dataset for visualization and analysis
data = pd.read_csv(r"C:\Users\mrnde\house-price-prediction\data\train.csv") 

# Encode categorical features
for col in data.select_dtypes(include=['object']).columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Select numerical columns
numerical_data = data.select_dtypes(include=['number'])

# Compute correlation with SalePrice
correlation = numerical_data.corr()["SalePrice"].sort_values(ascending=False)

# Convert correlation to DataFrame for visualization
corr_df = correlation.reset_index().rename(columns={"index": "Feature", "SalePrice": "Correlation"})

# Streamlit UI
st.set_page_config(page_title="House Price Prediction", layout="wide")

st.title("🏡 House Price Prediction & Analysis Dashboard")

# ---- Sidebar for Input Features ----
st.sidebar.header("Enter House Details")

OverallQual = st.sidebar.number_input("Overall Quality (1-10)", min_value=1, max_value=10, value=7)
GrLivArea = st.sidebar.number_input("Above Ground Living Area (sq ft)", min_value=500, max_value=5000, value=2000)
GarageCars = st.sidebar.number_input("Garage Capacity", min_value=0, max_value=4, value=2)
TotalBsmtSF = st.sidebar.number_input("Total Basement Area (sq ft)", min_value=0, max_value=3000, value=1000)
FullBath = st.sidebar.number_input("Number of Full Bathrooms", min_value=0, max_value=4, value=2)
YearBuilt = st.sidebar.number_input("Year Built", min_value=1900, max_value=2025, value=2005)

# ---- Prediction Button ----
if st.sidebar.button("Predict Price"):
    input_data = pd.DataFrame([[OverallQual, GrLivArea, GarageCars, TotalBsmtSF, FullBath, YearBuilt]], 
                              columns=["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF", "FullBath", "YearBuilt"])
    
    prediction = model.predict(input_data)[0]
    st.sidebar.success(f"🏠 Predicted House Price: **${prediction:,.2f}**")

# ---- Data Visualization Section ----
st.subheader("📊 Data Analysis & Visualization")

# Dynamic Line Chart for Feature Correlation
fig_corr = px.line(corr_df, x="Feature", y="Correlation", title="Feature Correlation with Sale Price", markers=True)
st.plotly_chart(fig_corr, use_container_width=True)

# ---- Feature Trend Selection ----
st.subheader("📈 Feature Trend Analysis")

# X-axis and Y-axis selection
x_axis = st.selectbox("Select X-axis:", numerical_data.columns, index=0)
y_axis = st.selectbox("Select Y-axis:", numerical_data.columns, index=1)

# ---- Data Range Selection ----
st.write("### 📏 Select Data Range for Visualization")
min_value = st.slider("Minimum Value", int(data[x_axis].min()), int(data[x_axis].max()), int(data[x_axis].min()))
max_value = st.slider("Maximum Value", int(data[x_axis].min()), int(data[x_axis].max()), int(data[x_axis].max()))

filtered_data = data[(data[x_axis] >= min_value) & (data[x_axis] <= max_value)]

# ---- Chart Type Selection ----
chart_type = st.radio("Choose Visualization Type:", ["Line Chart", "Scatter Plot", "Bar Chart", "Pie Chart"])

# ---- Generate Selected Chart ----
if chart_type == "Line Chart":
    fig = px.line(filtered_data, x=x_axis, y=y_axis, title=f"Trend of {y_axis} vs {x_axis}")
elif chart_type == "Scatter Plot":
    fig = px.scatter(filtered_data, x=x_axis, y=y_axis, title=f"Scatter Plot of {y_axis} vs {x_axis}")
elif chart_type == "Bar Chart":
    fig = px.bar(filtered_data, x=x_axis, y=y_axis, title=f"Bar Chart of {y_axis} vs {x_axis}")
elif chart_type == "Pie Chart":
    fig = px.pie(filtered_data, names=x_axis, values=y_axis, title=f"Pie Chart of {y_axis} vs {x_axis}")

st.plotly_chart(fig, use_container_width=True)

# ---- Display Insights ----
st.markdown("### 🔍 Insights:")
st.write("- Features with **high positive correlation** contribute more to increasing house prices.")
st.write("- Features with **negative correlation** decrease house prices.")
st.write("- Use the **dropdowns** to dynamically change the X and Y axes.")
st.write("- Adjust the **data range** slider to focus on specific values.")

# ---- Display Raw Data ----
if st.checkbox("Show Raw Data"):
    st.write(filtered_data)

st.write("🔹 **Built with Python, Streamlit, and Plotly** | 🚀 **Interactive & Dynamic**")
