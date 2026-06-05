import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
data= pd.read_csv(r"C:\Users\Astha\OneDrive\Documents\crime_prediction_ML_project\crime_prediction_dataset.csv")

# Load Model
model = joblib.load("crime_model.pkl")

city_encoder = joblib.load("city_encoder.pkl")
area_encoder = joblib.load("area_encoder.pkl")
crime_encoder = joblib.load("crime_encoder.pkl")
weather_encoder = joblib.load("weather_encoder.pkl")
target_encoder = joblib.load("target_encoder.pkl")

# Title
st.title("🚔 Crime Hotspot Prediction System")

# Sidebar
st.sidebar.header("Enter Area Details")

city = st.sidebar.selectbox(
    "City",
    city_encoder.classes_
)

area = st.sidebar.selectbox(
    "Area",
    area_encoder.classes_
)

crime_type = st.sidebar.selectbox(
    "Crime Type",
    crime_encoder.classes_
)

population = st.sidebar.number_input(
    "Population",
    min_value=1000
)

past_crime_count = st.sidebar.number_input(
    "Past Crime Count",
    min_value=0
)

unemployment_rate = st.sidebar.slider(
    "Unemployment Rate",
    0.0,
    100.0
)

police_stations = st.sidebar.number_input(
    "Police Stations",
    min_value=0
)

weather = st.sidebar.selectbox(
    "Weather",
    weather_encoder.classes_
)

month = st.sidebar.slider(
    "Month",
    1,
    12
)

# Predict
if st.sidebar.button("Predict Crime Risk"):

    input_data = pd.DataFrame({

        'city':[city_encoder.transform([city])[0]],
        'area':[area_encoder.transform([area])[0]],
        'crime_type':[crime_encoder.transform([crime_type])[0]],
        'population':[population],
        'past_crime_count':[past_crime_count],
        'unemployment_rate':[unemployment_rate],
        'police_stations':[police_stations],
        'weather':[weather_encoder.transform([weather])[0]],
        'month':[month]

    })

    prediction = model.predict(input_data)

    level = target_encoder.inverse_transform(prediction)[0]

    st.subheader("Prediction Result")

    st.success(f"Crime Level : {level}")

    # ALERT SYSTEM

    if level.lower() == "high":

        st.error(
            "🚨 ALERT! HIGH CRIME HOTSPOT DETECTED"
        )

    elif level.lower() == "medium":

        st.warning(
            "⚠ MEDIUM CRIME RISK"
        )

    else:

        st.success(
            "✅ LOW CRIME RISK"
        )

# Hotspot Analysis

st.subheader("Top Crime Hotspots")

hotspots = data.groupby('area')['crime_count'].sum()

hotspots = hotspots.sort_values(
    ascending=False
).head(10)

st.bar_chart(hotspots)

# Heatmap

st.subheader("Crime Correlation Heatmap")

numeric_data = data.select_dtypes(
    include='number'
)

fig, ax = plt.subplots(figsize=(10,6))

sns.heatmap(
    numeric_data.corr(),
    annot=True,
    ax=ax
)

st.pyplot(fig)

# Dataset Preview

st.subheader("Dataset Preview")

st.dataframe(data.head())

# Crime Map

if 'latitude' in data.columns and 'longitude' in data.columns:

    st.subheader("Crime Hotspot Map")

    st.map(
        data[['latitude','longitude']]
    )