# The necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv", sep=";")
    df = df.dropna(axis=1, how='all')
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    df['day_of_week'] = df['Date'].dt.day_name()

    if not df['Date'].dt.hour.isnull().all():
        df['hour'] = df['Date'].dt.hour
        df['peak_hour'] = df['hour'].apply(lambda h: 1 if (7 <= h <= 10) or (16 <= h <= 19) else 0)
        df['hour'] = df['hour'].fillna(0)
        df['peak_hour'] = df['peak_hour'].fillna(0)

    # Ensure numeric conversion
    columns_to_convert = [
        'Average delay of all trains at departure',
        'Number of scheduled trains',
        'Number of cancelled trains',
        'Number of trains delayed at departure',
        'Average delay of late trains at departure'
    ]
    for col in columns_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    df['is_delayed'] = df['Average delay of all trains at departure'].apply(lambda x: 1 if x > 5 else 0)
    df = df.drop_duplicates().reset_index(drop=True)
    
    return df

df = load_data()

# Load trained model
model = joblib.load("best_model.pkl")

# Encode stations
station_names = df['Departure station'].dropna().astype(str).unique()
station_encoder = {name: idx for idx, name in enumerate(sorted(station_names))}
inv_station_encoder = {v: k for k, v in station_encoder.items()}

df['Departure station'] = df['Departure station'].map(station_encoder)

# Sidebar filters
st.sidebar.header("🔍 Filters")
stations = list(station_encoder.keys())
selected_station = st.sidebar.selectbox("Station", ["All"] + stations)
date_range = st.sidebar.date_input("Period", [])

if selected_station != "All":
    df = df[df['Departure station'] == station_encoder[selected_station]]

if len(date_range) == 2:
    df = df[(df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))]

# Title
st.title("🚆 Train Delay Dashboard")

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("⏱️ Average delay", f"{df['Average delay of all trains at departure'].mean():.2f} min")
col2.metric("❌ Cancellations", int(df['Number of cancelled trains'].sum()))
punctuality = 100 * (1 - df['is_delayed'].mean())
col3.metric("✅ Punctuality", f"{punctuality:.1f}%")

# Distribution of Delays
st.subheader("📊 Distribution of Delays")
fig1, ax1 = plt.subplots()
sns.histplot(df['Average delay of all trains at departure'], kde=True, bins=30, ax=ax1)
st.pyplot(fig1)

# Boxplot
fig2, ax2 = plt.subplots()
sns.boxplot(x=df['Average delay of all trains at departure'], ax=ax2)
st.pyplot(fig2)

# Stations with the most delays
st.subheader("🏙️ Stations with the most delays")
top_delays = df.groupby('Departure station')['Average delay of all trains at departure'].mean().sort_values(ascending=False).head(10)
top_delays.index = [inv_station_encoder[i] for i in top_delays.index]
fig3, ax3 = plt.subplots()
top_delays.plot(kind='bar', ax=ax3, color='skyblue')
plt.xticks(rotation=45)
st.pyplot(fig3)

# Delays by hour
st.subheader("🕒 Delays by time of day")
hourly = df.groupby('hour')['Average delay of all trains at departure'].mean()
fig4, ax4 = plt.subplots()
hourly.plot(marker='o', color='orange', ax=ax4)
st.pyplot(fig4)

# Correlation heatmap
st.subheader("🔥 Correlation between variables")
numeric_df = df.select_dtypes(include=np.number)
corr = numeric_df.corr()
fig5, ax5 = plt.subplots(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax5)
st.pyplot(fig5)

# Delay Forecast
st.subheader("🤖 Delay Forecast")

with st.form("predict_form"):
    user_station = st.selectbox("Departure station", stations)
    user_hour = st.slider("Hora do dia", 0, 23, 8)
    peak = 1 if (7 <= user_hour <= 10) or (16 <= user_hour <= 19) else 0
    scheduled = st.number_input("Scheduled trains", 0)
    cancelled = st.number_input("Canceled trains", 0)
    delayed = st.number_input("Delayed trains", 0)
    avg_late_delay = st.number_input("Average delay of delayed trains", 0.0)

    submitted = st.form_submit_button("🔮 Predict")
    if submitted:
        input_data = pd.DataFrame([[
            station_encoder[user_station],
            user_hour,
            peak,
            scheduled,
            cancelled,
            delayed,
            avg_late_delay
        ]], columns=[
            'Departure station', 'hour', 'peak_hour',
            'Number of scheduled trains', 'Number of cancelled trains',
            'Number of trains delayed at departure',
            'Average delay of late trains at departure'
        ])

        try:
            prediction = model.predict(input_data)[0]

            if hasattr(model, "predict_proba"):
                probability = model.predict_proba(input_data)[0][1]
                st.success(f"Probability of delay: {probability:.2%}")
                st.info("🚨 Likely delayed" if prediction == 1 else "🟢 Likely on time")
            else:
                st.success(f"⏱️ Estimated average delay: {prediction:.2f} minutes")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
