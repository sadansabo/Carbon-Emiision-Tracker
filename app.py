import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Sample dataset
data = {
    "City": ["Lagos", "Kano", "Port Harcourt", "Kaduna", "Ibadan"],
    "Industrial_Activity": [90, 70, 85, 60, 50],
    "Energy_Consumption": [1200, 900, 1100, 800, 600],
    "Emissions": [950, 720, 880, 640, 480]
}
df = pd.DataFrame(data)

# Train simple model
X = df[["Industrial_Activity", "Energy_Consumption"]]
y = df["Emissions"]
model = LinearRegression().fit(X, y)

st.title("🌍 Carbon Emission Tracker Agent (Nigeria)")
st.write("Tracks and predicts industrial CO₂ emissions in Nigerian cities.")

# Show dataset
st.subheader("📊 Sample Dataset")
st.dataframe(df)

# Prediction form
st.subheader("🔮 Predict Emissions")
city = st.selectbox("Select City", df["City"])
activity = st.slider("Industrial Activity Index", 0, 100, 70)
energy = st.slider("Energy Consumption (MWh)", 500, 2000, 1000)

pred = model.predict([[activity, energy]])[0]
st.metric(label=f"Predicted Emissions for {city}", value=f"{pred:.2f} tons CO₂")

# Real-time simulation
st.subheader("📈 Real-Time Emission Updates")
chart_placeholder = st.empty()

sim_data = []
for i in range(10):
    activity_sim = np.random.randint(40, 100)
    energy_sim = np.random.randint(500, 2000)
    pred_sim = model.predict([[activity_sim, energy_sim]])[0]
    sim_data.append(pred_sim)

    fig, ax = plt.subplots()
    ax.plot(sim_data, marker="o")
    ax.set_ylabel("Predicted CO₂ (tons)")
    ax.set_xlabel("Time Step")
    ax.set_title("Real-Time Emission Updates")
    chart_placeholder.pyplot(fig)
    time.sleep(1)
