import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(page_title="Mini IDS Dashboard", layout="wide")
st.title("🛡️ Mini Intrusion Detection System (AI Based)")
st.write("Anomaly-Based Network Traffic Detection using Logistic Regression")

# -------------------------------------------------
# 1. Generate Synthetic Network Traffic
# -------------------------------------------------
def generate_data(n=1000):
    np.random.seed()

    data = []

    for _ in range(n):
        label = np.random.choice([0, 1], p=[0.7, 0.3])

        if label == 0:
            packet_size = np.random.randint(300, 900)
            duration = np.random.uniform(1, 5)
            failed_logins = np.random.randint(0, 2)
            port = np.random.choice([80, 443, 53])
        else:
            packet_size = np.random.randint(50, 2000)
            duration = np.random.uniform(0.1, 1)
            failed_logins = np.random.randint(3, 10)
            port = np.random.choice([23, 4444, 3389])

        data.append([packet_size, duration, failed_logins, port, label])

    df = pd.DataFrame(data, columns=[
        "packet_size",
        "duration",
        "failed_logins",
        "port",
        "label"
    ])

    return df

# -------------------------------------------------
# 2. Train Model
# -------------------------------------------------
def train_model(df):
    X = df[["packet_size", "duration", "failed_logins", "port"]]
    y = df["label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return model, scaler, acc, cm

# -------------------------------------------------
# 3. Generate Data Button
# -------------------------------------------------
if st.button("🔄 Generate New Network Traffic"):
    st.session_state["refresh"] = True

if "refresh" not in st.session_state:
    st.session_state["refresh"] = False

df = generate_data(1000)

model, scaler, accuracy, cm = train_model(df)

# -------------------------------------------------
# 4. Show Model Accuracy
# -------------------------------------------------
st.subheader("📊 Model Performance")
st.success(f"Model Accuracy: {round(accuracy * 100, 2)}%")

# -------------------------------------------------
# 5. Predictions
# -------------------------------------------------
X_scaled_full = scaler.transform(df[["packet_size", "duration", "failed_logins", "port"]])
df["prediction"] = model.predict(X_scaled_full)
df["probability"] = model.predict_proba(X_scaled_full)[:, 1]

normal_count = len(df[df["prediction"] == 0])
attack_count = len(df[df["prediction"] == 1])

# -------------------------------------------------
# 6. Traffic Summary
# -------------------------------------------------
st.subheader("📈 Traffic Summary")

col1, col2 = st.columns(2)
col1.metric("Normal Traffic", normal_count)
col2.metric("Suspicious Traffic", attack_count)

# -------------------------------------------------
# 7. Visualization
# -------------------------------------------------
st.subheader("📊 Traffic Distribution")

fig, ax = plt.subplots()
ax.bar(["Normal", "Suspicious"], [normal_count, attack_count])
ax.set_ylabel("Traffic Count")
st.pyplot(fig)

# -------------------------------------------------
# 8. Confusion Matrix
# -------------------------------------------------
st.subheader("📉 Confusion Matrix")

fig2, ax2 = plt.subplots()
ax2.matshow(cm)
for (i, j), val in np.ndenumerate(cm):
    ax2.text(j, i, val, ha='center', va='center')
ax2.set_xlabel("Predicted")
ax2.set_ylabel("Actual")
st.pyplot(fig2)

# -------------------------------------------------
# 9. Alert System
# -------------------------------------------------
st.subheader("🚨 Alert Status")

attack_ratio = attack_count / len(df)

if attack_ratio > 0.35:
    st.error("⚠️ High Suspicious Network Activity Detected!")
elif attack_ratio > 0.20:
    st.warning("⚠️ Moderate Suspicious Activity")
else:
    st.success("✅ Network Operating Normally")

# -------------------------------------------------
# 10. Show Logs
# -------------------------------------------------
st.subheader("🔎 Sample Traffic Logs")
st.dataframe(df.head(20))
