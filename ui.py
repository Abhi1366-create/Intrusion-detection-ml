import streamlit as st
import pandas as pd
import numpy as np
import joblib
import datetime
import random
import os

# check models
required = [
    "models/binary_model.pkl",
    "models/multi_model.pkl",
    "models/preprocessor.pkl",
    "models/label_encoder.pkl"
]

missing = [f for f in required if not os.path.exists(f)]
if missing:
    st.error("models not found run train.py")
    st.stop()

# loading models
model_bin = joblib.load("models/binary_model.pkl")
model_multi = joblib.load("models/multi_model.pkl")
pre = joblib.load("models/preprocessor.pkl")
le = joblib.load("models/label_encoder.pkl")

st.set_page_config(layout="wide")
st.title("intrusion detection dashboard")

if "logs" not in st.session_state:
    st.session_state.logs = []

# manual input
duration = st.sidebar.number_input("duration", 0.0, 10000.0, 0.0)
protocol = st.sidebar.selectbox("protocol_type", ["tcp","udp","icmp"])
service = st.sidebar.selectbox("service", ["http","ftp","smtp","other"])
flag = st.sidebar.selectbox("flag", ["SF","S0","REJ","RSTR","SH"])

def build_row():
    return {
        'duration': duration,
        'protocol_type': protocol,
        'service': service,
        'flag': flag,
        'src_bytes': 0, 'dst_bytes': 0,
        'land': 0, 'wrong_fragment': 0, 'urgent': 0, 'hot': 0,
        'num_failed_logins': 0, 'logged_in': 0, 'num_compromised': 0,
        'root_shell': 0, 'su_attempted': 0, 'num_root': 0,
        'num_file_creations': 0, 'num_shells': 0, 'num_access_files': 0,
        'num_outbound_cmds': 0, 'is_host_login': 0, 'is_guest_login': 0,
        'count': 0, 'srv_count': 0,
        'serror_rate': 0.0, 'srv_serror_rate': 0.0,
        'rerror_rate': 0.0, 'srv_rerror_rate': 0.0,
        'same_srv_rate': 0.0, 'diff_srv_rate': 0.0,
        'srv_diff_host_rate': 0.0,
        'dst_host_count': 0, 'dst_host_srv_count': 0,
        'dst_host_same_srv_rate': 0.0,
        'dst_host_diff_srv_rate': 0.0,
        'dst_host_same_src_port_rate': 0.0,
        'dst_host_srv_diff_host_rate': 0.0,
        'dst_host_serror_rate': 0.0,
        'dst_host_srv_serror_rate': 0.0,
        'dst_host_rerror_rate': 0.0,
        'dst_host_srv_rerror_rate': 0.0
    }

def predict(sample):
    df = pd.DataFrame([sample])
    X = pre.transform(df)
    prob = model_bin.predict_proba(X)[0][1]
    if prob < 0.25:
        return "normal", prob
    probs = model_multi.predict_proba(X)[0]
    idx = np.argmax(probs)
    return le.inverse_transform([idx])[0], probs[idx]

# run manual detection
if st.sidebar.button("run detection"):
    sample = build_row()
    label, conf = predict(sample)
    risk = "HIGH" if conf > 0.8 else "MEDIUM" if conf > 0.5 else "LOW"
    st.session_state.logs.append({
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "type": label,
        "confidence": round(float(conf),3),
        "risk": risk
    })

# simulator
attack = st.sidebar.selectbox("simulate", ["Normal","DoS","Probe","R2L","U2R"])

def sim_sample(label):
    x = build_row()

    if label == "DoS":
        x['count'] = 200
        x['serror_rate'] = 0.9

    if label == "Probe":
        x['count'] = 50
        x['diff_srv_rate'] = 0.8

    if label == "R2L":
        x['num_failed_logins'] = 5
        x['logged_in'] = 0

    if label == "U2R":
        x['num_compromised'] = 3
        x['root_shell'] = 1

    return x

# run simulation
if st.sidebar.button("run simulation"):
    for _ in range(10):
        s = sim_sample(attack)
        label, conf = predict(s)
        risk = "HIGH" if conf > 0.8 else "MEDIUM" if conf > 0.5 else "LOW"
        st.session_state.logs.append({
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "type": label,
            "confidence": round(float(conf),3),
            "risk": risk
        })

# dashboard
st.subheader("logs")

if st.session_state.logs:
    df = pd.DataFrame(st.session_state.logs[::-1])
    st.dataframe(df, use_container_width=True)
    st.bar_chart(df["type"].value_counts())
    st.bar_chart(df["risk"].value_counts())
else:
    st.write("no data")