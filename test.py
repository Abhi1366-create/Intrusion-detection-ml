import pandas as pd
import numpy as np
import joblib

# loading models
model_bin = joblib.load("models/binary_model.pkl")
model_multi = joblib.load("models/multi_model.pkl")
pre = joblib.load("models/preprocessor.pkl")
le = joblib.load("models/label_encoder.pkl")

# build sample
def build(base):
    row = {
        'duration':0,'protocol_type':'tcp','service':'http','flag':'SF',
        'src_bytes':0,'dst_bytes':0,'land':0,'wrong_fragment':0,'urgent':0,'hot':0,
        'num_failed_logins':0,'logged_in':0,'num_compromised':0,'root_shell':0,
        'su_attempted':0,'num_root':0,'num_file_creations':0,'num_shells':0,
        'num_access_files':0,'num_outbound_cmds':0,'is_host_login':0,'is_guest_login':0,
        'count':0,'srv_count':0,'serror_rate':0,'srv_serror_rate':0,'rerror_rate':0,
        'srv_rerror_rate':0,'same_srv_rate':0,'diff_srv_rate':0,'srv_diff_host_rate':0,
        'dst_host_count':0,'dst_host_srv_count':0,'dst_host_same_srv_rate':0,
        'dst_host_diff_srv_rate':0,'dst_host_same_src_port_rate':0,
        'dst_host_srv_diff_host_rate':0,'dst_host_serror_rate':0,
        'dst_host_srv_serror_rate':0,'dst_host_rerror_rate':0,
        'dst_host_srv_rerror_rate':0
    }
    row.update(base)
    return row

# prediction
def predict(sample):
    X = pre.transform(pd.DataFrame([sample]))
    prob = model_bin.predict_proba(X)[0][1]
    if prob < 0.25:
        return "normal", prob
    probs = model_multi.predict_proba(X)[0]
    idx = np.argmax(probs)
    return le.inverse_transform([idx])[0], probs[idx]

tests = [
    {'duration':0,'protocol_type':'tcp','service':'http','flag':'SF'},
    {'duration':1000,'protocol_type':'tcp','service':'ftp','flag':'REJ'},
    {'duration':500,'protocol_type':'udp','service':'other','flag':'S0'}
]

# running tests
for t in tests:
    s = build(t)
    label, conf = predict(s)
    print(t, "->", label, round(conf,3))