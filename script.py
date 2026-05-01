import os
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

from xgboost import XGBClassifier

# check dataset
if not os.path.exists("data/KDDTrain+.txt") or not os.path.exists("data/KDDTest+.txt"):
    print("dataset not found in data folder")
    exit()

DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# loading data
train_path = os.path.join(DATA_DIR, "KDDTrain+.txt")
test_path = os.path.join(DATA_DIR, "KDDTest+.txt")

cols = [
    'duration','protocol_type','service','flag','src_bytes','dst_bytes',
    'land','wrong_fragment','urgent','hot','num_failed_logins',
    'logged_in','num_compromised','root_shell','su_attempted','num_root',
    'num_file_creations','num_shells','num_access_files',
    'num_outbound_cmds','is_host_login','is_guest_login','count',
    'srv_count','serror_rate','srv_serror_rate','rerror_rate',
    'srv_rerror_rate','same_srv_rate','diff_srv_rate',
    'srv_diff_host_rate','dst_host_count','dst_host_srv_count',
    'dst_host_same_srv_rate','dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate','dst_host_srv_diff_host_rate',
    'dst_host_serror_rate','dst_host_srv_serror_rate',
    'dst_host_rerror_rate','dst_host_srv_rerror_rate','class','difficulty'
]

train_df = pd.read_csv(train_path, header=None, names=cols)
test_df = pd.read_csv(test_path, header=None, names=cols)

# cleaning columns
train_df.drop('difficulty', axis=1, inplace=True, errors='ignore')
test_df.drop('difficulty', axis=1, inplace=True, errors='ignore')

X_train = train_df.drop('class', axis=1)
X_test = test_df.drop('class', axis=1)

# mapping attack types
attack_map = {
    'back':'dos','land':'dos','neptune':'dos','pod':'dos','smurf':'dos','teardrop':'dos',
    'apache2':'dos','udpstorm':'dos','processtable':'dos','mailbomb':'dos',
    'satan':'probe','ipsweep':'probe','nmap':'probe','portsweep':'probe','mscan':'probe','saint':'probe',
    'guess_passwd':'r2l','ftp_write':'r2l','imap':'r2l','phf':'r2l','multihop':'r2l',
    'warezmaster':'r2l','warezclient':'r2l','spy':'r2l','xlock':'r2l','xsnoop':'r2l',
    'snmpguess':'r2l','snmpgetattack':'r2l','httptunnel':'r2l','sendmail':'r2l','named':'r2l',
    'buffer_overflow':'u2r','loadmodule':'u2r','rootkit':'u2r','perl':'u2r',
    'sqlattack':'u2r','xterm':'u2r','ps':'u2r'
}

y_train_multi = train_df['class'].map(attack_map).fillna('normal')
y_test_multi = test_df['class'].map(attack_map).fillna('normal')

# binary labels
y_train_binary = (y_train_multi != 'normal').astype(int)
y_test_binary = (y_test_multi != 'normal').astype(int)

# preprocessing
cat = ['protocol_type','service','flag']
num = [c for c in X_train.columns if c not in cat]

pre = ColumnTransformer([
    ('num', StandardScaler(), num),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat)
])

X_train_p = pre.fit_transform(X_train)
X_test_p = pre.transform(X_test)

# stage 1 model
w_bin = compute_class_weight('balanced', classes=np.unique(y_train_binary), y=y_train_binary)
w_dict = dict(enumerate(w_bin))
w_dict[1] *= 3.5
sw_bin = np.array([w_dict[i] for i in y_train_binary])

model_bin = XGBClassifier(
    n_estimators=700,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=1,
    reg_alpha=1,
    reg_lambda=2,
    eval_metric='logloss',
    n_jobs=-1
)

model_bin.fit(X_train_p, y_train_binary, sample_weight=sw_bin)

# prediction
proba = model_bin.predict_proba(X_test_p)[:, 1]
pred_bin = (proba >= 0.25).astype(int)

print("binary results")
print(accuracy_score(y_test_binary, pred_bin))
print(classification_report(y_test_binary, pred_bin))

# stage 2 model
mask_train = y_train_binary == 1
mask_test = y_test_binary == 1

X_train_attack = X_train_p[mask_train]
X_test_attack = X_test_p[mask_test]

y_train_attack = y_train_multi[mask_train]
y_test_attack = y_test_multi[mask_test]

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train_attack)
y_test_enc = le.transform(y_test_attack)

w_multi = compute_class_weight('balanced', classes=np.unique(y_train_enc), y=y_train_enc)
sw_multi = np.array([w_multi[i] for i in y_train_enc])

model_multi = XGBClassifier(
    n_estimators=700,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric='mlogloss',
    n_jobs=-1
)

model_multi.fit(X_train_attack, y_train_enc, sample_weight=sw_multi)

pred_multi = model_multi.predict(X_test_attack)

print("attack classification")
print(classification_report(y_test_enc, pred_multi, target_names=le.classes_))

# saving models
joblib.dump(model_bin, "models/binary_model.pkl")
joblib.dump(model_multi, "models/multi_model.pkl")
joblib.dump(pre, "models/preprocessor.pkl")
joblib.dump(le, "models/label_encoder.pkl")

print("model saved")