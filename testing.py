import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

numeric_imputer = joblib.load("numeric_imputer.pkl")
scaler = joblib.load("scaler.pkl")
obj_imputer = joblib.load("obj_imputer.pkl")
encoder = joblib.load("encoder.pkl")
model = joblib.load("model.pkl")

df_test = pd.read_csv("cybersecurity_intrusion_data.csv")


X_test = df_test.drop('attack_detected', axis=1)
y_test = df_test['attack_detected']


objects = X_test.drop(columns=['network_packet_size', 'login_attempts', 'session_duration','ip_reputation_score', 'failed_logins', 'unusual_time_access'])
numeric = X_test.drop(columns=['session_id', 'protocol_type', 'encryption_used', 'browser_type'])
objects = objects.drop('session_id', axis=1)

numeric_imputed = numeric_imputer.transform(numeric)
numeric_final = scaler.transform(numeric_imputed)
numeric_df = pd.DataFrame(numeric_final, columns=numeric.columns)

obj_imputed = obj_imputer.transform(objects)
obj_encoded = encoder.transform(obj_imputed)
obj_df = pd.DataFrame(obj_encoded.toarray(), columns = ['protocol_type_ICMP', 'protocol_type_TCP', 'protocol_type_UDP','encryption_used_AES', 'encryption_used_DES', 'browser_type_Chrome','browser_type_Edge', 'browser_type_Firefox', 'browser_type_Safari','browser_type_Unknown'])

X_final = pd.concat([obj_df, numeric_df], axis=1)

y_pred = model.predict(X_final)
print("Test Accuracy: {:.1%}".format(accuracy_score(y_test, y_pred)))
