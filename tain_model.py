import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import joblib
#%%
df = pd.read_csv('cybersecurity_intrusion_data.csv')

X_train, X_test, y_train, y_test = train_test_split(df.drop('attack_detected',axis=1),df['attack_detected'],test_size=0.3)


objects = X_train.drop(columns=['network_packet_size','login_attempts','session_duration','ip_reputation_score', 'failed_logins','unusual_time_access'])
numeric = X_train.drop(columns=['session_id','protocol_type','encryption_used','browser_type'])
objects= objects.drop('session_id',axis=1)


numeric_imputer = SimpleImputer(strategy="mean")
scaler = StandardScaler()
numeric_imputed = numeric_imputer.fit_transform(numeric)
numeric_final = scaler.fit_transform(numeric_imputed)
numeric_imputed_df = pd.DataFrame(numeric_final,columns=numeric.columns)

obj_enc = OneHotEncoder()
obj_imputer = SimpleImputer(strategy="most_frequent")
obj_imputed = obj_imputer.fit_transform(objects)
obj_encoding = obj_enc.fit_transform(obj_imputed)
obj_encoeded = pd.DataFrame(obj_encoding.toarray(),columns = ['protocol_type_ICMP', 'protocol_type_TCP', 'protocol_type_UDP','encryption_used_AES', 'encryption_used_DES', 'browser_type_Chrome','browser_type_Edge', 'browser_type_Firefox', 'browser_type_Safari','browser_type_Unknown'])

X_train_final = pd.concat([obj_encoeded, numeric_imputed_df], axis=1)


X_train_final.info()


print("\n")
print("Logistic")
logistic = LogisticRegression(max_iter=5000)
logistic.fit(X_train_final,y_train)
y_predeict = logistic.predict(X_train_final)
print("Accuracy Score is: {:.1%}".format(accuracy_score(y_train,y_predeict)))
cross = cross_val_score(logistic,X_train_final,y_train,cv = 10)
print("Cross mean score: {:.1%}".format(cross.mean()))

print("\n")
print("SVM")
svc = SVC()
svc.fit(X_train_final,y_train)
y_predeict = svc.predict(X_train_final)
print("Accuracy Score is: {:.1%}".format(accuracy_score(y_train,y_predeict)))
cross = cross_val_score(svc,X_train_final,y_train,cv = 10)
print("Cross mean score: {:.1%}".format(cross.mean()))

print("\n")
print("Decision Tree")
tree = DecisionTreeClassifier()
tree.fit(X_train_final,y_train)
y_predeict = tree.predict(X_train_final)
print("Accuracy Score is: {:.1%}".format(accuracy_score(y_train,y_predeict)))
cross = cross_val_score(tree,X_train_final,y_train,cv = 10)
print("Cross mean score: {:.1%}".format(cross.mean()))

print("\n")
print("Decision Forest")
forest = RandomForestClassifier(n_estimators=300, max_depth=20, n_jobs=-1)
forest.fit(X_train_final,y_train)
y_predeict = forest.predict(X_train_final)
print("Accuracy Score is: {:.1%}".format(accuracy_score(y_train,y_predeict)))
cross = cross_val_score(forest,X_train_final,y_train,cv = 10)
print("Cross mean score: {:.1%}".format(cross.mean()))

param_grid = {
    'n_estimators': [100, 200, 300,400],
    'max_depth': [10, 20, 30,40],
    'min_samples_split': [2, 5, 10,15]
}

gridforest = RandomForestClassifier(n_jobs=-1)

grid_search = GridSearchCV(
    estimator=gridforest,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
print("\n")
print("Grid Search")
grid_search.fit(X_train_final, y_train)
print("Best Parameters:", grid_search.best_params_)
print("Best Cross Validation Score: {:.1%}".format(grid_search.best_score_))

best_forest = grid_search.best_estimator_
best_forest.fit(X_train_final, y_train)

train = accuracy_score(y_train, best_forest.predict(X_train_final))
print("Training Accuracy: {:.1%}".format(train))


joblib.dump(numeric_imputer, "numeric_imputer.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(obj_imputer, "obj_imputer.pkl")
joblib.dump(obj_enc, "encoder.pkl")
joblib.dump(best_forest, "model.pkl")
