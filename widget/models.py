from django.db import models
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def train_model():
    df_selected = pd.read_csv('Preprocessed_Data.csv')

    X = df_selected[['new_cases', 'new_deaths', 'total_cases', 'total_deaths', 'Rt']]
    y = df_selected['risk_level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train_scaled, y_train)

    with open('model.pkl', 'wb') as model_file:
        pickle.dump(clf, model_file)
    
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)

train_model()
print('model is ready')
