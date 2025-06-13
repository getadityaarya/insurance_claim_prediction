
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, confusion_matrix
from xgboost import XGBClassifier

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    X = df.drop('claim', axis=1)
    y = df['claim']
    return train_test_split(X, y, test_size=0.2, random_state=42)

def train_models(X_train, y_train):
    log_reg = LogisticRegression(max_iter=1000)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    log_reg.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    return log_reg, xgb

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return precision, recall, cm
