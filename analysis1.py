import pandas as pd
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, 
    roc_curve, precision_recall_curve, log_loss
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

df = pd.read_csv("BalancedDataset.csv")

df1 = pd.read_csv("dataset.csv")
df1 = df1.drop(columns=['nameOrig','nameDest','isFlaggedFraud'])


scaler = StandardScaler()
numeric_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
df1[numeric_cols] = scaler.fit_transform(df1[numeric_cols])

# Validate transaction consistency: check if balance differences match the amount
def validate_transaction(row):
    source_diff = row['oldbalanceOrg'] - row['newbalanceOrig']
    dest_diff = row['newbalanceDest'] - row['oldbalanceDest']

    # Check if balance difference matches the amount
    if row['type'] in ['TRANSFER', 'CASH_OUT']:
        return (source_diff == row['amount']) and (dest_diff == -row['amount'])
    elif row['type'] in ['PAYMENT', 'DEBIT']:
        return source_diff == row['amount']
    elif row['type'] in ['CASH_IN', 'CASH_O']:
        return dest_diff == row['amount']
    else:
        return False

# Add validation column
df['isValidTransaction'] = df.apply(validate_transaction, axis=1)
df1['isValidTransaction'] = df1.apply(validate_transaction, axis=1)
# Encode transaction type
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])
df1['type'] = le.fit_transform(df1['type'])


# Split features and target
X_train = df.drop(['isFraud'], axis=1)
y_train = df['isFraud']
X_test = df1.drop(['isFraud'], axis=1)
y_test = df1['isFraud']


# Normalize numerical values
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

# Split data into training and testing sets

# Train a RandomForest model


best_model = GaussianNB(var_smoothing=5.3366992312063123e-05)
best_model.fit(X_train, y_train)
joblib.dump(best_model, "best_naive_bayes_model.pkl")
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))
print("Log Loss:", log_loss(y_test, y_prob))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- ROC Curve ---")
fpr, tpr, _ = roc_curve(y_test, y_prob)
print("False Positive Rate:", fpr)
print("True Positive Rate:", tpr)

print("\n--- Precision-Recall Curve ---")
precision, recall, _ = precision_recall_curve(y_test, y_prob)
print("Precision:", precision)
print("Recall:", recall)