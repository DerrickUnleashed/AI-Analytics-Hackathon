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

from sklearn.preprocessing import StandardScaler, LabelEncoder
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
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    #"SVM": SVC(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "AdaBoost": AdaBoostClassifier(random_state=42),
    "Extra Trees": ExtraTreesClassifier(random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    "LightGBM": LGBMClassifier(random_state=42,verbose=-1),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
}

accuracies = []
max = 0
best_model=''
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    if accuracy > max:
        max = accuracy
        best_model = name
    accuracies.append((name, accuracy))
    print(f"{name} Accuracy: {accuracy:.7f}")


print(f"Best Model : {best_model} Best Accuracy: {max:.7f}")

model_names, accuracy_values = zip(*accuracies)
plt.figure(figsize=(12, 7))
plt.barh(model_names, accuracy_values, color='skyblue')
plt.xlabel("Accuracy")
plt.title("Model Accuracies Comparison")
plt.xlim(0, 1)  # Adjust the x-axis range for better visualization
for index, value in enumerate(accuracy_values):
    plt.text(value, index, f"{value:.5f}", va='center')  # Add accuracy values next to bars
plt.show()
accuracy_df = pd.DataFrame(accuracies, columns=["Model", "Accuracy"])
accuracy_df.to_csv("model_accuracies.csv", index=False)