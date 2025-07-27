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

warnings.filterwarnings('ignore')
writer = SummaryWriter("runs/model_accuracies")

df = pd.read_csv("testing.csv")
# Drop irrelevant columns
df = df.drop(['nameOrig', 'nameDest'], axis=1)

# Encode categorical feature 'type'
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# Split features and target
X = df.drop(['isFraud', 'isFlaggedFraud'], axis=1)
y = df['isFraud']

# Normalize numerical values
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a RandomForest model
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "KNN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(random_state=42),
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
    #writer.add_scalar("Accuracy/" + name, accuracy)
    print(f"{name} Accuracy: {accuracy:.7f}")

print(f"Best Model : {best_model} Best Accuracy: {max:.7f}")
writer.close()

model_names, accuracy_values = zip(*accuracies)
plt.figure(figsize=(12, 7))
plt.barh(model_names, accuracy_values, color='skyblue')
plt.xlabel("Accuracy")
plt.title("Model Accuracies Comparison")
plt.xlim(0.999, 1)  # Adjust the x-axis range for better visualization
for index, value in enumerate(accuracy_values):
    plt.text(value, index, f"{value:.5f}", va='center')  # Add accuracy values next to bars
plt.show()
accuracy_df = pd.DataFrame(accuracies, columns=["Model", "Accuracy"])
accuracy_df.to_csv("model_accuracies.csv", index=False)