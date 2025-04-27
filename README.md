# Lab Assignment 5.2 - Ensemble Learning Techniques

# 📚 Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier, AdaBoostClassifier, VotingClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ⚙️ Load Dataset
# (Replace 'your_data.csv' with your actual dataset)
df = pd.read_csv('your_data.csv')

# 👀 Explore the data
print(df.head())
print(df.info())
print(df.describe())

# Assume the last column is the target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# ✂️ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📊 Standardize the features (important for KNN and Logistic Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🚀 Base Models
log_clf = LogisticRegression()
dt_clf = DecisionTreeClassifier()
knn_clf = KNeighborsClassifier()

# 🔥 Stacking Classifier
stack_model = StackingClassifier(
    estimators=[
        ('lr', log_clf),
        ('dt', dt_clf),
        ('knn', knn_clf)
    ],
    final_estimator=LogisticRegression()
)

# 🪄 AdaBoost Classifier
adaboost_model = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

# 🗳️ Voting Classifier
voting_hard = VotingClassifier(
    estimators=[
        ('lr', log_clf),
        ('dt', dt_clf),
        ('knn', knn_clf)
    ],
    voting='hard'
)

voting_soft = VotingClassifier(
    estimators=[
        ('lr', log_clf),
        ('dt', dt_clf),
        ('knn', knn_clf)
    ],
    voting='soft'
)

# 🏋️‍♂️ Fit and Evaluate All Models

models = {
    "Stacking Classifier": stack_model,
    "AdaBoost Classifier": adaboost_model,
    "Voting Classifier (Hard)": voting_hard,
    "Voting Classifier (Soft)": voting_soft
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"🔵 {name}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("-" * 50)
