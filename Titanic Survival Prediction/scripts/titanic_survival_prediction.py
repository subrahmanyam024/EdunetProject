import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# 🏗️ Step 1: Load and Preprocess Data
train_df = pd.read_csv("../data/train.csv")
test_df = pd.read_csv("../data/test.csv")

# Fill missing values using median imputation
train_df.fillna({"Age": train_df["Age"].median(), "Fare": train_df["Fare"].median()}, inplace=True)
test_df.fillna({"Age": test_df["Age"].median(), "Fare": test_df["Fare"].median()}, inplace=True)

# Encode categorical features (Sex)
encoder = LabelEncoder()
train_df["Sex"] = encoder.fit_transform(train_df["Sex"])
test_df["Sex"] = encoder.transform(test_df["Sex"])

# ✅ Fix: Define FamilySize feature properly
train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1

# 🚢 Step 2: Feature Engineering
# Extract Title from Name and Group Rare Titles
train_df["Title"] = train_df["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
test_df["Title"] = test_df["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())

# Title Mapping: Grouping Similar Titles
title_mapping = {
    "Mr": "AdultMale", "Miss": "YoungFemale", "Mrs": "AdultFemale", "Master": "Child",
    "Dr": "Professional", "Rev": "Professional", "Col": "Military",
    "Major": "Military", "Countess": "Noble", "Capt": "Military", "Lady": "Noble",
    "Jonkheer": "Noble", "Don": "Noble", "Sir": "Noble"
}
train_df["Title"] = train_df["Title"].map(title_mapping)
test_df["Title"] = test_df["Title"].map(title_mapping)

# Encoding Title feature
train_df["Title"] = encoder.fit_transform(train_df["Title"])
test_df["Title"] = encoder.transform(test_df["Title"])

# 📊 Step 2.5: Visualizing `Embarked` & Checking Correlation Before Removal
plt.figure(figsize=(6, 4))
sns.barplot(x=train_df["Embarked"], y=train_df["Survived"])
plt.title("Survival Rates by Embarked Location")
plt.xlabel("Embarked")
plt.ylabel("Survival Rate")
plt.show()

# Compute correlation of `Embarked` with survival
embarked_corr = train_df["Embarked"].map({"C": 1, "Q": 2, "S": 3}).corr(train_df["Survived"])
print(f"Correlation of `Embarked` with Survival: {embarked_corr:.2f}")

# Decision to remove Embarked based on correlation
if abs(embarked_corr) < 0.2:
    print("📌 The correlation is weak, meaning `Embarked` does not significantly affect survival rates.")
    print("🔹 Removing `Embarked` from the feature set to simplify the model without losing accuracy.")

# ✅ Removing `Embarked` due to **low correlation with survival**
selected_features = ["Pclass", "Sex", "Age", "FamilySize", "Title"]

# 🚀 Step 3: Train Models & Evaluate Overfitting
# Function to compute train & validation accuracy
def evaluate_model(model, X_train, y_train):
    train_acc = accuracy_score(y_train, model.predict(X_train))
    val_acc = np.mean(cross_val_score(model, X_train, y_train, cv=5))  # Cross-validation accuracy
    return train_acc, val_acc

# Train Logistic Regression
lr_model = LogisticRegression(max_iter=500, random_state=42)
lr_model.fit(train_df[selected_features], train_df["Survived"])
lr_train_acc, lr_val_acc = evaluate_model(lr_model, train_df[selected_features], train_df["Survived"])

# Train Decision Tree
dt_model = DecisionTreeClassifier(max_depth=6, random_state=42)
dt_model.fit(train_df[selected_features], train_df["Survived"])
dt_train_acc, dt_val_acc = evaluate_model(dt_model, train_df[selected_features], train_df["Survived"])

# Train Random Forest
rf_model = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf_model.fit(train_df[selected_features], train_df["Survived"])
rf_train_acc, rf_val_acc = evaluate_model(rf_model, train_df[selected_features], train_df["Survived"])

# Train XGBoost
xgb_model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, min_child_weight=3)
xgb_model.fit(train_df[selected_features], train_df["Survived"])
xgb_train_acc, xgb_val_acc = evaluate_model(xgb_model, train_df[selected_features], train_df["Survived"])

# 🏆 Step 3.5: Compare Model Performance & Select Best
accuracies = {
    "Logistic Regression": (lr_train_acc, lr_val_acc),
    "Decision Tree": (dt_train_acc, dt_val_acc),
    "Random Forest": (rf_train_acc, rf_val_acc),
    "XGBoost": (xgb_train_acc, xgb_val_acc)
}

print("\n📊 Model Performance Comparison:")
for model_name, (train_acc, val_acc) in accuracies.items():
    print(f"{model_name}: Train Accuracy = {train_acc:.2f}, Validation Accuracy = {val_acc:.2f}")

best_model = max(accuracies, key=lambda x: accuracies[x][1])
best_train_acc, best_val_acc = accuracies[best_model]

print(f"\n✅ Best Model Selected: {best_model}")
print(f"🔹 This model has the highest validation accuracy ({best_val_acc:.2f}), meaning it generalizes best.")

# 📂 Step 4: Generate Final Submission File
results_dir = os.path.join(os.getcwd(), "results")
os.makedirs(results_dir, exist_ok=True)

final_predictions = {
    "Logistic Regression": lr_model.predict(test_df[selected_features]),
    "Decision Tree": dt_model.predict(test_df[selected_features]),
    "Random Forest": rf_model.predict(test_df[selected_features]),
    "XGBoost": xgb_model.predict(test_df[selected_features])
}[best_model]

submission_df = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": final_predictions
})

submission_df.to_csv(os.path.join(results_dir, "titanic_final_predictions.csv"), index=False)
print("✅ Final submission file saved successfully in:", results_dir)

# 🖼️ **Final Visualization: Feature Importance**
sns.barplot(x=selected_features, y=xgb_model.feature_importances_)
plt.title("Feature Importance (XGBoost)")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.show()
