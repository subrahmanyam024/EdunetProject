
import pandas as pd

# Load the dataset
train_df = pd.read_csv("../data/train.csv")

# Check missing values
print("Missing values:\n", train_df.isnull().sum())

# Fill missing values
train_df["Age"].fillna(train_df["Age"].median(), inplace=True)
train_df["Embarked"].fillna(train_df["Embarked"].mode()[0], inplace=True)
train_df.drop(columns=["Cabin"], inplace=True)  # Dropping Cabin due to too many missing values

# Verify cleanup
print("Missing values after cleanup:\n", train_df.isnull().sum())

# Extract title from names
train_df["Title"] = train_df["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())

# Show unique titles
print(train_df["Title"].value_counts())

# Group similar titles & encode them numerically
title_mapping = {
    "Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5,
    "Rev": 6, "Col": 7, "Major": 8, "Mlle": 9, "Mme": 10, 
    "Countess": 11, "Capt": 12, "Lady": 13, "Jonkheer": 14, "Don": 15, "Sir": 16
}
train_df["Title"] = train_df["Title"].map(title_mapping)

# Fill any missing titles with a default value (e.g., 0)
train_df["Title"].fillna(0, inplace=True)


train_df["FamilySize"] = train_df["SibSp"] + train_df["Parch"] + 1
#sibsp => sibling/spouse count and parch=> parents/childrens count

from sklearn.preprocessing import LabelEncoder

# Encode Sex column
sex_encoder = LabelEncoder()
train_df["Sex"] = sex_encoder.fit_transform(train_df["Sex"])

# Encode Embarked column
embarked_encoder = LabelEncoder()
train_df["Embarked"] = embarked_encoder.fit_transform(train_df["Embarked"])

# Select relevant features
features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title"]
X = train_df[features]  # Features
y = train_df["Survived"]  # Target

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Train model
model = LogisticRegression()
model.fit(X, y)

# Predict on the training data
train_predictions = model.predict(X)

# Evaluate accuracy
train_accuracy = accuracy_score(y, train_predictions)
print(f"Training Accuracy: {train_accuracy:.2f}")

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y, train_predictions)
print("Confusion Matrix:\n", cm)

#Look at Precision & Recall Instead of just accuracy, we should check precision (how well we predict survival) and recall (how well we identify all survivors):

from sklearn.metrics import classification_report

print("Model Performance:\n", classification_report(y, train_predictions))

#Try Another Model (Decision Tree or Random Forest) Logistic Regression is solid, but let's see if Decision Tree or Random Forest performs better:
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier()
dt_model.fit(X, y)

dt_predictions = dt_model.predict(X)
dt_accuracy = accuracy_score(y, dt_predictions)

print(f"Decision Tree Accuracy: {dt_accuracy:.2f}")

#Checking for Overfitting
#Try evaluating your model on test.csv (unseen data) instead of just training data:
# Load test data
test_df = pd.read_csv("../data/test.csv")

# Apply same preprocessing steps to test data (missing values, encoding)
# Ensure test_df has the same features as X_train
test_df["Title"] = test_df["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
test_df["Title"] = test_df["Title"].map(title_mapping).fillna(0)
test_df["FamilySize"] = test_df["SibSp"] + test_df["Parch"] + 1
test_df["Sex"] = sex_encoder.transform(test_df["Sex"])
test_df["Embarked"] = embarked_encoder.transform(test_df["Embarked"])
test_df["Age"].fillna(train_df["Age"].median(), inplace=True)
test_df["Fare"].fillna(train_df["Fare"].median(), inplace=True)
test_df.drop(columns=["Cabin"], inplace=True)

# Select same features
X_test = test_df[features]

# Predict on test data
test_predictions = dt_model.predict(X_test)

# Print predictions
print("Test Predictions:\n", test_predictions)


# Assuming actual test labels are in a file, otherwise we skip this step
test_labels = pd.read_csv("../data/gender_submission.csv")["Survived"]
test_accuracy = accuracy_score(test_labels, test_predictions)
print(f"Test Accuracy: {test_accuracy:.2f}")

#the above model is overfitting because accuarcy of train data we got 98% while 
#evaluating the same model with test data we got 80% so the model is overfitted so we are updating the model

#Let’s apply some techniques to improve generalization
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(max_depth=5)  # Limit depth to 5
dt_model.fit(X, y)

dt_predictions = dt_model.predict(X_test)
dt_test_accuracy = accuracy_score(test_labels, dt_predictions)
print(f"New Test Accuracy: {dt_test_accuracy:.2f}")

#see if Random Forest performs even better
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X, y)

rf_predictions = rf_model.predict(X_test)
rf_test_accuracy = accuracy_score(test_labels, rf_predictions)

print(f"Random Forest Test Accuracy: {rf_test_accuracy:.2f}")

#Get Feature Importance from the Decision Tree
import matplotlib.pyplot as plt

# Extract feature importance from the Decision Tree model
feature_importance = dt_model.feature_importances_

# Plot feature importance
plt.figure(figsize=(8, 5))
plt.barh(features, feature_importance, color="skyblue")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance in Decision Tree")
plt.show()


# Extract feature importance from the Random Forest model
rf_importance = rf_model.feature_importances_

# Plot feature importance for Random Forest
plt.figure(figsize=(8, 5))
plt.barh(features, rf_importance, color="lightgreen")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.title("Feature Importance in Random Forest")
plt.show()


#Optimizing Title Handling & Feature Selection

#1. Refining Title Categories
#Instead of keeping every unique title, let's group them
# Simplify titles into broad categories
title_mapping = {
    "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Child",
    "Dr": "Professional", "Rev": "Professional", "Col": "Officer",
    "Major": "Officer", "Mlle": "Miss", "Mme": "Mrs",
    "Countess": "Noble", "Capt": "Officer", "Lady": "Noble",
    "Jonkheer": "Noble", "Don": "Noble", "Sir": "Noble"
}

train_df["Title"] = train_df["Title"].map(title_mapping)
test_df["Title"] = test_df["Title"].map(title_mapping)

# Encode the new Title feature
title_encoder = LabelEncoder()
train_df["Title"] = title_encoder.fit_transform(train_df["Title"])
test_df["Title"] = title_encoder.transform(test_df["Title"])

#Remove Less Important Features
#Since Embarked and Fare had low importance, let's exclude them
selected_features = ["Pclass", "Sex", "Age", "FamilySize", "Title"]
X_train = train_df[selected_features]
X_test = test_df[selected_features]

#Train an XGBoost Model
#XGBoost often outperforms Decision Trees & Random Forest due to its boosting technique
from xgboost import XGBClassifier

# Train XGBoost model
xgb_model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1)
xgb_model.fit(X_train, y)

# Predict on test data
xgb_predictions = xgb_model.predict(X_test)

# Evaluate accuracy
xgb_test_accuracy = accuracy_score(test_labels, xgb_predictions)
print(f"XGBoost Test Accuracy: {xgb_test_accuracy:.2f}")


#Optimized XGBoost Training
from xgboost import XGBClassifier

# Fine-tuned XGBoost model
xgb_model = XGBClassifier(
    n_estimators=300,       # More trees for better learning
    max_depth=6,            # Slightly deeper trees
    learning_rate=0.05,     # Lower learning rate for smoother adjustments
    min_child_weight=3,     # Helps prevent overfitting
    subsample=0.8,          # Uses 80% of data per tree (reduces variance)
    colsample_bytree=0.8,   # Uses 80% of features per tree
    random_state=42
)

# Train and evaluate
xgb_model.fit(X_train, y)
xgb_predictions = xgb_model.predict(X_test)
xgb_test_accuracy = accuracy_score(test_labels, xgb_predictions)

print(f"Fine-Tuned XGBoost Test Accuracy: {xgb_test_accuracy:.2f}")


#Analyze Misclassified Cases
#1. Identify Wrong Predictions
# Find misclassified cases by comparing predictions with test labels
misclassified = test_df[test_labels != xgb_predictions]

# Show incorrect predictions
print("Misclassified Samples:\n", misclassified[["PassengerId", "Pclass", "Sex", "Age", "Title", "FamilySize"]])

#Finding Error Patterns
print(misclassified.groupby("Title")["PassengerId"].count())
print(misclassified.groupby("Pclass")["PassengerId"].count())
print(misclassified.groupby("Sex")["PassengerId"].count())


#Creating Interaction Features
#Introduce Age-Sex Interaction
#Young females had better survival rates—this feature helps reflect that
train_df["Age*Sex"] = train_df["Age"] * train_df["Sex"]
test_df["Age*Sex"] = test_df["Age"] * test_df["Sex"]
#Introduce Pclass-Fare Interaction
#High-ticket price passengers often had better survival rates—this feature enhances that link
train_df["Pclass*Fare"] = train_df["Pclass"] * train_df["Fare"]
test_df["Pclass*Fare"] = test_df["Pclass"] * test_df["Fare"]
#Train XGBoost with Interaction Features
from xgboost import XGBClassifier

selected_features = ["Pclass", "Sex", "Age", "FamilySize", "Title", "Age*Sex", "Pclass*Fare"]

xgb_model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, min_child_weight=3)
xgb_model.fit(train_df[selected_features], y)

xgb_predictions = xgb_model.predict(test_df[selected_features])
xgb_test_accuracy = accuracy_score(test_labels, xgb_predictions)
print(f"Refined XGBoost Test Accuracy (With Interaction Features): {xgb_test_accuracy:.2f}")

#Feature Selection Test
#Train Model Without Age*Sex
selected_features = ["Pclass", "Sex", "Age", "FamilySize", "Title", "Pclass*Fare"]

xgb_model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05)
xgb_model.fit(train_df[selected_features], y)

xgb_predictions = xgb_model.predict(test_df[selected_features])
xgb_test_accuracy = accuracy_score(test_labels, xgb_predictions)
print(f"Test Accuracy (Without Age*Sex): {xgb_test_accuracy:.2f}")

#Train Model Without Pclass*Fare
selected_features = ["Pclass", "Sex", "Age", "FamilySize", "Title", "Age*Sex"]

xgb_model.fit(train_df[selected_features], y)
xgb_predictions = xgb_model.predict(test_df[selected_features])
xgb_test_accuracy = accuracy_score(test_labels, xgb_predictions)
print(f"Test Accuracy (Without Pclass*Fare): {xgb_test_accuracy:.2f}")


#Train the Final Model with Optimized Features
from xgboost import XGBClassifier

# Define the best feature set
final_features = ["Pclass", "Sex", "Age", "FamilySize", "Title"]

# Train the final XGBoost model
xgb_final_model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, min_child_weight=3)
xgb_final_model.fit(train_df[final_features], y)

# Generate final predictions
final_predictions = xgb_final_model.predict(test_df[final_features])
#Save Predictions in Submission Format
# Create DataFrame for final submission
submission_df = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    "Survived": final_predictions
})

import os

# Ensure the 'results' folder exists
os.makedirs("results", exist_ok=True)

# Save to CSV for submission
submission_df.to_csv("results/titanic_final_predictions.csv", index=False)

print("✅ Final submission file saved successfully!")
