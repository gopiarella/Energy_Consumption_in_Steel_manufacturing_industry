# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 15:49:13 2025

@author: gopia
"""

# Import necessary libraries 
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# Load the preprocessed dataset
dataset = pd.read_csv('labeled_data.csv')

# Step 1: Extract Features and Target Variable
X = dataset.drop('clusters', axis=1)  # Features
y = dataset['clusters']  # Target

# Step 2: Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the XGBoost classifier
xgb = XGBClassifier(learning_rate = 0.01, random_state=42)
xgb.fit(X_train, y_train)

# Make predictions
y_pred_train_xgb = xgb.predict(X_train)
y_pred_test_xgb = xgb.predict(X_test)

# Evaluate the model
print("XGBoost - Train Accuracy:", accuracy_score(y_train, y_pred_train_xgb))
print("XGBoost - Test Accuracy:", accuracy_score(y_test, y_pred_test_xgb))

print("Classification Report (XGBoost):")
print(classification_report(y_test, y_pred_test_xgb))

print("Confusion Matrix (XGBoost):")
print(confusion_matrix(y_test, y_pred_test_xgb))


from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

# Define model
xgboost= XGBClassifier(random_state=42)

# Define the parameter grid
param_grid_xgb = {
    'n_estimators': [100, 200, 300],      # Number of boosting rounds
    'learning_rate': [0.01, 0.1, 0.3],    # Step size
    'max_depth': [3, 6, 9],                # Max depth of the tree
    'subsample': [0.7, 0.8, 1.0],          # Fraction of samples used
    'colsample_bytree': [0.7, 0.8, 1.0],  # Fraction of features used
    'gamma': [0, 0.1, 0.2]                # Minimum loss reduction to make a further partition
}

# Initialize GridSearchCV
grid_search_xgb = GridSearchCV(estimator=xgboost, param_grid=param_grid_xgb, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit the model
grid_search_xgb.fit(X_train, y_train)

# Get the best parameters and score
print("Best parameters for XGBoost: ", grid_search_xgb.best_params_)
print("Best cross-validation score for XGBoost: {:.2f}".format(grid_search_xgb.best_score_))

# Best model
best_xgb_model = grid_search_xgb.best_estimator_

# Test the best model
y_pred_train_xgb = best_xgb_model.predict(X_train)
y_pred_test_xgb = best_xgb_model.predict(X_test)
train_accuracy_xgb = accuracy_score(y_train, y_pred_train_xgb)
test_accuracy_xgb = accuracy_score(y_test, y_pred_test_xgb)
print(f"Test Accuracy for XGBoost after tuning: {test_accuracy_xgb:.4f}")
print(f"Train Accuracy for XGBoost after tuning: {train_accuracy_xgb:.4f}")

X_train.columns






# Import necessary libraries for ROC curve
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Get predicted probabilities for the positive class (1)
y_pred_prob = xgb.predict_proba(X_test)[:, 1]

# Compute ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Diagonal line (chance level)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

