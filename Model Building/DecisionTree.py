# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 17:41:04 2025

@author: gopia
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Load the preprocessed dataset
dataset = pd.read_csv('labeled_data.csv')  # Replace with your dataset file path

# Step 1: Extract Features and Target Variable
X = dataset.drop('clusters', axis=1)  # Features
y = dataset['clusters']  # Target

# Step 2: Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree classifier
dt = DecisionTreeClassifier(random_state=42)

# Step 3: Train the model
dt.fit(X_train, y_train)

# Make predictions
y_pred_train_dt = dt.predict(X_train)
y_pred_test_dt = dt.predict(X_test)

# Step 4: Evaluate the model
print("Decision Tree - Train Accuracy:", accuracy_score(y_train, y_pred_train_dt))
print("Decision Tree - Test Accuracy:", accuracy_score(y_test, y_pred_test_dt))

print("Classification Report (Decision Tree):")
print(classification_report(y_test, y_pred_test_dt))

print("Confusion Matrix (Decision Tree):")
print(confusion_matrix(y_test, y_pred_test_dt))


# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2', None],
}

# Initialize the DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)

# Grid search
grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best Parameters: {grid_search.best_params_}")

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred_test_best = best_model.predict(X_test)

print("Best Decision Tree - Test Accuracy:", accuracy_score(y_test, y_pred_test_best))
print("Classification Report (Best Decision Tree):")
print(classification_report(y_test, y_pred_test_best))
print("Confusion Matrix (Best Decision Tree):")
print(confusion_matrix(y_test, y_pred_test_best))
