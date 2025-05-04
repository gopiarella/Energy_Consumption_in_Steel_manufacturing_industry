# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 16:45:39 2025

@author: gopia
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 1.0],
}

# Grid search
grid_search = GridSearchCV(estimator=ada_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best model after tuning
best_model = grid_search.best_estimator_

# Evaluate the best model
train_accuracy = best_model.score(X_train, y_train)
test_accuracy = best_model.score(X_test, y_test)
y_pred = best_model.predict(X_test)

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))



# Load the preprocessed dataset
dataset = pd.read_csv('labeled_data.csv')  # Replace with your dataset file path

# Step 1: Extract Features and Target Variable
X = dataset.drop('clusters', axis=1)  # Features
y = dataset['clusters']  # Target

# Step 2: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Initialize the AdaBoost classifier
ada_model = AdaBoostClassifier(random_state=42)

# Step 4: Train the AdaBoost model
ada_model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
y_pred = ada_model.predict(X_test)

# Step 6: Calculate and display metrics
print("Accuracy Score: ", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Optionally, calculate training accuracy
train_accuracy = ada_model.score(X_train, y_train)
test_accuracy = ada_model.score(X_test, y_test)

print(f"\nTraining Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")


# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 1.0],
}

# Grid search
grid_search = GridSearchCV(estimator=ada_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best model after tuning
best_model = grid_search.best_estimator_

# Evaluate the best model
train_accuracy = best_model.score(X_train, y_train)
test_accuracy = best_model.score(X_test, y_test)
y_pred = best_model.predict(X_test)

print(f"Best Hyperparameters: {grid_search.best_params_}")
print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

