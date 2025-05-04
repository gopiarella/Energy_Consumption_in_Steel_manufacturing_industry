# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 17:58:38 2025

@author: gopia
"""

import pandas as pd
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Define the parameter grid for alpha (regularization strength)
param_grid = {
    'alpha': [0.1, 1, 10, 100, 1000]
}

# Initialize the Ridge Classifier
ridge = RidgeClassifier()

# Grid search
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best Parameters: {grid_search.best_params_}")

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred_test_best = best_model.predict(X_test)

print("Best Ridge Classifier - Test Accuracy:", accuracy_score(y_test, y_pred_test_best))
print("Classification Report (Best Ridge Classifier):")
print(classification_report(y_test, y_pred_test_best))
print("Confusion Matrix (Best Ridge Classifier):")
print(confusion_matrix(y_test, y_pred_test_best))


# Load the preprocessed dataset
dataset = pd.read_csv('labeled_data.csv')  # Replace with your dataset file path

# Step 1: Extract Features and Target Variable
X = dataset.drop('clusters', axis=1)  # Features
y = dataset['clusters']  # Target

# Step 2: Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Ridge Classifier
ridge = RidgeClassifier(random_state=42)

# Step 3: Train the model
ridge.fit(X_train, y_train)

# Make predictions
y_pred_train_ridge = ridge.predict(X_train)
y_pred_test_ridge = ridge.predict(X_test)

# Step 4: Evaluate the model
print("Ridge Classifier - Train Accuracy:", accuracy_score(y_train, y_pred_train_ridge))
print("Ridge Classifier - Test Accuracy:", accuracy_score(y_test, y_pred_test_ridge))

print("Classification Report (Ridge Classifier):")
print(classification_report(y_test, y_pred_test_ridge))

print("Confusion Matrix (Ridge Classifier):")
print(confusion_matrix(y_test, y_pred_test_ridge))


# Define the parameter grid for alpha (regularization strength)
param_grid = {
    'alpha': [0.1, 1, 10, 100, 1000]
}

# Initialize the Ridge Classifier
ridge = RidgeClassifier()

# Grid search
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best Parameters: {grid_search.best_params_}")

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred_test_best = best_model.predict(X_test)

print("Best Ridge Classifier - Test Accuracy:", accuracy_score(y_test, y_pred_test_best))
print("Classification Report (Best Ridge Classifier):")
print(classification_report(y_test, y_pred_test_best))
print("Confusion Matrix (Best Ridge Classifier):")
print(confusion_matrix(y_test, y_pred_test_best))
