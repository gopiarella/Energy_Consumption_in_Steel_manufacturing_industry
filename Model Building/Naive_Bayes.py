# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 17:48:29 2025

@author: gopia
"""

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the preprocessed dataset
dataset = pd.read_csv('labeled_data.csv')  # Replace with your dataset file path

# Step 1: Extract Features and Target Variable
X = dataset.drop('clusters', axis=1)  # Features
y = dataset['clusters']  # Target

# Step 2: Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
nb = GaussianNB()

# Step 3: Train the model
nb.fit(X_train, y_train)

# Make predictions
y_pred_train_nb = nb.predict(X_train)
y_pred_test_nb = nb.predict(X_test)

# Step 4: Evaluate the model
print("Naive Bayes - Train Accuracy:", accuracy_score(y_train, y_pred_train_nb))
print("Naive Bayes - Test Accuracy:", accuracy_score(y_test, y_pred_test_nb))

print("Classification Report (Naive Bayes):")
print(classification_report(y_test, y_pred_test_nb))

print("Confusion Matrix (Naive Bayes):")
print(confusion_matrix(y_test, y_pred_test_nb))


# Define the parameter grid
param_grid = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5],  # Smoothing for variance
}

# Initialize the Naive Bayes classifier
nb = GaussianNB()

# Grid search
grid_search = GridSearchCV(estimator=nb, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best Parameters: {grid_search.best_params_}")

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred_test_best = best_model.predict(X_test)

print("Best Naive Bayes - Test Accuracy:", accuracy_score(y_test, y_pred_test_best))
print("Classification Report (Best Naive Bayes):")
print(classification_report(y_test, y_pred_test_best))
print("Confusion Matrix (Best Naive Bayes):")
print(confusion_matrix(y_test, y_pred_test_best))
