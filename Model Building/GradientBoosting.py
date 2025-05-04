# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 15:53:42 2025

@author: gopia
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier

# Load the preprocessed dataset
dataset = pd.read_csv('labeled_data.csv')

# Step 1: Extract Features and Target Variable
X = dataset.drop('clusters', axis=1)  # Features
y = dataset['clusters']  # Target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting model
gb_model = GradientBoostingClassifier(random_state=42)

# Fit the model to the training data
gb_model.fit(X_train, y_train)

# Predict the test data
y_pred = gb_model.predict(X_test)

# Calculate and display metrics
train_accuracy = gb_model.score(X_train, y_train)  # Training accuracy
test_accuracy = gb_model.score(X_test, y_test)  # Testing accuracy

print("Training Accuracy: ", train_accuracy)
print("Test Accuracy: ", test_accuracy)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))



gb_model = GradientBoostingClassifier(random_state=42)

# Define the hyperparameters for tuning
param_grid = {
    'n_estimators': [50, 100, 150],  # Number of trees
    'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Step size
    'max_depth': [3, 5, 7],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required at each leaf node
    'subsample': [0.8, 0.9, 1.0],  # Fraction of samples used for fitting each tree
    'max_features': ['auto', 'sqrt', 'log2']  # Features to consider for the best split
}

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=gb_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

# Fit the GridSearchCV
grid_search.fit(X_train, y_train)

# Best Hyperparameters found by GridSearchCV
print(f"Best Hyperparameters: {grid_search.best_params_}")

# Get the best model after tuning
best_model = grid_search.best_estimator_

# Evaluate the best model on the training and testing data
train_accuracy = best_model.score(X_train, y_train)  # Training accuracy
test_accuracy = best_model.score(X_test, y_test)  # Testing accuracy

# Make predictions on the test set
y_pred = best_model.predict(X_test)

# Print evaluation metrics
print("Training Accuracy: ", train_accuracy)
print("Test Accuracy: ", test_accuracy)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
