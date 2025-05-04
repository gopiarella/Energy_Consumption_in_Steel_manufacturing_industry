# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 17:46:55 2025

@author: gopia
"""

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCVS

# Load the preprocessed dataset
dataset = pd.read_csv('labeled_data.csv')  # Replace with your dataset file path

# Step 1: Extract Features and Target Variable
X = dataset.drop('clusters', axis=1)  # Features
y = dataset['clusters']  # Target

# Step 2: Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN classifier
knn = KNeighborsClassifier()

# Step 3: Train the model
knn.fit(X_train, y_train)

# Make predictions
y_pred_train_knn = knn.predict(X_train)
y_pred_test_knn = knn.predict(X_test)

# Step 4: Evaluate the model
print("KNN - Train Accuracy:", accuracy_score(y_train, y_pred_train_knn))
print("KNN - Test Accuracy:", accuracy_score(y_test, y_pred_test_knn))

print("Classification Report (KNN):")
print(classification_report(y_test, y_pred_test_knn))

print("Confusion Matrix (KNN):")
print(confusion_matrix(y_test, y_pred_test_knn))



# Define the parameter grid
param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11],  # Number of neighbors to consider
    'metric': ['euclidean', 'manhattan', 'minkowski'],  # Distance metric
}

# Initialize the KNN classifier
knn = KNeighborsClassifier()

# Grid search
grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best Parameters: {grid_search.best_params_}")

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred_test_best = best_model.predict(X_test)

print("Best KNN - Test Accuracy:", accuracy_score(y_test, y_pred_test_best))
print("Classification Report (Best KNN):")
print(classification_report(y_test, y_pred_test_best))
print("Confusion Matrix (Best KNN):")
print(confusion_matrix(y_test, y_pred_test_best))
