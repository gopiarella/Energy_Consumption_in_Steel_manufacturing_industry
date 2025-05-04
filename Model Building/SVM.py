# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 17:42:32 2025

@author: gopia
"""

import pandas as pd
from sklearn.svm import SVC
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

# Initialize the SVM classifier
svm = SVC(random_state=42)

# Step 3: Train the model
svm.fit(X_train, y_train)

# Make predictions
y_pred_train_svm = svm.predict(X_train)
y_pred_test_svm = svm.predict(X_test)

# Step 4: Evaluate the model
print("SVM - Train Accuracy:", accuracy_score(y_train, y_pred_train_svm))
print("SVM - Test Accuracy:", accuracy_score(y_test, y_pred_test_svm))

print("Classification Report (SVM):")
print(classification_report(y_test, y_pred_test_svm))

print("Confusion Matrix (SVM):")
print(confusion_matrix(y_test, y_pred_test_svm))



# Define parameter grid
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
    'gamma': ['scale', 'auto', 0.1, 1],  # Kernel coefficient for 'rbf', 'poly', 'sigmoid'
    'degree': [3, 4, 5],  # Degree of the polynomial kernel function (only if 'poly' is used)
}

# Initialize the SVM classifier
svm = SVC(random_state=42)

# Grid search
grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters
print(f"Best Parameters: {grid_search.best_params_}")

# Evaluate the best model
best_model = grid_search.best_estimator_
y_pred_test_best = best_model.predict(X_test)

print("Best SVM - Test Accuracy:", accuracy_score(y_test, y_pred_test_best))
print("Classification Report (Best SVM):")
print(classification_report(y_test, y_pred_test_best))
print("Confusion Matrix (Best SVM):")
print(confusion_matrix(y_test, y_pred_test_best))
