# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 16:55:08 2025

@author: gopia
"""

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from feature_engine.outliers import Winsorizer
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering

# Load dataset
dataset = pd.read_excel('GT_DataCollection_Client Info (Student).xlsx', header=2)
preprocessed = pd.read_csv('preprocessed_data.csv')
# Select relevant columns for clustering
data = dataset[['ENERGY (Energy Consumption)', 'TT_TIME (Total Cycle Time Including Breakdown)', 'Production (MT)']]

# Handle missing values and check for skewness
data['Production (MT)'] = data['Production (MT)'].fillna(data['Production (MT)'].mean()) 
for i in data[data.columns]: 
    print(data[i].skew())  # Checking skewness of the data

# Winsorization (capping extreme outliers using IQR method)
features = ['ENERGY (Energy Consumption)', 'TT_TIME (Total Cycle Time Including Breakdown)', 'Production (MT)']
winsor = Winsorizer(capping_method='iqr',  # Method to cap using IQR
                    tail='both',           # Apply to both lower and upper tails
                    fold=1.5,              # Defines how far to extend the cap
                    variables=features)    # List of features to apply Winsorization to
data_winsorized = winsor.fit_transform(data)
print(data_winsorized.head())

# Power Transformation (Yeo-Johnson to normalize skewed data)
pt = PowerTransformer(method='yeo-johnson')
data_winsorized_transformed = pt.fit_transform(data_winsorized)
data_winsorized_transformed = pd.DataFrame(data_winsorized_transformed, columns=data_winsorized.columns)

# Standard Scaling 
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_winsorized_transformed)
data_scaled = pd.DataFrame(data_scaled, columns=data_winsorized_transformed.columns)
print(data_scaled.head())

# Apply KMeans Clustering (2 Clusters)
kmeans = KMeans(n_clusters=2, random_state=42) 
kmeans_labels = kmeans.fit_predict(data_scaled)
print("Centroids of the clusters:", kmeans.cluster_centers_)

# Calculate Silhouette Score for KMeans Clustering
kmeans_score = silhouette_score(data_scaled, kmeans.labels_)
print("Silhouette Score (KMeans):", kmeans_score)

# Apply DBSCAN Clustering (Make sure DBSCAN is defined here)
dbscan = DBSCAN(eps=0.8, min_samples=5)
dbscan_labels = dbscan.fit_predict(data_scaled)

# Calculate Silhouette Score for DBSCAN Clustering
dbscan_score = silhouette_score(data_scaled, dbscan_labels) 
print("Silhouette Score (DBSCAN):", dbscan_score)

# Apply Gaussian Mixture Model (GMM) Clustering
gmm = GaussianMixture(n_components=2)
gmm_labels = gmm.fit_predict(data_scaled)
gmm_score = silhouette_score(data_scaled, gmm_labels) 
print("Silhouette Score (GMM):", gmm_score)

# Apply Agglomerative Clustering
agg_clust = AgglomerativeClustering(n_clusters=2)
agg_clust_labels = agg_clust.fit_predict(data_scaled)

# Calculate Silhouette Score for Agglomerative Clustering
agg_score = silhouette_score(data_scaled, agg_clust_labels) 
print("Silhouette Score (Agglomerative Clustering):", agg_score)

# Store results in a table format
results = []

# KMeans Results
results.append({
    'Model': 'KMeans',
    'Silhouette Score': kmeans_score,
    'Cluster 0 Size': sum(kmeans.labels_ == 0),
    'Cluster 1 Size': sum(kmeans.labels_ == 1)
})

# DBSCAN Results
results.append({
    'Model': 'DBSCAN',
    'Silhouette Score': dbscan_score,
    'Cluster 0 Size': sum(dbscan_labels == 0),
    'Cluster 1 Size': sum(dbscan_labels == 1)
})

# GMM Results
results.append({
    'Model': 'GMM',
    'Silhouette Score': gmm_score,
    'Cluster 0 Size': sum(gmm_labels == 0),
    'Cluster 1 Size': sum(gmm_labels == 1)
})

# Agglomerative Clustering Results
results.append({
    'Model': 'Agglomerative',
    'Silhouette Score': agg_score,
    'Cluster 0 Size': sum(agg_clust_labels == 0),
    'Cluster 1 Size': sum(agg_clust_labels == 1)
})

# Convert results into a DataFrame and print
df_results = pd.DataFrame(results)
print(df_results) 


data['clusters']=kmeans_labels
grouped_data = data.groupby('clusters').mean()  # This will calculate the mean for each feature in each cluster
print(grouped_data)

import joblib
joblib.dump(kmeans,'kmeans_model.pkl')

# Check rows with negative total cycle time
negative_cycle_time = data[data['TT_TIME (Total Cycle Time Including Breakdown)'] < 0]
print(negative_cycle_time)


data['Cluster_Label'] = data['clusters'].map({0: 'Non-Optimum', 1: 'Optimum'})

plt.figure(figsize=(10, 6))
plt.scatter(data['ENERGY (Energy Consumption)'], data['Production (MT)'], c=data['clusters'], cmap='viridis')
plt.title('Clustering with Optimum (1) and Non-Optimum (0) Labels')
plt.xlabel('Energy Consumption')
plt.ylabel('Production (MT)')
plt.colorbar(label='Cluster Label')
plt.show()

preprocessed['clusters']=kmeans_labels 
preprocessed.to_csv('labeled_data.csv')















