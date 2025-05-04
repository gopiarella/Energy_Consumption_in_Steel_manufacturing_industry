# -*- coding: utf-8 -*- 
"""
Created on Tue Mar 11 12:40:03 2025

@author: gopia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Reading the dataset
dataset = pd.read_excel('GT_DataCollection_Client Info (Student).xlsx', header=2)

# Check the shape of the dataset (rows and columns)
print(dataset.shape)

# Display first few rows of the dataset
dataset.head()

# Display basic info about the dataset
dataset.info()

# Display summary statistics of numerical columns
dataset.describe().T

# Display data types of each column
dataset.dtypes

# Dropping columns that are not needed for analysis
dataset.drop(columns=['SRNO','HEATNO'], axis=1, inplace=True)

# Identify numerical and categorical features
numerical_features = dataset.select_dtypes(exclude=['object', 'datetime64']).columns
categorical_features = dataset.select_dtypes(include='object').columns

# --- Exploratory Data Analysis (EDA) --- #

# Compute mean, median, mode, variance, standard deviation, range, skewness, and kurtosis for numerical columns
for i in dataset[numerical_features]:
    mean = dataset[i].mean()
    print(f'mean of {i} is: {mean}')
    
for i in dataset[numerical_features]:
    median = dataset[i].median()
    print(f'Median of {i} is {median}')
    
for i in dataset[categorical_features]:
    mode = dataset[i].mode()[0]
    print(f'Mode of {i} is: {mode}')

for i in dataset[numerical_features]:
    variance = dataset[i].var()
    print(f'Variance of {i} is: {variance}')

for i in dataset[numerical_features]:
    stdv = dataset[i].std()
    print(f'Standard Deviation of {i} is: {stdv}')

for i in dataset[numerical_features]:
    range_val = dataset[i].max() - dataset[i].min()
    print(f'Range of {i} is: {range_val}')

for i in dataset[numerical_features]:
    skew = dataset[i].skew()
    print(f'Skewness of {i} is: {skew}')

for i in dataset[numerical_features]:
    kurt = dataset[i].kurt()
    print(f'Kurtosis of {i} is: {kurt}')

# --- Visualizations --- #

# Histograms for numerical features
for i in dataset[numerical_features]:
    sns.histplot(data=dataset, x=i)
    plt.title(f'Histogram for {i}')
    plt.show()

# Countplots for categorical features
for i in dataset[categorical_features]:
    sns.countplot(data=dataset, x=i)
    plt.title(f'Count plot of {i}')
    plt.show()

# Boxplot for numerical features (grouping them into plots for readability)
import numpy as np
num_columns = len(numerical_features)
columns_per_plot = 6
num_plots = np.ceil(num_columns / columns_per_plot).astype(int)

for i in range(num_plots):
    start_col = i * columns_per_plot
    end_col = min((i + 1) * columns_per_plot, num_columns)
    subset = dataset[numerical_features[start_col:end_col]]
    subset.plot(kind='box', subplots=True, sharey=False, figsize=(15, 10))
    plt.subplots_adjust(wspace=0.8)
    plt.suptitle(f'Boxplot')
    plt.show()

# Q-Q plots for normality check
from scipy.stats import probplot
for i in dataset[numerical_features]:
    probplot(dataset[i], dist='norm', plot=plt)
    plt.title(f'Q-Q plot of {i}')
    plt.show()

# Correlation heatmap to understand relationships between numerical features
corr_matrix = dataset[numerical_features].corr()
plt.figure(figsize=(50, 30))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

# Scatter plots between pairs of features (visualize relationships)
features_to_plot = [
    ('KWH_PER_MIN', 'PIGIRON'),
    ('TAP_DURATION', 'TAPPING_TIME'),
    ('Pour_Back_Metal', 'TAP_DURATION'),
    ('LM_WT', 'Production (MT)'),
    ('INJ1_QTY\n(Coke Injection Qty)', 'INJ2_QTY\n(Coke Injection Qty)'),
    ('TAP_DURATION', 'Pour_Back_Metal'),
    ('TAP_DURATION', 'LM_WT')
]

for feature_x, feature_y in features_to_plot:
    sns.scatterplot(data=dataset, x=feature_x, y=feature_y)
    plt.title(f'Scatter plot between {feature_x} and {feature_y}')
    plt.xlabel(feature_x)
    plt.ylabel(feature_y)
    plt.show()

# --- Time Series Analysis --- #

# Setting the datetime index (assuming the 'DATETIME' column exists)
dataset.set_index('DATETIME', inplace=True)

# Decomposing the 'KWH_PER_TON (Energy Consumption Per Ton)' into trend, seasonal, and residual components
from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(dataset['KWH_PER_TON (Energy Consumption Per Ton)'], model='additive', period=12)

# Plotting the decomposed components
plt.figure(figsize=(12, 8))
plt.subplot(411)
plt.plot(decomposition.observed)
plt.title('Observed')
plt.subplot(412)
plt.plot(decomposition.trend)
plt.title('Trend')
plt.subplot(413)
plt.plot(decomposition.seasonal)
plt.title('Seasonal')
plt.subplot(414)
plt.plot(decomposition.resid)
plt.title('Residual')
plt.tight_layout()
plt.show()

# Plotting 'KWH_PER_TON' over time
plt.figure(figsize=(12, 6))
plt.plot(dataset.index, dataset['KWH_PER_TON (Energy Consumption Per Ton)'])
plt.title("Energy Consumption Over Time")
plt.xlabel('Date')
plt.ylabel('Energy Consumption')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting 'Production (MT)' over time
plt.figure(figsize=(10, 5))
plt.plot(dataset['Production (MT)'])
plt.xlabel('Date')
plt.ylabel('Production (MT)')
plt.title('Production Over Time')
plt.show()

# Plotting both 'KWH_PER_TON' and 'Production (MT)' over time
plt.figure(figsize=(12, 6))
plt.plot(dataset.index, dataset['KWH_PER_TON (Energy Consumption Per Ton)'], label='Energy Consumption')
plt.plot(dataset.index, dataset['Production (MT)'], label='Production (MT)', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Energy Consumption and Production Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Plotting 'Melting Time' vs 'Energy Consumption' over time
plt.figure(figsize=(12, 6))
plt.plot(dataset.index, dataset['MELT_TIME (Melting Time)'], label='Melting Time', color='orange')
plt.plot(dataset.index, dataset['KWH_PER_TON (Energy Consumption Per Ton)'], label='Energy Consumption', color='blue')
plt.xlabel('Date')
plt.ylabel('Time (or Energy)')
plt.title('Melting Time vs. Energy Consumption Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- Handling Missing Values and Duplicates --- #

# Checking for duplicates
print(dataset.duplicated().sum())

# Checking for missing values
missing_values = dataset.isnull().sum()
missing_values = missing_values[missing_values > 0]
print(missing_values)

# Forward fill and backward fill for specific columns
dataset['LAB_REP_TIME'] = dataset['LAB_REP_TIME'].ffill()
dataset['PREV_TAP_TIME'] = dataset['PREV_TAP_TIME'].bfill()

# Fill missing values in 'Production (MT)' column with median
dataset['Production (MT)'] = dataset['Production (MT)'].fillna(dataset['Production (MT)'].median())

# Rechecking for missing values after filling
missing_values = dataset.isnull().sum()
missing_values = missing_values[missing_values > 0]
print(missing_values)

# Extract year, month, and day from the datetime index
dataset['year'] = dataset.index.year
dataset['month'] = dataset.index.month
dataset['day'] = dataset.index.day

# Resetting index and removing datetime columns
dataset = dataset.reset_index(drop=True)
dataset['pow_on_to_tap_diff'] = (dataset['TAPPING_TIME'] - dataset['POW_ON_TIME']).dt.total_seconds()
dataset['lab_report_and_tap_diff_seconds'] = (dataset['LAB_REP_TIME'] - dataset['PREV_TAP_TIME']).dt.total_seconds()

# Dropping original datetime columns
dataset = dataset.drop(columns=['POW_ON_TIME', 'TAPPING_TIME', 'LAB_REP_TIME', 'PREV_TAP_TIME'])

# Dropping columns with zero variance
variances = dataset[numerical_features].var()
print(variances[variances == 0].index)

# Drop columns with zero variance
dataset = dataset.drop(columns=variances[variances == 0].index)

# Drop columns with near-zero variance (if required)
threshold = 0.01
near_zero_variance_columns = variances[variances < threshold].index.tolist()
print("Near-zero variance columns:", near_zero_variance_columns)

# One-hot encoding categorical variables
dataset = pd.get_dummies(dataset, columns=['GRADE', 'SECTION_IC'], drop_first=True)

# Convert all columns to integers
dataset = dataset.astype(int)

# Check the first few rows of the processed data
dataset.head()

# --- Scaling the Data --- #

# Apply RobustScaler for scaling features (robust to outliers)
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
df_scaled = scaler.fit_transform(dataset) 

import joblib
# Save the scaler to a file
joblib.dump(scaler, 'scaler.pkl')


# Convert the scaled data into a DataFrame with original column names
df_scaled = pd.DataFrame(df_scaled, columns=dataset.columns)

# Display the first few rows of the scaled dataset
df_scaled.head()

# Save the preprocessed data to a CSV file
df_scaled.to_csv('preprocessed_data.csv', index=False)




