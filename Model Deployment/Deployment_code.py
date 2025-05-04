import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import PowerTransformer, StandardScaler
from feature_engine.outliers import Winsorizer
import os

# Function to preprocess data for clustering
def preprocess_data(dataset):
    dataset['Production (MT)'] = dataset['Production (MT)'].fillna(dataset['Production (MT)'].mean())
    features = ['ENERGY (Energy Consumption)', 'TT_TIME (Total Cycle Time Including Breakdown)', 'Production (MT)']
    winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=features)
    data_winsorized = winsor.fit_transform(dataset[features])
    pt = PowerTransformer(method='yeo-johnson')
    data_winsorized_transformed = pt.fit_transform(data_winsorized)
    data_winsorized_transformed = pd.DataFrame(data_winsorized_transformed, columns=features)
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_winsorized_transformed)
    return data_scaled, scaler

def apply_clustering(data_scaled):
    kmeans = KMeans(n_clusters=2, random_state=42)
    kmeans_labels = kmeans.fit_predict(data_scaled)
    return kmeans_labels, kmeans

class DropUnwantedColumns:
    def __init__(self, drop_columns=['SRNO', 'HEATNO'], datetime_columns=True):
        self.drop_columns = drop_columns
        self.datetime_columns = datetime_columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.drop(columns=self.drop_columns, errors='ignore')
        if self.datetime_columns:
            datetime_columns = X.select_dtypes(include=[np.datetime64]).columns
            X = X.drop(columns=datetime_columns, errors='ignore')
        return X

def main():
    # Streamlit Styling with Steel Industry Background Image
    st.markdown(
        """
        <style>
        body {
            background-color: #f7f7f7;
            font-family: 'Arial', sans-serif;
            background-image: url('https://housing.com/news/wp-content/uploads/2022/11/is-code-for-steel-compressed.jpg'); /* Replace with your image URL */
            background-size: cover;
            background-position: center center;
            background-repeat: no-repeat;
        }
        .title {
            color: #003366;
            font-size: 32px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
        }
        .subtitle {
            color: #003366;
            font-size: 24px;
            text-align: center;
            margin-top: 20px;
        }
        .main-content {
            background-color: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
        .section-header {
            color: #003366;
            font-size: 20px;
            font-weight: bold;
            text-align: left;
            margin-top: 20px;
        }
        .btn {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 10px 20px;
            border-radius: 5px;
            border: none;
        }
        .btn:hover {
            background-color: #45a049;
        }
        .warning {
            color: #ff4d4d;
            font-size: 16px;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # HTML Template with Steel Industry Terminology and Design
    st.markdown(
        """
        <div style="text-align:center; padding: 20px;">
            <h2 style="font-size: 36px; color: #003366;">Welcome to the Steel Manufacturing Plant Data Analysis</h2>
            <p style="font-size: 18px; color: #333333;">We leverage data science and AI to improve steel production processes. Let's explore clustering and classification to optimize performance.</p>
            <img src="https://housing.com/news/wp-content/uploads/2022/11/is-code-for-steel-compressed.jpg" 
                 style="max-width: 60%; height: auto; margin-top: 20px;">
        </div>
        """, unsafe_allow_html=True
    )

    # Step 1: Clustering Phase
    st.subheader("Clustering Phase", anchor="subtitle")
    
    uploaded_file = st.file_uploader("Upload your dataset for Clustering", type=['xlsx', 'csv'])
    
    if uploaded_file is not None:
        dataset = pd.read_excel(uploaded_file, header=2)
        
        st.write("Preprocessing the data for clustering...")
        data_scaled, scaler = preprocess_data(dataset)
        
        st.write("Applying KMeans clustering...")
        kmeans_labels, kmeans = apply_clustering(data_scaled)
        
        dataset['Cluster_Label'] = kmeans_labels
        st.write("Cluster Labels added to dataset (0: Non-Optimum, 1: Optimum)")

        st.write(dataset.head())

        st.subheader("Cluster Visualization", anchor="section-header")
        fig, ax = plt.subplots()
        plt.scatter(dataset['ENERGY (Energy Consumption)'], dataset['Production (MT)'], c=dataset['Cluster_Label'], cmap='viridis')
        plt.xlabel('Energy Consumption (KWh)')
        plt.ylabel('Production (MT)')
        plt.title('Clustering: Optimum vs Non-Optimum')
        st.pyplot(fig)

        csv = dataset.to_csv(index=False)
        st.download_button(
            label="Download Clustered Data",
            data=csv,
            file_name="clustered_data.csv",
            mime="text/csv",
            key="download-clustered-data"
        )

    # Step 2: Classification Phase
    st.subheader("Classification Phase", anchor="subtitle")
    uploaded_labeled_file = st.file_uploader("Upload the clustered data (from previous step)", type=['xlsx', 'csv'])
    
    if uploaded_labeled_file is not None:
        try:
            if uploaded_labeled_file.name.endswith('xlsx'):
                dataset = pd.read_excel(uploaded_labeled_file, header=0)
            else:
                dataset = pd.read_csv(uploaded_labeled_file)
            
            st.write("Dataset loaded for classification", dataset.head())
            
            if 'Cluster_Label' not in dataset.columns:
                st.error("The dataset must contain a 'Cluster_Label' column (Optimum / Non-optimum).", unsafe_allow_html=True)
                return

            st.write("Preprocessing the data for classification...")

            preprocessing_pipeline = Pipeline(steps=[ 
                ('drop_columns', DropUnwantedColumns(drop_columns=['SRNO', 'HEATNO'])) 
            ])
            
            dataset_cleaned = preprocessing_pipeline.fit_transform(dataset)

            X = dataset.drop(columns=['Cluster_Label'])  
            y = dataset['Cluster_Label']  

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            categorical_features = X.select_dtypes(include=['object']).columns
            for col in categorical_features:
                X[col] = X[col].astype(str)

            numerical_features = X.select_dtypes(exclude=['object', 'datetime64']).columns

            numerical_pipeline = Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy='mean')), 
                ('scaler', RobustScaler())  
            ])

            categorical_pipeline = Pipeline(steps=[ 
                ('imputer', SimpleImputer(strategy='most_frequent')),  
                ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))  
            ])

            preprocessor = ColumnTransformer(
                transformers=[ 
                    ('num', numerical_pipeline, numerical_features),
                    ('cat', categorical_pipeline, categorical_features)
                ]
            )

            classification_pipeline = Pipeline(steps=[ 
                ('preprocessor', preprocessor), 
                ('classifier', xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'))  
            ])

            classification_pipeline.fit(X_train, y_train)
            y_pred = classification_pipeline.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            confusion = confusion_matrix(y_test, y_pred)

            st.write(f'Accuracy: {accuracy:.2f}')

            st.subheader("Confusion Matrix Heatmap", anchor="section-header")
            fig, ax = plt.subplots(figsize=(8, 6))  
            sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-optimum', 'Optimum'], yticklabels=['Non-optimum', 'Optimum'])
            ax.set_xlabel('Predicted Cluster')
            ax.set_ylabel('True Cluster')
            ax.set_title('Confusion Matrix: Steel Plant Classification')
            st.pyplot(fig)

            if not os.path.exists('classification_pipeline_xgboost.pkl'):
                joblib.dump(classification_pipeline, 'classification_pipeline_xgboost.pkl')

            dataset['Predicted_Cluster_Label'] = classification_pipeline.predict(X)
        
        except Exception as e:
            st.error(f"Error loading dataset: {e}")

if __name__ == "__main__":
    main()
