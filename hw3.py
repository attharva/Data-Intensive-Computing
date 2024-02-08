from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import StandardScaler
import streamlit as st
import plotly.express as px
import altair as alt
import seaborn as sns
sns.set_style("whitegrid")
import base64
import datetime
from matplotlib import rcParams
from  matplotlib.ticker import PercentFormatter
import os
import shutil
import tempfile
from typing import Optional
import joblib
import streamlit as st

from codeinterpreterapi import CodeInterpreterSession
st.set_option('deprecation.showPyplotGlobalUse', False)



st.title("NYC Airbnb Tool")
st.markdown("")

# "with" notation
with st.sidebar:   
    file_upload = st.file_uploader("Upload a Dataset", type=["csv", "txt"])
    button = st.button("Submit", key="submit")

if not file_upload:
    st.markdown("Please upload a dataset to continue.")
else:
    df = pd.read_csv(file_upload)
        # User Inputs
    min_price = st.sidebar.slider('Minimum Price', min_value=0, max_value=df['price'].max(), value=0)
    max_price = st.sidebar.slider('Maximum Price', min_value=min_price, max_value=df['price'].max(), value=df['price'].max())
    min_number_reviews = st.sidebar.slider('Minimum Number of Reviews', min_value=0, max_value=df['number_of_reviews'].max(), value=0)
    min_nights = st.sidebar.slider('Minimum Nights', min_value=0, max_value=df['minimum_nights'].max(), value=0)
    # global_boroughs = st.sidebar.multiselect('', df['neighbourhood'].unique(), default='Manhattan')
    selected_neighbourhood_groups = st.sidebar.multiselect('Select Neighbourhood Groups', df['neighbourhood_group'].unique(), default=df['neighbourhood_group'].unique())

    show_map = st.sidebar.checkbox('Show Map')

    defaultcols = ["price", "minimum_nights", "room_type", "neighbourhood", "name", "number_of_reviews"]
    cols = st.multiselect('', df.columns.tolist(), default=defaultcols)
    st.dataframe(df[cols].head(10))
    # Some data cleaning 
    df = df.sort_values(by = 'last_review')
    df['last_review'].fillna(method='ffill', inplace=True)
    # Replacing the missing values in the 'reviews_per_month' with its median value
    df['reviews_per_month'].fillna(df['reviews_per_month'].mean(),inplace = True)
    # Looking for duplicate values in the data
    df.duplicated().sum()
    df = df.drop_duplicates()
    #Data type conversion
    # Data Type Conversion
    # Converting 'last_review' to datetime format
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    # Feature Engineering
    # Extracting month from 'last_review'
    df['month'] = df['last_review'].dt.month
    # Converting 'room_type' to categorical data type
    df['room_type'] = df['room_type'].astype('category')
    # Dropping the data which does not make sense.
    # Dropping the data where the price is listed as 0
    price_0_idx = df[df['price'] == 0].index.values
    df.drop(price_0_idx,inplace = True)
    st.markdown("All data cleaning has been done")
    # Removing the data where the 'minimum_nights' has a value greater than 365 days.
    min_night_idx = df[df['minimum_nights'] > 365].index.values
    df.drop(min_night_idx,inplace = True)
    plt.figure(figsize=(12,8))
    plt.title(' We have run a few data cleaning operations. Here is a Heatmap of Missing Values in the dataset')
    sns.heatmap(df.isnull(),yticklabels=False,cbar = False)
    st.pyplot()

    # Filter Data
    df = df[(df['price'] >= min_price) & (df['price'] <= max_price) & (df['number_of_reviews'] >= min_number_reviews) & (df['neighbourhood_group'].isin(selected_neighbourhood_groups))
            & (df['minimum_nights'] >= min_nights)]

    ################################## DISTRICT ###############################
    if st.button("Explore Data"):
        st.subheader('Summary Statistics')
        st.write(df.describe())
        # Room Type Distribution
        
        st.subheader('Categorical Distributions')
        selected_categorical_variable = st.selectbox('Select Categorical Variable', ["neighbourhood_group", "neighbourhood", "room_type"], index=2)
        st.subheader(f'{selected_categorical_variable} Distribution Count')
        cat_variable_counts = df[selected_categorical_variable].value_counts()
        st.bar_chart(cat_variable_counts)

        st.subheader(f'{selected_categorical_variable} Distribution of price')

        average_price_by_neighbourhood = df.groupby(selected_categorical_variable)['price'].mean().sort_values(ascending=False)
        st.bar_chart(average_price_by_neighbourhood)

        # room_type_counts = df['room_type'].value_counts()
        # st.bar_chart(room_type_counts)
        # Price Distribution by Neighbourhood
        st.subheader('Price Distribution by Neighbourhood')
        # Average Price by Neighbourhood

        selected_neighbourhood = st.selectbox('Select Neighbourhood', df['neighbourhood'].unique())
        neighbourhood_data = df[df['neighbourhood'] == selected_neighbourhood]
        plt.hist(neighbourhood_data['price'], bins=30, edgecolor='black', alpha=0.7)
        st.pyplot()

        # Map of Listings
        if show_map:
            st.subheader('Map of Listings')
            st.map(df[['latitude', 'longitude']].dropna())

        # Price vs. Number of Reviews
        st.subheader('Price vs. Number of Reviews')
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='number_of_reviews', y='price', data=df, alpha=0.5)
        plt.title('Price vs. Number of Reviews')
        plt.xlabel('Number of Reviews')
        plt.ylabel('Price')
        st.pyplot()
        
        sns.distplot(df['minimum_nights'])
        plt.title('Minimum Nights')
        plt.xlabel('Minimum Nights')
        plt.grid()
        st.pyplot()
        
        top_neighbourhoods = df['neighbourhood'].value_counts().head(10)
        explode = (0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0)  # Exploding 1st and 2nd slice for emphasis
        colors = sns.color_palette("Set3")
        plt.figure(figsize=(14, 7))
        plt.pie(top_neighbourhoods, labels=top_neighbourhoods.index, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops=dict(width=0.3), explode=explode)
        plt.title('Prominent Neighbourhoods by Number of Properties')
        st.pyplot()

        # Average Rate Across Neighbourhood Clusters (Point Plot)
        avg_price_neighbourhood_group = df.groupby('neighbourhood_group')['price'].mean()
        plt.figure(figsize=(14, 7))
        sns.pointplot(x=avg_price_neighbourhood_group.index, y=avg_price_neighbourhood_group.values, linestyles="--", markers='D')
        plt.title('Mean Nightly Rate Across Boroughs')
        plt.xlabel('Borough')
        plt.ylabel('Mean Rate ($)')
        plt.grid(True, axis='y')
        st.pyplot()

    st.subheader('K Means')
    import numpy as np

# Define a function for initializing centroids
    def initialize_centroids(X, k):
        """Randomly initialize centroids from the dataset X.
        Args:
        X (ndarray): Data points for clustering.
        k (int): Number of clusters.
        Returns:
        centroids (ndarray): Initialized centroids.
        """
        m, n = X.shape
        centroids = np.zeros((k, n))
        idx = np.random.permutation(m)
        centroids = X[idx[:k], :]
        return centroids
    if st.button('Run Clustering and Classification'):
# Selecting only 'price' and 'minimum_nights' features for k-Means
        X_kmeans = df[['price', 'minimum_nights']].values

        # Scale the features
        scaler_km = StandardScaler()
        X_kmeans_scaled = scaler_km.fit_transform(X_kmeans)

        # Initialize centroids for k=3 for demonstration
        initial_centroids = initialize_centroids(X_kmeans_scaled, k=3)

        

        def find_closest_centroids(X, centroids):
            """Assigns each data point to the closest centroid.
            Args:
            X (ndarray): Data points for clustering.
            centroids (ndarray): Current centroids.
            Returns:
            idx (ndarray): Index of the closest centroid for each data point.
            """
            m = X.shape[0]
            k = centroids.shape[0]
            idx = np.zeros(m, dtype=int)

            for i in range(m):
                distances = np.linalg.norm(X[i] - centroids, axis=1)
                closest_centroid = np.argmin(distances)
                idx[i] = closest_centroid

            return idx

        def compute_centroids(X, idx, k):
            """Recalculates centroids as the mean of all points assigned to each cluster.
            Args:
            X (ndarray): Data points for clustering.
            idx (ndarray): Index of the closest centroid for each data point.
            k (int): Number of clusters.
            Returns:
            centroids (ndarray): New centroids calculated as the mean of assigned points.
            """
            _, n = X.shape
            centroids = np.zeros((k, n))

            for i in range(k):
                points = X[idx == i]
                centroids[i] = np.mean(points, axis=0) if points.shape[0] > 0 else np.zeros((n,))

            return centroids

        def run_kmeans(X, initial_centroids, max_iters=10):
            """Runs the k-Means algorithm.
            Args:
            X (ndarray): Data points for clustering.
            initial_centroids (ndarray): Initial centroids.
            max_iters (int): Maximum number of iterations.
            Returns:
            centroids (ndarray): Final centroids.
            idx (ndarray): Index of the closest centroid for each data point.
            """
            centroids = initial_centroids
            for _ in range(max_iters):
                idx = find_closest_centroids(X, centroids)
                centroids = compute_centroids(X, idx, k=centroids.shape[0])

            return centroids, idx

        # Run k-Means
        kmeans_centroids, kmeans_idx = run_kmeans(X_kmeans_scaled, initial_centroids, max_iters=10)

        
        # Plotting the data points and centroids to visualize the clusters
        plt.figure(figsize=(12, 8))
        plt.scatter(X_kmeans_scaled[:, 0], X_kmeans_scaled[:, 1], c=kmeans_idx, cmap='viridis', marker='o', alpha=0.5, label='Data points')
        plt.scatter(kmeans_centroids[:, 0], kmeans_centroids[:, 1], s=300, c='red', marker='x', label='Centroids')
        plt.title('2D visualization of k-Means Clustering')
        plt.xlabel('Scaled Price')
        plt.ylabel('Scaled Minimum Nights')
        plt.legend()
        st.pyplot()
        
        st.markdown("""
    The k-Means algorithm has been run for 10 iterations, resulting in the final centroids shown above. These centroids represent the center of the clusters in the scaled feature space of price and minimum_nights.

    Next, let's visualize the clusters along with the centroids on a scatter plot to see how our implementation of k-Means has clustered the data. We'll map each cluster to a different color and plot the centroids as well.​
    """)
        st.subheader('Decision Tree')
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler, LabelEncoder

        # Selecting relevant features and the target for the k-NN model
        # Selecting relevant features for the model
        features = df[['price', 'minimum_nights', 'number_of_reviews', 'reviews_per_month',
                        'calculated_host_listings_count', 'availability_365']]

        # Define the binary target variable based on the price threshold
        price_threshold = df['price'].median()
        binary_target = (df['price'] > price_threshold).astype(int)

        # Split the dataset into training and testing sets
        X_train, X_test, binary_y_train, binary_y_test = train_test_split(features, binary_target, test_size=0.2, random_state=42)

    
    st.subheader('Prediction using Random Forest Regressor')
    loaded_model = joblib.load('airbnb_price_prediction_model.joblib')
    st.markdown("We have loaded the model using joblib")
    loaded_model = joblib.load('airbnb_price_prediction_model.joblib')


    # User input form
    
    # User input form on the main page
    st.header('Enter Property Details')
    latitude = 0.0
    longitude = 0.0
    minimum_nights = st.number_input('Minimum Nights', value=1)
    number_of_reviews = st.number_input('Number of Reviews', value=0)
    reviews_per_month = st.number_input('Reviews per Month', value=0.0)
    calculated_host_listings_count = st.number_input('Host Listings Count', value=1)
    availability_365 = st.number_input('Availability (in days)', value=0)
    neighbourhood_group = st.selectbox('Neighbourhood Group', ['Bronx', 'Brooklyn', 'Manhattan', 'Queens', 'Staten Island'])
    neighbourhood = st.selectbox('Neighbourhood', df['neighbourhood'].unique())
    room_type = st.selectbox('Room Type', ['Entire home/apt', 'Private room', 'Shared room'])

    # Create a DataFrame with the user input
    user_input = pd.DataFrame({
        'latitude': [latitude],
        'longitude': [longitude],
        'minimum_nights': [minimum_nights],
        'number_of_reviews': [number_of_reviews],
        'reviews_per_month': [reviews_per_month],
        'calculated_host_listings_count': [calculated_host_listings_count],
        'availability_365': [availability_365],
        'neighbourhood_group': [neighbourhood_group],
        'neighbourhood': [neighbourhood],
        'room_type': [room_type]
    })

    if st.button('Predict Price'):
        # Create a DataFrame with the user input
        user_input = pd.DataFrame({
            'latitude': [latitude],
            'longitude': [longitude],
            'minimum_nights': [minimum_nights],
            'number_of_reviews': [number_of_reviews],
            'reviews_per_month': [reviews_per_month],
            'calculated_host_listings_count': [calculated_host_listings_count],
            'availability_365': [availability_365],
            'neighbourhood_group': [neighbourhood_group],
            'neighbourhood': [neighbourhood],
            'room_type': [room_type]
        })

        # Make predictions
        prediction = loaded_model.predict(user_input)

        # Display the prediction
        st.subheader('Predicted Price:')
        st.write(f"${prediction[0]:.2f}")