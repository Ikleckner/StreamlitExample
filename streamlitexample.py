import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

@st.cache_data
def load_data():
    df = pd.read_csv("/Users/IsabellaKleckner/Downloads/axe hackathon/movie_dataset.csv")  # Change to your dataset path
    df = df[['budget', 'popularity', 'genres', 'release_date', 'runtime', 'vote_average', 'vote_count', 'revenue']]
    df.dropna(inplace=True)
    return df

df = load_data()

# Sidebar - User Inputs
st.sidebar.header("Explore Movie Revenue")
feature = st.sidebar.selectbox("Select a Feature to Compare with Revenue", 
                               ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count', 'genres', 'release_date'])

# Show selected feature vs revenue
st.title(f"Movie Revenue Analysis: {feature} vs Revenue")

# Scatter plot for numerical features
if feature in ['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']:
    fig, ax = plt.subplots()
    sns.scatterplot(x=df[feature], y=df['revenue'], alpha=0.6)
    ax.set_xlabel(feature.capitalize())
    ax.set_ylabel("Revenue")
    ax.set_title(f"{feature.capitalize()} vs Revenue")
    st.pyplot(fig)

# Bar plot for genres
elif feature == "genres":
    genre_revenue = df.groupby('genres')['revenue'].mean().sort_values(ascending=False).head(10)
    fig, ax = plt.subplots()
    genre_revenue.plot(kind='bar', ax=ax, color="royalblue")
    ax.set_ylabel("Average Revenue")
    ax.set_title("Top 10 Genres by Revenue")
    st.pyplot(fig)

# Revenue by Release Month
elif feature == "release_date":
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['release_month'] = df['release_date'].dt.month
    month_revenue = df.groupby('release_month')['revenue'].mean()
    
    fig, ax = plt.subplots()
    sns.barplot(x=month_revenue.index, y=month_revenue.values, ax=ax, palette="viridis")
    ax.set_xlabel("Release Month")
    ax.set_ylabel("Average Revenue")
    ax.set_title("Average Revenue by Release Month")
    st.pyplot(fig)

# Predictive Model - Linear Regression
st.sidebar.subheader("Predict Revenue")
if st.sidebar.button("Run Prediction Model"):
    X = df[['budget', 'popularity', 'runtime', 'vote_average', 'vote_count']]
    y = df['revenue']
    
    model = LinearRegression()
    model.fit(X, y)
    predictions = model.predict(X)

    st.write("### Regression Model Results")
    st.write(f"R² Score: {model.score(X, y):.2f}")

    # Show actual vs predicted revenue
    fig, ax = plt.subplots()
    sns.scatterplot(x=y, y=predictions, alpha=0.6)
    ax.set_xlabel("Actual Revenue")
    ax.set_ylabel("Predicted Revenue")
    ax.set_title("Actual vs Predicted Revenue")
    st.pyplot(fig)