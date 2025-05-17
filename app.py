# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("🚢 Titanic Dataset - Exploratory Data Analysis")

@st.cache_data
def load_data():
    return pd.read_csv("data/train.csv")  # Use relative path

df = load_data()

st.subheader("Raw Data")
if st.checkbox("Show raw data"):
    st.write(df.head())

st.subheader("Missing Values")
st.write(df.isnull().sum())

st.subheader("Survival Count by Gender")
fig, ax = plt.subplots()
sns.countplot(data=df, x='Sex', hue='Survived', ax=ax)
st.pyplot(fig)

st.subheader("Correlation Heatmap")
fig2, ax2 = plt.subplots()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax2)
st.pyplot(fig2)
