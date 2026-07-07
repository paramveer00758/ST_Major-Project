import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. DATA LOADING (Directly using your uploaded file)
DATA_PATH = "Customer_Sentiment.csv"
df = pd.read_csv(DATA_PATH)

# Drop any blank rows in our target columns to keep it clean
df = df.dropna(subset=['review_text', 'sentiment'])


# 2. SIMPLE TEXT PREPROCESSING FUNCTION
def clean_text(text):
    text = str(text).lower()             # Convert text to lowercase
    text = re.sub(r'[^\w\s]', '', text) # Remove all punctuation symbols
    text = re.sub(r'\d+', '', text)     # Remove basic numbers
    return text

# Apply the cleaning function to create a new column
df['Cleaned_Review'] = df['review_text'].apply(clean_text)


# 3. CORE ML TRAINING PIPELINE
X = df['Cleaned_Review']
y = df['sentiment']

# Simple TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Basic Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.4, random_state=42)

# Simple Logistic Regression Classifier
model = LogisticRegression()
model.fit(X_train, y_train)

# Calculate Basic Accuracy Score
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# 4. STREAMLIT FRONT-END CLEAN LAYOUT
st.title("📊 Emotion Detection from Text using NLP")
st.write("A simple and basic Machine Learning project to analyze customer review sentiments.")

# Display basic project metrics
st.subheader("📈 Model Metrics")
st.write(f"**Dataset Name:** {DATA_PATH}")
st.write(f"**Total Dataset Rows:** {len(df)}")
st.write(f"**Model Test Accuracy:** {accuracy * 100:.2f}%")

st.markdown("---")

# INTERACTIVE TESTING FIELD
st.subheader("🔮 Live Sentiment Tester")
user_input = st.text_input("Enter a review sentence to test the model:")

if st.button("Check Sentiment"):
    if user_input.strip() != "":
        # Process input text using our pipeline steps
        cleaned_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        
        # Predict result
        result = model.predict(vectorized_input)[0]
        
        # Simple output formatting
        st.write(f"### Predicted Result: **{result.capitalize()}**")
    else:
        st.write("Please enter some text first.")

st.markdown("---")

# GRAPHS & VISUALIZATIONS SECTION
st.subheader("📊 Data Visualizations")

# Graph 1: Sentiment Distribution Chart
fig1, ax1 = plt.subplots(figsize=(6, 3))
sns.countplot(data=df, x='sentiment', palette="Set2", ax=ax1)
ax1.set_title("Dataset Sentiment Counts")
st.pyplot(fig1)

# Graph 2: Basic Confusion Matrix
st.write("**Confusion Matrix Chart:**")
cm = confusion_matrix(y_test, y_pred, labels=["positive", "neutral", "negative"])
fig2, ax2 = plt.subplots(figsize=(5, 3.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=["Pos", "Neu", "Neg"], yticklabels=["Pos", "Neu", "Neg"], ax=ax2)
ax2.set_ylabel('Actual Labels')
ax2.set_xlabel('Predicted Labels')
st.pyplot(fig2)

st.markdown("---")

# DATASET AUDIT VIEW
st.subheader("⚙️ Dataset Preview (First 10 Rows)")
st.dataframe(df[['review_text', 'Cleaned_Review', 'sentiment']].head(5))