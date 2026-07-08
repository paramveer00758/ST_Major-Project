import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="NLP Sentiment Analyzer", page_icon="🧠", layout="wide")

# ==========================================
# SESSION STATE INITIALIZATION (Authentication)
# ==========================================
if 'users_db' not in st.session_state:
    # A simple mock database for users. In production, use a real DB with hashed passwords.
    st.session_state.users_db = {'admin': 'pass123'} 
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'current_user' not in st.session_state:
    st.session_state.current_user = ""

# ==========================================
# HELPER FUNCTIONS (Cached for performance)
# ==========================================

@st.cache_data
def load_data():
    """Loads the provided dataset."""
    file_path = 'Customer_Sentiment.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return df
    else:
        return None

def preprocess_text(text):
    """Cleans text for NLP processing."""
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove punctuations and numbers (keep only alphabets)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def train_nlp_model(df):
    """Trains the NLP model pipeline and returns necessary components."""
    # 1. Data Cleaning
    df_clean = df.dropna(subset=['review_text', 'sentiment']).copy()
    
    # 2. Text Preprocessing
    df_clean['cleaned_text'] = df_clean['review_text'].apply(preprocess_text)
    
    # Remove empty strings after cleaning
    df_clean = df_clean[df_clean['cleaned_text'] != ""]
    
    X = df_clean['cleaned_text']
    y = df_clean['sentiment']
    
    # Train-test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Feature Engineering / Vectorization (TF-IDF)
    # Using built-in english stop words to avoid NLTK download dependency issues
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # 4. Model Training
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    
    # 5. Evaluation
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return vectorizer, model, acc, report, df_clean

# ==========================================
# UI COMPONENTS
# ==========================================

def login_registration_page():
    """Handles the Mild Frontend for Login & Registration."""
    st.markdown("<h1 style='text-align: center;'>🧠 NLP Sentiment Analysis Platform</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Please login or create an account to access the system.</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            st.subheader("Login to your account")
            login_user = st.text_input("Username", key="login_user")
            login_pass = st.text_input("Password", type="password", key="login_pass")
            
            if st.button("Login", use_container_width=True):
                if login_user in st.session_state.users_db and st.session_state.users_db[login_user] == login_pass:
                    st.session_state.logged_in = True
                    st.session_state.current_user = login_user
                    st.success("Login successful! Redirecting...")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
                    
        with tab2:
            st.subheader("Create a new account")
            reg_user = st.text_input("New Username", key="reg_user")
            reg_pass = st.text_input("New Password", type="password", key="reg_pass")
            reg_pass_confirm = st.text_input("Confirm Password", type="password", key="reg_pass_confirm")
            
            if st.button("Register", use_container_width=True):
                if reg_user in st.session_state.users_db:
                    st.error("Username already exists. Please choose another.")
                elif reg_pass != reg_pass_confirm:
                    st.error("Passwords do not match.")
                elif len(reg_user) < 3 or len(reg_pass) < 4:
                    st.error("Username must be at least 3 characters and password at least 4 characters.")
                else:
                    st.session_state.users_db[reg_user] = reg_pass
                    st.success("Registration successful! You can now log in.")

def main_app():
    """The main dashboard and NLP tool."""
    # Sidebar Navigation
    with st.sidebar:
        st.markdown(f"### 👋 Welcome, {st.session_state.current_user}!")
        st.divider()
        st.markdown("### Navigation")
        page = st.radio("Go to:", ["📊 Data & EDA", "⚙️ Model Training Pipeline", "🔍 Live Sentiment Analyzer", "📁 Batch Processing"])
        st.divider()
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.current_user = ""
            st.rerun()

    # Load Data securely
    df = load_data()
    if df is None:
        st.error("⚠️ Dataset not found! Please ensure 'Customer_Sentiment.csv' is placed in the same directory as this script.")
        st.stop()

    # Train/Load model behind the scenes
    with st.spinner("Initializing NLP Engine & Training Model..."):
        vectorizer, model, accuracy, report, df_clean = train_nlp_model(df)

    # ----------------------------------------
    # PAGE 1: Data & EDA
    # ----------------------------------------
    if page == "📊 Data & EDA":
        st.title("Exploratory Data Analysis (EDA)")
        st.write("Overview of the customer review dataset.")
        
        # Top metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", len(df))
        c2.metric("Clean Records", len(df_clean))
        c3.metric("Product Categories", df['product_category'].nunique())
        c4.metric("Platforms", df['platform'].nunique())

        st.subheader("1. Raw Dataset Sample")
        st.dataframe(df.sample(15), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("2. Sentiment Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.countplot(data=df_clean, x='sentiment', palette='viridis', order=['positive', 'neutral', 'negative'], ax=ax)
            plt.title("Count of Sentiments")
            st.pyplot(fig)
            
        with col2:
            st.subheader("3. Ratings by Sentiment")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.boxplot(data=df_clean, x='sentiment', y='customer_rating', palette='Set2', order=['positive', 'neutral', 'negative'], ax=ax2)
            plt.title("Distribution of Customer Ratings per Sentiment")
            st.pyplot(fig2)

        st.subheader("4. Sentiment Across Platforms")
        fig3, ax3 = plt.subplots(figsize=(12, 5))
        sns.countplot(data=df_clean, x='platform', hue='sentiment', palette='viridis', ax=ax3)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig3)

    # ----------------------------------------
    # PAGE 2: Model Training Pipeline
    # ----------------------------------------
    elif page == "⚙️ Model Training Pipeline":
        st.title("NLP Model Pipeline & Metrics")
        
        st.markdown("""
        ### Pipeline Architecture
        1. **Data Cleaning:** Removed missing values.
        2. **Text Preprocessing:** Lowercasing, Regex extraction (removed punctuation, symbols, URLs, numbers).
        3. **Feature Engineering:** TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization (Max Features: 5000, English Stopwords removed).
        4. **Model:** Multinomial Logistic Regression.
        """)

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Model Performance")
            st.metric("Overall Accuracy", f"{accuracy * 100:.2f}%")
            
            st.write("**Text Preprocessing Example:**")
            sample_text = df['review_text'].iloc[0]
            st.info(f"**Original:** {sample_text}")
            st.success(f"**Cleaned:** {preprocess_text(sample_text)}")

        with col2:
            st.subheader("Classification Report")
            report_df = pd.DataFrame(report).transpose()
            # Drop the support column for cleaner display of percentages
            st.dataframe(report_df.style.background_gradient(cmap='Greens', subset=['precision', 'recall', 'f1-score']))

    # ----------------------------------------
    # PAGE 3: Live Sentiment Analyzer
    # ----------------------------------------
    elif page == "🔍 Live Sentiment Analyzer":
        st.title("Live Text Sentiment Analyzer")
        st.write("Type or paste a customer review below to test the intelligent NLP model.")

        user_input = st.text_area("Customer Review:", height=150, placeholder="E.g., I absolutely loved the packaging and the delivery was so fast! Highly recommended.")
        
        if st.button("Analyze Sentiment", type="primary"):
            if user_input.strip() == "":
                st.warning("Please enter some text to analyze.")
            else:
                with st.spinner("Analyzing..."):
                    # Preprocess and Predict
                    clean_input = preprocess_text(user_input)
                    vec_input = vectorizer.transform([clean_input])
                    prediction = model.predict(vec_input)[0]
                    probabilities = model.predict_proba(vec_input)[0]
                    classes = model.classes_
                    
                    # Formatting logic based on sentiment
                    if prediction == 'positive':
                        color = "green"
                        emoji = "✅"
                        action = "Send a 'Thank You' email and invite them to leave a review on social media."
                    elif prediction == 'negative':
                        color = "red"
                        emoji = "⚠️"
                        action = "Escalate this ticket to the Customer Support team immediately for resolution."
                    else:
                        color = "gray"
                        emoji = "😐"
                        action = "Log the feedback for future product improvement discussions."

                    # Display Results Creatively
                    st.markdown("### Analysis Results")
                    
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        st.markdown(f"<h2 style='color: {color};'>{emoji} {prediction.capitalize()}</h2>", unsafe_allow_html=True)
                        st.write(f"**Confidence Score:** {max(probabilities)*100:.1f}%")
                        
                        st.markdown("#### 💡 Smart Business Recommendation:")
                        st.info(action)
                        
                    with c2:
                        # Probability Bar Chart
                        prob_df = pd.DataFrame({'Sentiment': classes, 'Probability': probabilities})
                        fig, ax = plt.subplots(figsize=(5, 3))
                        sns.barplot(x='Probability', y='Sentiment', data=prob_df, palette=['red', 'gray', 'green'], ax=ax)
                        ax.set_xlim(0, 1)
                        plt.title("Model Confidence Distribution")
                        st.pyplot(fig)

    # ----------------------------------------
    # PAGE 4: Batch Processing
    # ----------------------------------------
    elif page == "📁 Batch Processing":
        st.title("Batch Sentiment Processing")
        st.write("Upload a CSV file containing a column named `review_text` to process multiple reviews at once.")
        
        uploaded_file = st.file_uploader("Upload CSV", type=['csv'])
        
        if uploaded_file is not None:
            batch_df = pd.read_csv(uploaded_file)
            
            if 'review_text' not in batch_df.columns:
                st.error("The uploaded CSV MUST contain a column named 'review_text'.")
            else:
                st.success(f"File uploaded successfully! Found {len(batch_df)} rows.")
                
                if st.button("Run Batch Analysis"):
                    with st.spinner("Processing records..."):
                        # Process
                        batch_df['cleaned_text'] = batch_df['review_text'].apply(preprocess_text)
                        
                        # Handle completely empty text rows so model doesn't crash
                        batch_df['cleaned_text'] = batch_df['cleaned_text'].replace("", "empty")
                        
                        vec_batch = vectorizer.transform(batch_df['cleaned_text'])
                        predictions = model.predict(vec_batch)
                        
                        batch_df['predicted_sentiment'] = predictions
                        
                        # Drop the temporary clean text
                        batch_df = batch_df.drop(columns=['cleaned_text'])
                        
                        st.write("### Processing Complete!")
                        st.dataframe(batch_df.head(10))
                        
                        # Allow download
                        csv_data = batch_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Annotated Results",
                            data=csv_data,
                            file_name='batch_sentiment_results.csv',
                            mime='text/csv',
                        )

# ==========================================
# MAIN APP ROUTER
# ==========================================
if __name__ == "__main__":
    # Ensure dataset exists warning (will bypass if uploaded)
    if not os.path.exists("Customer_Sentiment.csv"):
        st.warning("⚠️ For the app to function properly, please ensure `Customer_Sentiment.csv` is uploaded and in the same directory as this script.")

    if st.session_state.logged_in:
        main_app()
    else:
        login_registration_page()
