# 🧠 NLP Sentiment Analysis Platform

An intelligent, end-to-end web application designed to automatically analyze customer reviews and categorize their sentiments as **Positive**, **Negative**, or **Neutral**.

This project combines Natural Language Processing (NLP) with a user-friendly frontend to help businesses derive actionable insights from customer feedback.

---

## ✨ Key Features

- 🔐 **User Authentication**: Secure, session-based login and registration system.  
- 📊 **Exploratory Data Analysis (EDA)**: Interactive dashboard featuring visualizations of sentiment distributions, ratings, and platform comparisons.  
- ⚙️ **Automated NLP Pipeline**: Built-in text cleaning (removing URLs, punctuation), TF-IDF vectorization, and a trained Logistic Regression model.  
- 🔍 **Live Sentiment Analyzer**: Test individual reviews in real-time. It provides:  
  - The predicted sentiment (Positive/Neutral/Negative).  
  - A confidence score with a probability distribution chart.  
  - Smart Business Recommendations based on the result (e.g., escalating negative reviews to support).  
- 📁 **Batch Processing**: Upload a CSV of raw customer reviews, process them in bulk, and instantly download the fully annotated results.  

---

## 🛠️ Technology Stack

- **Frontend & UI**: Streamlit  
- **Machine Learning & NLP**: Scikit-Learn (Logistic Regression, TF-IDF)  
- **Data Manipulation**: Pandas, NumPy  
- **Data Visualization**: Matplotlib, Seaborn  
- **Text Processing**: Python `re` (Regex)  

---

## 🚀 Installation & Setup

Follow these steps to run the project locally:

1. **Clone the repository** (or download the project files):  
   Ensure you have `app.py`, `requirements.txt`, and the dataset `Customer_Sentiment.csv` in the same directory.

2. **Install dependencies**:  
   It is recommended to use a virtual environment. Install the required packages using:  
   ```bash
   pip install -r requirements.txt
Run the Application:
```bash
python -m streamlit run app.py
```
Access the App:
The app will automatically open in your default web browser (usually at ```http://localhost:8501```).
Create a new account or log in to get started!
