# 📧 SMS Spam Detection using Machine Learning

This project focuses on detecting spam messages in SMS data using text classification techniques.  
By leveraging Natural Language Processing (NLP) and machine learning algorithms, the model classifies incoming messages as **spam** or **ham** (legitimate), enabling better filtering and user protection.



## 🚀 Features

### 🧹 Data Preprocessing
- Loaded and cleaned SMS dataset
- Removed irrelevant characters, stopwords, and punctuation
- Converted all text to lowercase for uniformity
- Applied **TF-IDF Vectorization** for feature extraction

### 🤖 Model Training
- Implemented **Logistic Regression** and **Naive Bayes** classifiers
- Used **TF-IDF features** for high-dimensional text representation

### 📊 Model Evaluation
- Evaluated models using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC Score

### 💾 Model Saving
- Saved trained models and vectorizer for future predictions

---

## 🛠 Tech Stack
- **Python**
- **Pandas**
- **NumPy**
- **scikit-learn**
- **NLTK** (text preprocessing)
- **Matplotlib** / **Seaborn**

---

## 📂 Dataset
The dataset contains SMS messages labeled as either `spam` or `ham`.  

**Dataset Source:** [Provided for CodSoft Internship (SMS Spam Collection Dataset)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

---

## 📊 Model Workflow
1. Load dataset
2. Preprocess messages (cleaning & tokenization)
3. Convert text to numerical features using **TF-IDF**
4. Train models: Logistic Regression and Naive Bayes
5. Evaluate performance metrics
6. Save models and vectorizer



## 📌 Output Example

Model Evaluation Results:
| Model              | Accuracy | Precision | Recall   | F1 Score | ROC-AUC  |
|--------------------|----------|-----------|----------|----------|----------|
| NaiveBayes         | 0.968610 | 1.0       | 0.765101 | 0.866920 | 0.989453 |
| LogisticRegression | 0.967713 | 1.0       | 0.758389 | 0.862595 | 0.987564 |
