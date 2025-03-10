# 🐦 Twitter Sentiment Analysis using NLP

![Python](https://img.shields.io/badge/Python-3.7-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Natural%20Language%20Processing-green.svg?style=for-the-badge)
![ML](https://img.shields.io/badge/Machine%20Learning-Classification-orange.svg?style=for-the-badge)
![Accuracy](https://img.shields.io/badge/Accuracy-95%25-success.svg?style=for-the-badge)

A machine learning project to detect hate speech and offensive language in Twitter posts using Natural Language Processing techniques. This model classifies tweets as either containing racist/sexist content or being neutral.

## 🎯 Objective

This project aims to:

- Detect and predict whether a tweet contains hate speech (racist or sexist sentiment)
- Develop a binary classification model with high accuracy
- Explore NLP preprocessing techniques for social media text
- Compare performance of different ML algorithms on text classification

## 📊 Dataset

The analysis is based on a labeled dataset from Analytics Vidhya containing:
- 31,962 tweets with binary labels
- Label '1': Tweet contains racist/sexist content
- Label '0': Tweet is neutral

## 🔍 Methodology

### Text Preprocessing

1. **Cleaning**
   - Removing URLs, HTML tags, and special characters
   - Converting text to lowercase
   - Removing stopwords and punctuation

2. **Tokenization**
   - Breaking down tweets into individual words/tokens

3. **Feature Engineering**
   - TF-IDF Vectorization
   - Bag of Words representation
   - N-gram features

### Model Development

The project implements and compares multiple classification algorithms:

- **Logistic Regression**: Achieved 95% accuracy
- **Decision Tree**: Achieved 94% accuracy
- **Support Vector Machines** (supplementary)
- **Naive Bayes** (supplementary)

### Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 95% | 93% | 96% | 94% |
| Decision Tree | 94% | 92% | 94% | 93% |

## 🖥️ Implementation

```python
# Sample code for text preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def preprocess_text(tweet):
    # Remove URLs
    tweet = re.sub(r'http\S+', '', tweet)
    # Convert to lowercase
    tweet = tweet.lower()
    # Remove special characters
    tweet = re.sub(r'[^\w\s]', '', tweet)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(tweet)
    tweet = [word for word in word_tokens if word not in stop_words]
    
    return ' '.join(tweet)

# Feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_features = vectorizer.fit_transform(processed_tweets)

# Model training
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_train, X_test, y_train, y_test = train_test_split(X_features, labels, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
predictions = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, predictions)}")
print(classification_report(y_test, predictions))
```

## 🔑 Key Findings

- Logistic Regression performed marginally better than Decision Tree for this task
- TF-IDF vectorization proved more effective than simple Bag of Words
- Including bigrams and trigrams improved model performance
- Common hate speech patterns could be identified through feature importance analysis
- Preprocessing steps significantly impacted model accuracy

## 🛠️ Resources Used

- **Python Version:** 3.7
- **Packages:** pandas, numpy, scikit-learn, matplotlib, seaborn, nltk
- **IDE:** Jupyter Notebook
- **Dataset:** [Twitter Sentiment Analysis - Analytics Vidhya](https://www.kaggle.com/dv1453/twitter-sentiment-analysis-analytics-vidya)

## 🔮 Future Improvements

- Implement more advanced NLP techniques (BERT, transformers)
- Enhance preprocessing for social media text (emojis, hashtags)
- Experiment with ensemble methods for improved accuracy
- Expand to multi-class classification (neutral, hate speech, offensive language)
- Deploy model as a web application for real-time analysis

## 📚 References

- Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python. O'Reilly Media.
- Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017). Automated Hate Speech Detection and the Problem of Offensive Language.
- Scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)

## 👨‍💻 Author

Dishant - [GitHub Profile](https://github.com/Dishant27)

---

**Note**: This project is for educational purposes and demonstrates NLP techniques for detecting hate speech in social media content.
