# -*- coding: utf-8 -*-
"""Research-topic-Prediction.py
Cleaned up for local execution in VS Code
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, accuracy_score

import nltk
from nltk.corpus import stopwords

# Download NLTK resources (first run only)
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# ============================
# Load datasets (use relative paths)
# ============================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submission_pd = pd.read_csv("submission6.csv")

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Sample shape:", submission_pd.shape)

labels = [
    "Computer Science",
    "Physics",
    "Mathematics",
    "Statistics",
    "Quantitative Biology",
    "Quantitative Finance",
]

# ============================
# Data prep
# ============================
test = test.drop(["ID"], axis=1)

X = train.loc[:, ["TITLE", "ABSTRACT"]]
y = train.loc[:, labels]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, shuffle=True
)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

y_test.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)

y1 = np.array(y_train)
y2 = np.array(y_test)

# ============================
# Cleaning text
# ============================
# Remove punctuations
X_train.replace(r"[^a-zA-Z]", " ", regex=True, inplace=True)
X_test.replace(r"[^a-zA-Z]", " ", regex=True, inplace=True)
test.replace(r"[^a-zA-Z]", " ", regex=True, inplace=True)

# Lowercase
for col in X_train.columns:
    X_train[col] = X_train[col].str.lower()
    X_test[col] = X_test[col].str.lower()
    test[col] = test[col].str.lower()

# Remove one-letter words
X_train["ABSTRACT"] = (
    X_train["ABSTRACT"].str.replace(r"\b\w\b", "").str.replace(r"\s+", " ")
)
X_test["ABSTRACT"] = (
    X_test["ABSTRACT"].str.replace(r"\b\w\b", "").str.replace(r"\s+", " ")
)
test["ABSTRACT"] = (
    test["ABSTRACT"].str.replace(r"\b\w\b", "").str.replace(r"\s+", " ")
)

# Remove multiple blank spaces
X_train = X_train.replace(r"\s+", " ", regex=True)
X_test = X_test.replace(r"\s+", " ", regex=True)
test = test.replace(r"\s+", " ", regex=True)

# Combine TITLE + ABSTRACT
X_train["combined"] = X_train["TITLE"] + " " + X_train["ABSTRACT"]
X_test["combined"] = X_test["TITLE"] + " " + X_test["ABSTRACT"]
test["combined"] = test["TITLE"] + " " + test["ABSTRACT"]

X_train = X_train.drop(["TITLE", "ABSTRACT"], axis=1)
X_test = X_test.drop(["TITLE", "ABSTRACT"], axis=1)
test = test.drop(["TITLE", "ABSTRACT"], axis=1)

# ============================
# Vectorization
# ============================
train_lines = [" ".join(str(x) for x in X_train.iloc[row, :]) for row in range(X_train.shape[0])]
test_lines = [" ".join(str(x) for x in X_test.iloc[row, :]) for row in range(X_test.shape[0])]
predtest_lines = [" ".join(str(x) for x in test.iloc[row, :]) for row in range(test.shape[0])]

countvector = CountVectorizer(ngram_range=(1, 2))
X_train_cv = countvector.fit_transform(train_lines)
X_test_cv = countvector.transform(test_lines)
test_cv = countvector.transform(predtest_lines)

tfidfvector = TfidfTransformer()
X_train_tf = tfidfvector.fit_transform(X_train_cv)
X_test_tf = tfidfvector.transform(X_test_cv)   # fixed (transform only)
test_tf = tfidfvector.transform(test_cv)       # fixed (transform only)

# ============================
# Model training
# ============================
model = LinearSVC(C=0.5, class_weight="balanced", random_state=42)
models = MultiOutputClassifier(model)

models.fit(X_train_tf, y1)
preds = models.predict(X_test_tf)

print("\n=== Classification Report ===")
print(classification_report(y2, preds))
print("Accuracy:", accuracy_score(y2, preds))

# ============================
# Predictions on test set
# ============================
predssv = models.predict(test_tf)

test_with_id = pd.read_csv("test.csv")  # reload with ID
submit = pd.DataFrame({
    "ID": test_with_id.ID,
    "Computer Science": predssv[:, 0],
    "Physics": predssv[:, 1],
    "Mathematics": predssv[:, 2],
    "Statistics": predssv[:, 3],
    "Quantitative Biology": predssv[:, 4],
    "Quantitative Finance": predssv[:, 5],
})

submit.to_csv("submission2.csv", index=False)
print("\nSubmission file 'submission2.csv' created successfully.")