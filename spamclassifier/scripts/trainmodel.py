import os
import urllib.request
import zipfile
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("1. Downloading SMS Spam Collection dataset...")
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)
zip_path = os.path.join(data_dir, "smsspamcollection.zip")

if not os.path.exists(zip_path):
    urllib.request.urlretrieve(url, zip_path)

print("2. Extracting dataset...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(data_dir)

csv_path = os.path.join(data_dir, "SMSSpamCollection")

print("3. Loading data into memory...")
# The dataset is tab-separated: label \t text
df = pd.read_csv(csv_path, sep='\t', header=None, names=['label', 'text'])

# Map labels to binary: ham=0, spam=1
df['label_num'] = df.label.map({'ham':0, 'spam':1})

X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label_num'], test_size=0.2, random_state=42)

model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
os.makedirs(model_dir, exist_ok=True)

# SECTION 1: Logistic Regression
print("\n--- Section 1: Logistic Regression ---")
lr_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression())
])
print("Training Logistic Regression model...")
lr_pipeline.fit(X_train, y_train)

lr_preds = lr_pipeline.predict(X_test)
lr_acc = accuracy_score(y_test, lr_preds)
print(f"Logistic Regression Accuracy: {lr_acc:.4f}")

lr_path = os.path.join(model_dir, 'spam_lr_model.joblib')
joblib.dump(lr_pipeline, lr_path)
print(f"Logistic Regression model saved to {lr_path}")


# SECTION 2: Naive Bayes
print("\n--- Section 2: Naive Bayes ---")
nb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', MultinomialNB())
])
print("Training Naive Bayes model...")
nb_pipeline.fit(X_train, y_train)

nb_preds = nb_pipeline.predict(X_test)
nb_acc = accuracy_score(y_test, nb_preds)
print(f"Naive Bayes Accuracy: {nb_acc:.4f}")

nb_path = os.path.join(model_dir, 'spam_nb_model.joblib')
joblib.dump(nb_pipeline, nb_path)
print(f"Naive Bayes model saved to {nb_path}")

print("\nAll training complete.")
