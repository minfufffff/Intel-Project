import pandas as pd
import pickle
import re
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


df = pd.read_csv('data/SMSSpamCollection', sep='\t', header=None)
df.columns = ['label','text']


def clean(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' URL ', text)
    text = re.sub(r'\d+', ' NUM ', text)
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    return text

df['text'] = df['text'].apply(clean)
df['label'] = df['label'].map({'ham':0,'spam':1})


def has_link(text):
    return int('url' in text)

def has_money(text):
    keywords = ['cash','prize','win','free','reward']
    return int(any(k in text for k in keywords))

df['has_link'] = df['text'].apply(has_link)
df['has_money'] = df['text'].apply(has_money)


tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words='english'
)

X_text = tfidf.fit_transform(df['text']).toarray()
X_extra = df[['has_link','has_money']].values

# รวม features
X = np.hstack((X_text, X_extra))
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


model = LogisticRegression(
    class_weight='balanced',
    max_iter=1000
)

model.fit(X_train, y_train)


pred = model.predict(X_test)

acc = accuracy_score(y_test, pred)
print(f"Accuracy: {acc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, pred))


pickle.dump(model, open('models/spam_model.pkl','wb'))
pickle.dump(tfidf, open('models/tfidf.pkl','wb'))