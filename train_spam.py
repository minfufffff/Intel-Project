import pandas as pd
import pickle
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# โหลด dataset
df = pd.read_csv('data/SMSSpamCollection', sep='\t', header=None)
df.columns = ['label','text']

# clean
def clean(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

df['text'] = df['text'].apply(clean)
df['label'] = df['label'].map({'ham':0,'spam':1})

# TF-IDF
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['text']).toarray()
y = df['label']

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# model
model = LogisticRegression()
model.fit(X_train, y_train)

# predict
pred = model.predict(X_test)

# accuracy
acc = accuracy_score(y_test, pred)
print(f"Spam Model Accuracy: {acc:.4f}")

# save
pickle.dump(model, open('models/spam_model.pkl','wb'))
pickle.dump(tfidf, open('models/tfidf.pkl','wb'))