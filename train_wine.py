import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression

# โหลด dataset
df = pd.read_csv('data/winequality-red.csv', sep=';')

print(df.head())
print(df.columns)

X = df.drop('quality', axis=1)
y = df['quality']

# แปลงเป็น classification
y = (y >= 6).astype(int)

# split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# models
model1 = LogisticRegression(max_iter=1000)
model2 = RandomForestClassifier(n_estimators=100)

model = VotingClassifier([
    ('lr', model1),
    ('rf', model2)
])

# train
model.fit(X_train, y_train)

# predict
pred = model.predict(X_test)

# accuracy
acc = accuracy_score(y_test, pred)
print(f"Wine Model Accuracy: {acc:.4f}")

# save model
pickle.dump(model, open('models/wine_model.pkl','wb'))