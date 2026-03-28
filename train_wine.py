import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data/winequality-red.csv', sep=';')

X = df.drop('quality', axis=1)
y = (df['quality'] >= 6).astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🔥 Pipeline + Scaling
model_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=2000))
])

model_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    random_state=42
)

model_lr.fit(X_train, y_train)
model_rf.fit(X_train, y_train)

pred_lr = model_lr.predict(X_test)
pred_rf = model_rf.predict(X_test)

acc_lr = accuracy_score(y_test, pred_lr)
acc_rf = accuracy_score(y_test, pred_rf)

print(f"Logistic Regression Accuracy: {acc_lr:.4f}")
print(f"Random Forest Accuracy: {acc_rf:.4f}")

best_model = model_rf if acc_rf > acc_lr else model_lr

print("Best model selected!")

# save
pickle.dump(best_model, open('models/wine_model.pkl','wb'))