import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from joblib import dump
from features import basic_features

# data.csv must have columns: prompt, essay, grade_level, assignment_type, grade
df = pd.read_csv("data.csv")

X = []
y = df["grade"].tolist()

for _, row in df.iterrows():
    feats = basic_features(row["essay"], row.get("prompt",""))
    feats["grade_level"] = row.get("grade_level","11")
    feats["assignment_type"] = row.get("assignment_type","essay")
    X.append(feats)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([
    ("vec", DictVectorizer(sparse=False)),
    ("clf", LogisticRegression(max_iter=2000))
])

model.fit(X_train, y_train)
pred = model.predict(X_test)

print(classification_report(y_test, pred))

dump(model, "grade_model.joblib")
print("Saved grade_model.joblib")