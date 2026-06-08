import pandas as pd
import sqlite3
import os
import matplotlib.pyplot as plt
import re
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score, recall_score,f1_score,accuracy_score,roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

path = r"C:\Users\Ольга\OneDrive\Рабочий стол\Диплом\tracks.db"

conn = sqlite3.connect(path)

dataset = pd.read_sql_query("""
SELECT track_name, artist_name,duration_ms,
danceability, energy, valence, tempo,
acousticness, instrumentalness, liveness,
speechiness, loudness,key,mode,time_signature
FROM extracted
""", conn)

my_df = pd.read_csv(r"C:\Users\Ольга\OneDrive\Рабочий стол\Диплом\train2.csv",sep="\t")
my_df.columns = my_df.columns.str.lower().str.strip()

def clean(text):
    text = text.lower()
    text = text.replace("feat.", "")
    text = text.replace("ft.", "")
    text = text.replace("remix", "")
    return text.strip()

my_df["track_name"] = my_df["track_name"].apply(clean)
my_df["artist_name"] = my_df["artist_name"].apply(clean)

dataset["track_name"] = dataset["track_name"].apply(clean)
dataset["artist_name"] = dataset["artist_name"].apply(clean)

merged = my_df.merge(
    dataset,
    on=["track_name"],
    how="left"
)

merged = merged.drop_duplicates(
    subset=["track_name"]
)

merged["artist_name"] = merged["artist_name_x"]
merged = merged.drop(columns=["artist_name_y"])
merged = merged.drop(columns=["artist_name_x"])

neg = pd.read_csv(
    r"C:\Users\Ольга\OneDrive\Рабочий стол\halfdiz.csv",encoding="utf-8",
    sep=";"
)

neg.columns = neg.columns.str.lower().str.strip()

neg["track_name"] = neg["track_name"].apply(clean)
neg["artist_name"] = neg["artist_name"].apply(clean)

neg_merged = neg.merge(
    dataset,
    on=["track_name", "artist_name"],
    how="left"
)

merged["label"] = 1
neg_merged["label"] = 0

merged = merged.dropna(subset=["energy"])
neg_merged = neg_merged.dropna(subset=["energy"])
neg_merged = neg_merged.drop_duplicates(
    subset=["track_name", "artist_name"]
)

data = pd.concat([merged, neg_merged], ignore_index=True)

data["duration_ms"] = data["duration_ms_x"].combine_first(data["duration_ms"])
data = data.drop(columns=["duration_ms_x", "duration_ms_y"])

data["energy_dance"] = data["energy"] * data["danceability"]
data["mood"] = data["valence"] * data["energy"]
data["softness"] = data["acousticness"] * (1 - data["energy"])

features = [
    "artist_name",
    "duration_ms",
    "energy",
    "danceability",
    "valence",
    "tempo",
    "acousticness",
    "instrumentalness",
    "loudness",
    "speechiness",
    "liveness",
    "energy_dance",
    "mood",
    "softness",
    "key",
    "genre",
    "year",
    "explicit",
    "mode",
    "time_signature"
]

X = data[features]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state = 42,stratify = y)
X_train, X_val, y_train, y_val = train_test_split(

    X_train,

    y_train,

    test_size=0.15,

    stratify=y_train,

    random_state=42

)

X_train = X_train.copy()
X_test = X_test.copy()
X_val = X_val.copy()


artist_counts = X_train["artist_name"].value_counts()

X_train["artist_count"] = X_train["artist_name"].map(artist_counts)
X_test["artist_count"] = X_test["artist_name"].map(artist_counts)
X_val["artist_count"] = X_val["artist_name"].map(artist_counts)

num_features = [
    "duration_ms", "energy", "danceability",
    "valence", "tempo", "acousticness",
    "instrumentalness", "loudness",
    "speechiness", "liveness",
    "artist_count","key","year","mode",
    "time_signature"
]

cat_features = ["genre", "explicit"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ]
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)
X_val_encoded = preprocessor.transform(X_val)

X_train_encoded = np.nan_to_num(X_train_encoded)
X_test_encoded = np.nan_to_num(X_test_encoded)
X_val_encoded = np.nan_to_num(X_val_encoded)

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)
X_val_scaled = scaler.transform(X_val_encoded)

model = RandomForestClassifier(n_estimators = 200,
max_depth = 10,
min_samples_leaf = 5,
random_state = 42,
class_weight = "balanced")

#model = MLPClassifier(hidden_layer_sizes = 1000,max_iter = 300,random_state = 42)

cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    model,
    X_train_scaled,
    y_train,
    cv=cv,
    scoring="roc_auc"
)

model.fit(X_train_scaled,y_train)

print(scores)
print(scores.mean())
print(scores.std())

y_pred = model.predict(X_test_scaled)
y_pred_val = model.predict(X_val_scaled)

train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)

print("TRAIN")
print(classification_report(y_train, train_pred))

print("TEST")
print(classification_report(y_test, test_pred))

precision_v = precision_score(y_val, y_pred_val)
recall_v = recall_score(y_val, y_pred_val)
F1_v =f1_score(y_val, y_pred_val)
accuracy_v = accuracy_score(y_val, y_pred_val)
y_proba_v = model.predict_proba(X_val_scaled)[:, 1]
roc_auc_v = roc_auc_score(y_val, y_proba_v)

print("Accuracy:", accuracy_v)
print("Precision:", precision_v)
print("Recall:", recall_v)
print("F1:", F1_v)
print("ROC-AUC:", roc_auc_v)
print(confusion_matrix(y_val, y_pred_val))

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
F1 =f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
y_proba = model.predict_proba(X_test_scaled)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", F1)
print("ROC-AUC:", roc_auc)
print(confusion_matrix(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap='Greens')
plt.title("Матрица смежности")
plt.xlabel("Предсказанная метка")
plt.ylabel("Истинная метка")
plt.show()

fpr, tpr, thresholds = roc_curve(y_test, y_proba)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC кривая")
plt.legend()
plt.show()

precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
plt.figure(figsize=(8,6))
plt.plot(recall, precision)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall кривая")

plt.show()

feature_names = preprocessor.get_feature_names_out()

importances = model.feature_importances_

feature_importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

feature_importance_df = feature_importance_df.sort_values(
    by="Importance",
    ascending=False
)

top_features = feature_importance_df.head(10)

plt.figure(figsize=(10, 6))
plt.barh(
    top_features["Feature"],
    top_features["Importance"]
)

plt.xlabel("Важность")
plt.ylabel("Признаки")
plt.title("Важность признаков для Random Forest")

plt.gca().invert_yaxis()

plt.show()
