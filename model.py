import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
print("Dataset Head:\n", df.head())
print("\nMissing Values:\n", df.isnull().sum())


sns.countplot(x='DEATH_EVENT', data=df)
plt.title("Class Distribution (0 = Survived, 1 = Died)")
plt.savefig("class_distribution.png")  # saves as image


X = df.drop("DEATH_EVENT", axis=1)
y = df["DEATH_EVENT"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

joblib.dump(model, "heart_failure_model.pkl")
print("\nModel saved as 'heart_failure_model.pkl'")