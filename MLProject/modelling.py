import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import joblib

# Aktifkan autolog MLflow
mlflow.autolog()

# Load dataset hasil preprocessing
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X_train = train_df.drop('Sales_Class', axis=1)
y_train = train_df['Sales_Class']
X_test = test_df.drop('Sales_Class', axis=1)
y_test = test_df['Sales_Class']

# Set experiment
mlflow.set_experiment("ps4-games-sales-classification")

# Jalankan training dan logging dengan MLflow
with mlflow.start_run(nested=True):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Akurasi: {acc:.4f} | F1-score: {f1:.4f}")

    # Simpan model ke file
    joblib.dump(model, "model.pkl")
