import pandas as pd
import joblib
from pathlib import Path

from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# =========================
# PATHS
# =========================

DATA_PATH = Path("data/final_employee_attrition.csv")
MODEL_PATH = Path("models/logistic_regression.pkl")

OUTPUT_DIR = Path("outputs/model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# LOAD DATA + MODEL
# =========================

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Attrition"])
y = df["Attrition"]

model = joblib.load(MODEL_PATH)


# =========================
# PREDICTIONS
# =========================

predictions = model.predict(X)


# =========================
# CONFUSION MATRIX
# =========================

cm = confusion_matrix(y, predictions)

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    ["No Attrition", "Attrition"]
)

plt.yticks(
    [0, 1],
    ["No Attrition", "Attrition"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix_logistic_regression.png"
)

plt.close()


# =========================
# PERMUTATION IMPORTANCE
# =========================

importance = permutation_importance(
    model,
    X,
    y,
    n_repeats=5,
    random_state=42,
    scoring="f1"
)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance.importances_mean
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

importance_df.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)


# =========================
# OUTPUT
# =========================

print("=" * 50)
print("MODEL EVALUATION COMPLETED!")
print("=" * 50)

print("\nConfusion Matrix:")
print(cm)

print("\nTop 10 Important Features:")

print(
    importance_df.head(10).to_string(index=False)
)

print(
    "\nSaved confusion matrix to:",
    OUTPUT_DIR / "confusion_matrix_logistic_regression.png"
)

print(
    "Saved feature importance to:",
    OUTPUT_DIR / "feature_importance.csv"
)