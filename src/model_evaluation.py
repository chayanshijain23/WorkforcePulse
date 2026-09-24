import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance


DATA_PATH = Path("data/processed_employee_attrition.csv")
MODEL_PATH = Path("models/logistic_regression.pkl")
OUTPUT_DIR = Path("outputs/model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Attrition"])
y = df["Attrition"]

model = joblib.load(MODEL_PATH)

# Predictions
predictions = model.predict(X)

# Confusion Matrix
cm = confusion_matrix(y, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Attrition", "Attrition"]
)

disp.plot()
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig(
    OUTPUT_DIR / "confusion_matrix_logistic_regression.png"
)
plt.close()

# Permutation Feature Importance
importance = permutation_importance(
    model,
    X,
    y,
    n_repeats=10,
    random_state=42,
    scoring="f1"
)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance.importances_mean
})

feature_importance = feature_importance.sort_values(
    "Importance",
    ascending=False
)

feature_importance.to_csv(
    OUTPUT_DIR / "feature_importance.csv",
    index=False
)

print("\nTop 15 features:")
print(feature_importance.head(15).to_string(index=False))

print("\nConfusion matrix:")
print(cm)

print("\nEvaluation completed successfully!")