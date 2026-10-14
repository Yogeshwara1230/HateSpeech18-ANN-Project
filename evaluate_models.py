import json
import numpy as np
import pandas as pd
import tensorflow as tf

from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = OUTPUT_DIR / "evaluation_reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Load the prepared dataset
data = np.load(OUTPUT_DIR / "prepared_data.npz")

X_test = data["X_test"]
y_test = data["y_test"]

# Class order used by LabelEncoder
class_names = ["hate", "idk/skip", "noHate", "relation"]
model_names = ["ANN_Baseline", "BiLSTM", "BiLSTM_Dropout"]

results = []

for model_name in model_names:
    print("\n" + "=" * 65)
    print("MODEL:", model_name)
    print("=" * 65)

    model_path = OUTPUT_DIR / f"{model_name}.keras"
    model = tf.keras.models.load_model(model_path)

    probabilities = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="macro", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="macro", zero_division=0
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=class_names,
        zero_division=0,
        digits=4
    )

    matrix = confusion_matrix(
        y_test, y_pred, labels=[0, 1, 2, 3]
    )

    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1-score:  {f1:.4f}")

    print("\nClassification Report:")
    print(report)

    print("Confusion Matrix:")
    print(matrix)

    report_path = REPORT_DIR / f"{model_name}_report.txt"

    with open(report_path, "w", encoding="utf-8") as file:
        file.write(f"Model: {model_name}\n\n")
        file.write(f"Accuracy: {accuracy:.4f}\n")
        file.write(f"Macro Precision: {precision:.4f}\n")
        file.write(f"Macro Recall: {recall:.4f}\n")
        file.write(f"Macro F1-score: {f1:.4f}\n\n")
        file.write(report)
        file.write("\nConfusion Matrix:\n")
        file.write(np.array2string(matrix))

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1-score": f1
    })

comparison = pd.DataFrame(results)
comparison.to_csv(
    REPORT_DIR / "detailed_model_comparison.csv",
    index=False
)

print("\n" + "=" * 65)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("Reports saved in:", REPORT_DIR)
print("=" * 65)
