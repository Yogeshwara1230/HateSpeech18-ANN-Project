from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

from sklearn.utils.class_weight import compute_class_weight


# =====================================================
# 1. CONFIGURATION
# =====================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = OUTPUT_DIR / "improved_experiment"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("IMPROVED BiLSTM HATE SPEECH CLASSIFICATION")
print("=" * 65)


# =====================================================
# 2. LOAD DATA
# =====================================================

data = np.load(OUTPUT_DIR / "prepared_data.npz")

X_train = data["X_train"].astype("int32")
X_val = data["X_val"].astype("int32")
X_test = data["X_test"].astype("int32")

y_train = data["y_train"].astype("int32")
y_val = data["y_val"].astype("int32")
y_test = data["y_test"].astype("int32")

with open(
    OUTPUT_DIR / "preprocessing_config.json",
    "r",
    encoding="utf-8"
) as file:
    config = json.load(file)

class_names = config["classes"]
num_classes = len(class_names)
vocabulary_size = config["vocabulary_size"]

print("Classes:", class_names)
print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))


# =====================================================
# 3. CLASS WEIGHTS
# =====================================================

class_ids = np.unique(y_train)

weights = compute_class_weight(
    class_weight="balanced",
    classes=class_ids,
    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in zip(class_ids, weights)
}

print("\nClass weights:")
for class_id, weight in class_weights.items():
    print(f"{class_names[class_id]}: {weight:.4f}")


# =====================================================
# 4. BUILD IMPROVED MODEL
# =====================================================

sequence_length = X_train.shape[1]
embedding_dim = 64

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(
            shape=(sequence_length,),
            dtype="int32"
        ),

        tf.keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=embedding_dim,
            mask_zero=True
        ),

        tf.keras.layers.SpatialDropout1D(0.25),

        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32)
        ),

        tf.keras.layers.Dropout(0.4),

        tf.keras.layers.Dense(
            32,
            activation="relu"
        ),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(
            num_classes,
            activation="softmax"
        )
    ],
    name="BiLSTM_Improved"
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =====================================================
# 5. TRAIN MODEL
# =====================================================

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(REPORT_DIR / "BiLSTM_Improved.keras"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    tf.keras.callbacks.CSVLogger(
        str(REPORT_DIR / "training_log.csv")
    )
]

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=64,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)


# =====================================================
# 6. EVALUATE ON TEST DATA
# =====================================================

print("\nEvaluating the best saved model...")

best_model = tf.keras.models.load_model(
    REPORT_DIR / "BiLSTM_Improved.keras"
)

probabilities = best_model.predict(
    X_test,
    batch_size=64,
    verbose=0
)

predictions = np.argmax(probabilities, axis=1)

accuracy = accuracy_score(y_test, predictions)

precision, recall, f1, _ = (
    precision_recall_fscore_support(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )
)

print("\nFINAL TEST RESULTS")
print("=" * 65)
print(f"Accuracy:        {accuracy:.4f}")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall:    {recall:.4f}")
print(f"Macro F1-score:  {f1:.4f}")

report = classification_report(
    y_test,
    predictions,
    labels=list(range(num_classes)),
    target_names=class_names,
    zero_division=0,
    digits=4
)

print("\nCLASSIFICATION REPORT")
print(report)

matrix = confusion_matrix(
    y_test,
    predictions,
    labels=list(range(num_classes))
)

print("CONFUSION MATRIX")
print(matrix)


# =====================================================
# 7. SAVE RESULTS
# =====================================================

with open(
    REPORT_DIR / "classification_report.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write("BiLSTM Improved - Test Results\n\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Macro Precision: {precision:.4f}\n")
    file.write(f"Macro Recall: {recall:.4f}\n")
    file.write(f"Macro F1-score: {f1:.4f}\n\n")
    file.write(report)
    file.write("\nConfusion Matrix:\n")
    file.write(np.array2string(matrix))

pd.DataFrame(
    matrix,
    index=class_names,
    columns=class_names
).to_csv(
    REPORT_DIR / "confusion_matrix.csv"
)

pd.DataFrame(
    [
        {
            "Model": "BiLSTM_Improved",
            "Accuracy": accuracy,
            "Macro Precision": precision,
            "Macro Recall": recall,
            "Macro F1-score": f1,
            "Best Validation Loss": min(
                history.history["val_loss"]
            ),
            "Epochs Trained": len(history.history["loss"])
        }
    ]
).to_csv(
    REPORT_DIR / "model_results.csv",
    index=False
)

print("\nModel and reports saved in:")
print(REPORT_DIR)

print("\nImproved training experiment completed.")