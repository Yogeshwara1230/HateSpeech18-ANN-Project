from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

# ==================================================
# 1. REPRODUCIBILITY
# ==================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 60)
print("HATE SPEECH CLASSIFICATION: MODEL TRAINING")
print("=" * 60)

# ==================================================
# 2. LOAD PREPARED DATA
# ==================================================

data_file = OUTPUT_DIR / "prepared_data.npz"
config_file = OUTPUT_DIR / "preprocessing_config.json"

if not data_file.exists():
    raise FileNotFoundError(
        "Prepared data not found. Run train_model.py first."
    )

with np.load(data_file) as data:
    X_train = data["X_train"]
    X_val = data["X_val"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_val = data["y_val"]
    y_test = data["y_test"]

with open(config_file, "r", encoding="utf-8") as file:
    config = json.load(file)

class_names = config["classes"]
num_classes = len(class_names)
vocabulary_size = config["vocabulary_size"]

print("\nTraining records:", len(X_train))
print("Validation records:", len(X_val))
print("Test records:", len(X_test))
print("Classes:", class_names)

# ==================================================
# 3. CLASS WEIGHTS
# ==================================================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {
    int(index): float(weight)
    for index, weight in enumerate(weights)
}

# ==================================================
# 4. MODEL ARCHITECTURES
# ==================================================

EMBEDDING_DIM = 64
BATCH_SIZE = 64
EPOCHS = 12
LEARNING_RATE = 0.001

def build_baseline():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=EMBEDDING_DIM,
            mask_zero=True
        ),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ], name="ANN_Baseline")

    return model


def build_bilstm(use_dropout=False):
    layers = [
        tf.keras.layers.Input(shape=(X_train.shape[1],)),
        tf.keras.layers.Embedding(
            input_dim=vocabulary_size,
            output_dim=EMBEDDING_DIM,
            mask_zero=True
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32)
        )
    ]

    if use_dropout:
        layers.append(tf.keras.layers.Dropout(0.4))

    layers.extend([
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    model_name = (
        "BiLSTM_Dropout" if use_dropout else "BiLSTM"
    )

    return tf.keras.Sequential(layers, name=model_name)


# ==================================================
# 5. TRAINING AND EVALUATION
# ==================================================

experiments = [
    ("ANN_Baseline", build_baseline),
    ("BiLSTM", lambda: build_bilstm(False)),
    ("BiLSTM_Dropout", lambda: build_bilstm(True))
]

results = []
histories = {}

for experiment_name, builder in experiments:

    print("\n" + "=" * 60)
    print("Training:", experiment_name)
    print("=" * 60)

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(SEED)

    model = builder()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )

    histories[experiment_name] = history.history

    # Evaluate only on the held-out test set
    probabilities = model.predict(
        X_test,
        batch_size=BATCH_SIZE,
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

    print("\nTest results:")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro precision:   {precision:.4f}")
    print(f"Macro recall:      {recall:.4f}")
    print(f"Macro F1-score:    {f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=list(range(num_classes)),
            target_names=class_names,
            zero_division=0
        )
    )

    # Save the trained model
    model_path = OUTPUT_DIR / f"{experiment_name}.keras"
    model.save(model_path)

    # Save confusion matrix
    cm = confusion_matrix(
        y_test,
        predictions,
        labels=list(range(num_classes))
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title(f"Confusion Matrix: {experiment_name}")
    plt.tight_layout()

    cm_path = OUTPUT_DIR / f"{experiment_name}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    results.append({
        "Model": experiment_name,
        "Accuracy": accuracy,
        "Macro Precision": precision,
        "Macro Recall": recall,
        "Macro F1-score": f1,
        "Best Validation Loss": min(history.history["val_loss"]),
        "Epochs Trained": len(history.history["loss"])
    })

    print(f"Saved model: {model_path}")
    print(f"Saved confusion matrix: {cm_path}")

# ==================================================
# 6. SAVE COMPARISON RESULTS
# ==================================================

results_df = pd.DataFrame(results)

results_path = OUTPUT_DIR / "model_comparison.csv"
results_df.to_csv(results_path, index=False)

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)
print(results_df.to_string(index=False))

print("\nResults saved to:", results_path)

# ==================================================
# 7. SAVE TRAINING CURVES
# ==================================================

for metric in ["loss", "accuracy"]:

    plt.figure(figsize=(9, 5))

    for model_name, history in histories.items():
        plt.plot(
            history[metric],
            label=f"{model_name} training"
        )
        plt.plot(
            history[f"val_{metric}"],
            linestyle="--",
            label=f"{model_name} validation"
        )

    plt.xlabel("Epoch")
    plt.ylabel(metric.capitalize())
    plt.title(f"Training and Validation {metric.capitalize()}")
    plt.legend()
    plt.tight_layout()

    plot_path = OUTPUT_DIR / f"training_{metric}.png"
    plt.savefig(plot_path, dpi=150)
    plt.close()

print("\nTraining curves saved.")
print("All experiments completed successfully.")
