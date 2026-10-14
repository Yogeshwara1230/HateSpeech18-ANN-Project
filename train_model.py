from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# ============================================
# 1. REPRODUCIBILITY
# ============================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

print("HATE SPEECH CLASSIFICATION - DATA PREPARATION")
print("=" * 50)

# ============================================
# 2. LOAD DATASET
# ============================================

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = (
    BASE_DIR / "data" / "processed" / "hate_speech18_clean.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE}"
    )

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

df = df.dropna(subset=["text", "label"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].reset_index(drop=True)

print("Usable records:", len(df))

# ============================================
# 3. ENCODE FOUR CLASSES
# ============================================

label_encoder = LabelEncoder()

df["label_id"] = label_encoder.fit_transform(df["label"])

class_names = label_encoder.classes_.tolist()
num_classes = len(class_names)

print("\nClass mapping:")

for index, name in enumerate(class_names):
    print(f"{index}: {name}")

# ============================================
# 4. STRATIFIED DATA SPLIT
# ============================================

X = df["text"].to_numpy()
y = df["label_id"].to_numpy()

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=SEED,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=SEED,
    stratify=y_temp
)

print("\nDataset split:")
print("Training:", len(X_train))
print("Validation:", len(X_val))
print("Testing:", len(X_test))

# ============================================
# 5. TEXT VECTORIZATION
# ============================================

MAX_TOKENS = 15000
SEQUENCE_LENGTH = 100

print("\nBuilding text vocabulary...")

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
    standardize="lower_and_strip_punctuation"
)

# Fit vocabulary on training text only
vectorizer.adapt(
    tf.data.Dataset.from_tensor_slices(X_train).batch(128)
)

vocabulary_size = len(vectorizer.get_vocabulary())

print("Vocabulary size:", vocabulary_size)
print("Maximum sequence length:", SEQUENCE_LENGTH)

# Convert texts to integer sequences
X_train_seq = vectorizer(X_train).numpy()
X_val_seq = vectorizer(X_val).numpy()
X_test_seq = vectorizer(X_test).numpy()

# ============================================
# 6. CLASS WEIGHTS
# ============================================

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {
    int(class_id): float(weight)
    for class_id, weight in enumerate(weights)
}

print("\nClass weights:")

for class_id, weight in class_weights.items():
    print(f"{class_names[class_id]}: {weight:.4f}")

# ============================================
# 7. SAVE PREPARED DATA
# ============================================

np.savez_compressed(
    OUTPUT_DIR / "prepared_data.npz",
    X_train=X_train_seq,
    X_val=X_val_seq,
    X_test=X_test_seq,
    y_train=y_train,
    y_val=y_val,
    y_test=y_test
)

# Save vocabulary configuration
config = {
    "max_tokens": MAX_TOKENS,
    "sequence_length": SEQUENCE_LENGTH,
    "vocabulary_size": vocabulary_size,
    "classes": class_names,
    "seed": SEED,
    "class_weights": class_weights
}

with open(
    OUTPUT_DIR / "preprocessing_config.json",
    "w",
    encoding="utf-8"
) as file:
    json.dump(config, file, indent=4)

# Save vocabulary
with open(
    OUTPUT_DIR / "vocabulary.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write("\n".join(vectorizer.get_vocabulary()))

print("\n" + "=" * 50)
print("DATA PREPARATION COMPLETED SUCCESSFULLY")
print("=" * 50)

print("Prepared data:", OUTPUT_DIR / "prepared_data.npz")
print("Configuration:", OUTPUT_DIR / "preprocessing_config.json")
print("Vocabulary:", OUTPUT_DIR / "vocabulary.txt")

print("\nNo neural network has been trained yet.")
print("Next step: build and train the three models.")
