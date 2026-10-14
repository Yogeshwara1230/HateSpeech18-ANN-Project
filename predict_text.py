from pathlib import Path
import json

import numpy as np
import tensorflow as tf


# ==================================================
# 1. PROJECT PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = (
    OUTPUT_DIR
    / "improved_experiment"
    / "BiLSTM_Improved.keras"
)

CONFIG_PATH = OUTPUT_DIR / "preprocessing_config.json"
VOCAB_PATH = OUTPUT_DIR / "vocabulary.txt"


# ==================================================
# 2. LOAD CONFIGURATION
# ==================================================

with open(CONFIG_PATH, "r", encoding="utf-8") as file:
    config = json.load(file)

class_names = config["classes"]
max_tokens = config["max_tokens"]
sequence_length = config["sequence_length"]

print("=" * 55)
print("HATE SPEECH CLASSIFICATION USING BiLSTM")
print("=" * 55)

print("\nLoading model...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)


# ==================================================
# 3. RESTORE THE ORIGINAL VOCABULARY
# ==================================================

if not VOCAB_PATH.exists():
    raise FileNotFoundError(
        f"Vocabulary file not found: {VOCAB_PATH}"
    )

with open(VOCAB_PATH, "r", encoding="utf-8") as file:
    vocabulary = file.read().splitlines()

# TextVectorization reserves entries for padding and
# unknown words. They are already present in the file.
if len(vocabulary) < 2:
    raise ValueError("The saved vocabulary is incomplete.")

if vocabulary[0] != "" or vocabulary[1] != "[UNK]":
    raise ValueError(
        "Unexpected vocabulary format. "
        "Check outputs/vocabulary.txt."
    )

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=max_tokens,
    output_mode="int",
    output_sequence_length=sequence_length,
    standardize="lower_and_strip_punctuation"
)

# Restore the vocabulary learned during training.
vectorizer.set_vocabulary(vocabulary[2:])

print("Model loaded successfully.")
print("Classes:", ", ".join(class_names))


# ==================================================
# 4. PREDICTION FUNCTION
# ==================================================

def predict_text(text):
    text = text.strip()

    if not text:
        print("Please enter a non-empty sentence.")
        return

    # Apply the same text vectorization used in training.
    sequence = vectorizer(
        tf.constant([text])
    ).numpy()

    # Predict probabilities for the four classes.
    probabilities = model.predict(
        sequence,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index])

    print("\n" + "-" * 55)
    print("INPUT TEXT:")
    print(text)

    print("\nPREDICTED CLASS:")
    print(predicted_class)

    print(f"\nModel confidence: {confidence:.2%}")

    print("\nScores for all classes:")

    for index, name in enumerate(class_names):
        print(f"{name:12s}: {probabilities[index]:.2%}")

    print("-" * 55)

    print(
        "Note: Confidence is the model's predicted probability, "
        "not a guarantee that the classification is correct."
    )


# ==================================================
# 5. INTERACTIVE INPUT
# ==================================================

print("\nEnter a sentence to classify.")
print("Type 'exit' to stop the program.")

while True:
    try:
        user_text = input("\nEnter text: ")

        if user_text.strip().lower() == "exit":
            print("\nPrediction program closed.")
            break

        predict_text(user_text)

    except (KeyboardInterrupt, EOFError):
        print("\nPrediction program closed.")
        break
