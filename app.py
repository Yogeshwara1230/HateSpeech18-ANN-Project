from pathlib import Path
import json

import numpy as np
import streamlit as st
import tensorflow as tf


# ==================================================
# PROJECT PATHS
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
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Hate Speech Classification",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ Hate Speech Classification")
st.subheader("ANN-Based Text Classification Using BiLSTM")

st.write(
    "Enter a sentence to classify it into one of the "
    "four categories learned by the trained model."
)

st.info(
    "This is an experimental machine-learning classifier. "
    "Its predictions may be incorrect, especially for "
    "underrepresented categories."
)


# ==================================================
# LOAD MODEL AND PREPROCESSING
# ==================================================

@st.cache_resource
def load_resources():
    with open(
        CONFIG_PATH, "r", encoding="utf-8"
    ) as file:
        config = json.load(file)

    with open(
        VOCAB_PATH, "r", encoding="utf-8"
    ) as file:
        vocabulary = file.read().splitlines()

    if len(vocabulary) < 2:
        raise ValueError("The vocabulary file is incomplete.")

    if vocabulary[0] != "" or vocabulary[1] != "[UNK]":
        raise ValueError("Unexpected vocabulary format.")

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=config["max_tokens"],
        output_mode="int",
        output_sequence_length=config["sequence_length"],
        standardize="lower_and_strip_punctuation"
    )

    vectorizer.set_vocabulary(vocabulary[2:])

    model = tf.keras.models.load_model(MODEL_PATH)

    return model, vectorizer, config["classes"]


try:
    model, vectorizer, class_names = load_resources()
    st.success("Model loaded successfully.")
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# ==================================================
# USER INPUT
# ==================================================

user_text = st.text_area(
    "Enter a sentence",
    placeholder="Type your sentence here...",
    height=130
)

if st.button("Predict", type="primary", width="stretch"):

    if not user_text.strip():
        st.warning("Please enter a sentence first.")

    else:
        sequence = vectorizer(
            tf.constant([user_text.strip()])
        ).numpy()

        probabilities = model.predict(
            sequence,
            verbose=0
        )[0]

        predicted_index = int(np.argmax(probabilities))
        predicted_class = class_names[predicted_index]
        confidence = float(probabilities[predicted_index])

        st.divider()
        st.subheader("Prediction Result")

        st.metric(
            label="Predicted Category",
            value=predicted_class
        )

        st.metric(
            label="Model Confidence",
            value=f"{confidence:.2%}"
        )

        st.write("### Scores for All Categories")

        for index, category in enumerate(class_names):
            score = float(probabilities[index])

            st.write(f"**{category}** — {score:.2%}")
            st.progress(score)

        st.caption(
            "These scores represent the model's predicted "
            "probabilities, not verified certainty."
        )


# ==================================================
# PROJECT INFORMATION
# ==================================================

with st.expander("About this project"):
    st.write(
        "The project compares an ANN baseline, BiLSTM, "
        "BiLSTM with dropout, and an improved BiLSTM model."
    )

    st.write(
        "Dataset: Hate Speech 18. The classifier predicts "
        "hate, idk/skip, noHate, or relation."
    )

    st.write(
        "The model has limited performance on some minority "
        "classes and should not be used as the sole basis "
        "for moderation decisions."
    )
