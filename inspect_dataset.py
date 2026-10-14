from pathlib import Path
import pandas as pd

print("Starting dataset inspection...")

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "processed" / "hate_speech18_clean.csv"

if not DATA_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

df = pd.read_csv(DATA_FILE)

print("\n" + "=" * 50)
print("HATE SPEECH 18 DATASET INSPECTION")
print("=" * 50)

print(f"\nTotal records: {len(df)}")
print(f"Total columns: {len(df.columns)}")
print(f"Columns: {df.columns.tolist()}")

print("\nMissing values:")
print(df.isnull().sum())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nLabel percentages:")
print((df["label"].value_counts(normalize=True) * 100).round(2))

print("\nText length statistics (words):")
text_lengths = df["text"].astype(str).str.split().str.len()
print(text_lengths.describe())

print("\nExample text from each label:")
for label in sorted(df["label"].unique()):
    example = df.loc[df["label"] == label, "text"].iloc[0]
    print(f"\nLabel: {label}")
    print(f"Text: {example[:200]}")

print("\nDataset inspection completed successfully!")
