import pandas as pd
from pathlib import Path

# Project paths
DATA_DIR = Path("data")
METADATA_PATH = DATA_DIR / "annotations_metadata.csv"
TEXT_DIR = DATA_DIR / "all_files"

# Load metadata
df = pd.read_csv(METADATA_PATH)

print("=" * 60)
print("HATE SPEECH 18 DATASET VERIFICATION")
print("=" * 60)

# 1. Dataset information
print("\n1. Metadata information")
print("Total metadata records:", len(df))
print("Total text files:", len(list(TEXT_DIR.glob("*.txt"))))
print("Columns:", df.columns.tolist())

# 2. Class distribution
print("\n2. Label distribution")
print(df["label"].value_counts(dropna=False))

# 3. Check whether each metadata record has a text file
df["file_id"] = df["file_id"].astype(str)

df["text_path"] = df["file_id"].apply(
    lambda file_id: TEXT_DIR / f"{file_id}.txt"
)

df["file_exists"] = df["text_path"].apply(
    lambda path: path.is_file()
)

missing_files = df[~df["file_exists"]]

print("\n3. File matching")
print("Records with matching text files:", df["file_exists"].sum())
print("Records with missing text files:", len(missing_files))

if not missing_files.empty:
    print("\nFirst 10 missing file IDs:")
    print(missing_files["file_id"].head(10).to_list())

# 4. Read available text files
def read_text(path):
    try:
        return path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError):
        return ""

df["text"] = df["text_path"].apply(
    lambda path: read_text(path) if path.is_file() else ""
)

empty_text = df["text"].eq("")

print("\n4. Text validation")
print("Empty or unreadable text records:", empty_text.sum())
print("Non-empty text records:", (~empty_text).sum())

# 5. Display examples
print("\n5. Sample records")
print(df[["file_id", "label", "text"]].head(10).to_string(index=False))

print("\n6. Dataset verification completed.")