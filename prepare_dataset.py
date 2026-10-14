from pathlib import Path
import pandas as pd

print("Starting dataset preparation...")

# Define project paths
BASE_DIR = Path(__file__).resolve().parent
METADATA_FILE = BASE_DIR / "data" / "annotations_metadata.csv"
TEXT_DIR = BASE_DIR / "data" / "all_files"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "hate_speech18_clean.csv"

# Check that the required files and folders exist
if not METADATA_FILE.exists():
    raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")

if not TEXT_DIR.exists():
    raise FileNotFoundError(f"Text folder not found: {TEXT_DIR}")

# Create the output folder if necessary
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load metadata
print("Loading metadata...")
df = pd.read_csv(METADATA_FILE)

print(f"Total metadata records: {len(df)}")
print(f"Original columns: {df.columns.tolist()}")

# Read text files and match them with metadata
records = []
missing_files = 0
empty_files = 0
unreadable_files = 0

print("Reading text files...")

for _, row in df.iterrows():
    file_id = str(row["file_id"])
    text_file = TEXT_DIR / f"{file_id}.txt"

    if not text_file.is_file():
        missing_files += 1
        continue

    try:
        text = text_file.read_text(
            encoding="utf-8",
            errors="replace"
        ).strip()
    except OSError:
        unreadable_files += 1
        continue

    if not text:
        empty_files += 1
        continue

    records.append({
        "file_id": file_id,
        "text": text,
        "label": str(row["label"])
    })

# Create the cleaned dataset
clean_df = pd.DataFrame(
    records,
    columns=["file_id", "text", "label"]
)

# Save the processed dataset
clean_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

# Display results
print("\n========== DATASET PREPARATION REPORT ==========")
print(f"Original metadata records: {len(df)}")
print(f"Records with missing text files: {missing_files}")
print(f"Empty text files skipped: {empty_files}")
print(f"Unreadable text files skipped: {unreadable_files}")
print(f"Valid text records saved: {len(clean_df)}")

if not clean_df.empty:
    print("\nLabel distribution:")
    print(clean_df["label"].value_counts())

    print("\nFirst 5 cleaned records:")
    print(clean_df.head().to_string(index=False))

print(f"\nOutput file: {OUTPUT_FILE}")
print(f"Output file exists: {OUTPUT_FILE.exists()}")
print("\nDataset preparation completed.")
