
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
OUTPUT = ROOT / "training" / "data_splits.csv"

records = []

# Read original training images only
for label in ["NORMAL", "PNEUMONIA"]:
    folder = DATASET / "train" / label

    for image in sorted(folder.iterdir()):
        if image.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            records.append({
                "filepath": str(image.relative_to(ROOT)),
                "label": label
            })

df = pd.DataFrame(records)

# Split training data:
# 80% training and 20% validation
train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

train_df = train_df.copy()
val_df = val_df.copy()

train_df["split"] = "train"
val_df["split"] = "validation"

# Combine and save
combined = pd.concat(
    [train_df, val_df],
    ignore_index=True
)

combined.to_csv(OUTPUT, index=False)

print("\nDATASET PREPARATION COMPLETE")
print("\nTraining images:", len(train_df))
print("Validation images:", len(val_df))

print("\nClass distribution:")
print(pd.crosstab(combined["split"], combined["label"]))

print("\nSaved:", OUTPUT)
