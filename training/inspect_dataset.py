
from pathlib import Path
import pandas as pd
from PIL import Image

# Locate the project and dataset
PROJECT = Path(__file__).resolve().parents[1]
DATASET = PROJECT / "dataset"

records = []

# Count images in every dataset folder
for split in ["train", "val", "test"]:
    for label in ["NORMAL", "PNEUMONIA"]:

        folder = DATASET / split / label

        if not folder.exists():
            print("Missing folder:", folder)
            continue

        images = [
            file for file in folder.iterdir()
            if file.suffix.lower() in
            [".jpg", ".jpeg", ".png"]
        ]

        records.append({
            "Split": split,
            "Class": label,
            "Images": len(images)
        })

# Display dataset information
df = pd.DataFrame(records)

print("\nCHEST X-RAY DATASET SUMMARY")
print(df.to_string(index=False))

print("\nTotal images:", df["Images"].sum())

# Check one image from each class
print("\nSAMPLE IMAGE INFORMATION")

for label in ["NORMAL", "PNEUMONIA"]:
    folder = DATASET / "train" / label

    images = list(folder.glob("*.jpeg"))

    if images:
        with Image.open(images[0]) as image:
            print(
                label,
                "| Size:", image.size,
                "| Mode:", image.mode
            )
