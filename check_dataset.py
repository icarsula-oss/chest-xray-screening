
from pathlib import Path

dataset = Path(__file__).resolve().parent / "dataset"

print("Dataset found:", dataset.exists())

for folder in ["train", "test", "val"]:
    path = dataset / folder
    print(f"{folder} folder found:", path.exists()
)