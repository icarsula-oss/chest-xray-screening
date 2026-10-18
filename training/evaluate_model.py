
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# -----------------------------------
# 1. PROJECT PATHS
# -----------------------------------

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "chest_xray_model.keras"
HISTORY_PATH = ROOT / "models" / "training_history.csv"
TEST_PATH = ROOT / "dataset" / "test"
OUTPUT = ROOT / "evaluation"

OUTPUT.mkdir(exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16

# -----------------------------------
# 2. LOAD TRAINED MODEL
# -----------------------------------

print("Loading trained CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully!")

# -----------------------------------
# 3. PLOT TRAINING HISTORY
# -----------------------------------

history = pd.read_csv(HISTORY_PATH)

for metric in ["accuracy", "loss"]:

    plt.figure(figsize=(8, 5))

    plt.plot(
        history[metric],
        label="Training"
    )

    plt.plot(
        history["val_" + metric],
        label="Validation"
    )

    plt.title(f"Training vs Validation {metric.title()}")
    plt.xlabel("Epoch")
    plt.ylabel(metric.title())
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        OUTPUT / f"{metric}_graph.png",
        dpi=200
    )

    plt.close()

# -----------------------------------
# 4. LOAD TEST IMAGES
# -----------------------------------

paths = []
labels = []

class_map = {
    "NORMAL": 0,
    "PNEUMONIA": 1
}

for class_name, label in class_map.items():

    folder = TEST_PATH / class_name

    for image_path in sorted(folder.iterdir()):

        if image_path.suffix.lower() in [
            ".jpg", ".jpeg", ".png"
        ]:
            paths.append(str(image_path))
            labels.append(label)

labels = np.array(labels)

print("Test images:", len(paths))

# -----------------------------------
# 5. PREPROCESS TEST IMAGES
# -----------------------------------

def load_image(path):
    image = tf.io.read_file(path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)
    image = preprocess_input(image)

    return image


test_ds = tf.data.Dataset.from_tensor_slices(paths)

test_ds = test_ds.map(
    load_image,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.batch(BATCH_SIZE)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# -----------------------------------
# 6. MAKE PREDICTIONS
# -----------------------------------

print("Evaluating test images...")

probabilities = model.predict(test_ds).reshape(-1)

predictions = (probabilities >= 0.5).astype(int)

# -----------------------------------
# 7. COMPUTE METRICS
# -----------------------------------

accuracy = accuracy_score(labels, predictions)

precision = precision_score(
    labels, predictions, zero_division=0
)

recall = recall_score(
    labels, predictions, zero_division=0
)

f1 = f1_score(
    labels, predictions, zero_division=0
)

cm = confusion_matrix(
    labels, predictions, labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()

specificity = (
    tn / (tn + fp)
    if (tn + fp) > 0 else float("nan")
)

print("\nMODEL EVALUATION RESULTS")
print("---------------------------")
print(f"Accuracy:    {accuracy:.4f}")
print(f"Precision:   {precision:.4f}")
print(f"Recall:      {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1 Score:    {f1:.4f}")

report = classification_report(
    labels,
    predictions,
    target_names=["NORMAL", "PNEUMONIA"],
    zero_division=0
)

print("\nCLASSIFICATION REPORT")
print(report)

# -----------------------------------
# 8. SAVE CONFUSION MATRIX
# -----------------------------------

fig, ax = plt.subplots(figsize=(6, 5))

image = ax.imshow(cm, cmap="Blues")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["NORMAL", "PNEUMONIA"])
ax.set_yticklabels(["NORMAL", "PNEUMONIA"])

ax.set_xlabel("Predicted Label")
ax.set_ylabel("Actual Label")
ax.set_title("Confusion Matrix")

for i in range(2):
    for j in range(2):
        ax.text(
            j, i, str(cm[i, j]),
            ha="center",
            va="center",
            color="black"
        )

fig.colorbar(image, ax=ax)
plt.tight_layout()

plt.savefig(
    OUTPUT / "confusion_matrix.png",
    dpi=200
)

plt.close()

# -----------------------------------
# 9. SAVE RESULTS
# -----------------------------------

results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "Specificity",
        "F1 Score"
    ],
    "Value": [
        accuracy,
        precision,
        recall,
        specificity,
        f1
    ]
})

results.to_csv(
    OUTPUT / "evaluation_metrics.csv",
    index=False
)

predictions_df = pd.DataFrame({
    "filepath": paths,
    "actual_label": labels,
    "predicted_label": predictions,
    "pneumonia_score": probabilities
})

predictions_df.to_csv(
    OUTPUT / "test_predictions.csv",
    index=False
)

with open(
    OUTPUT / "classification_report.txt", "w"
) as file:
    file.write(report)

print("\nEvaluation complete!")
print("Results saved to:", OUTPUT)
