
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# -------------------------------------
# 1. PROJECT CONFIGURATION
# -------------------------------------

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "training" / "data_splits.csv"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 10
SEED = 42

tf.keras.utils.set_random_seed(SEED)

# -------------------------------------
# 2. LOAD THE DATASET SPLITS
# -------------------------------------

df = pd.read_csv(CSV_PATH)

train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()

LABEL_MAP = {
    "NORMAL": 0,
    "PNEUMONIA": 1
}

train_df["target"] = train_df["label"].map(LABEL_MAP)
val_df["target"] = val_df["label"].map(LABEL_MAP)

print("Training images:", len(train_df))
print("Validation images:", len(val_df))

# -------------------------------------
# 3. IMAGE LOADING FUNCTION
# -------------------------------------

def load_image(filepath, label):
    image_bytes = tf.io.read_file(filepath)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(image, IMAGE_SIZE)
    image = preprocess_input(image)

    return image, tf.cast(label, tf.float32)


def create_dataset(dataframe, shuffle=False):
    paths = [
        str(ROOT / filepath)
        for filepath in dataframe["filepath"]
    ]

    labels = dataframe["target"].to_numpy(
        dtype=np.float32
    )

    dataset = tf.data.Dataset.from_tensor_slices(
        (paths, labels)
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=SEED,
            reshuffle_each_iteration=True
        )

    dataset = dataset.map(
        load_image,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


train_ds = create_dataset(train_df, shuffle=True)
val_ds = create_dataset(val_df)

# -------------------------------------
# 4. BUILD THE CNN MODEL
# -------------------------------------

base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))

x = base_model(inputs, training=False)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid"
)(x)

model = Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc")
    ]
)

model.summary()

# -------------------------------------
# 5. HANDLE CLASS IMBALANCE
# -------------------------------------

class_counts = train_df["target"].value_counts()

total = len(train_df)
num_classes = 2

class_weights = {
    int(cls): total / (num_classes * count)
    for cls, count in class_counts.items()
}

print("Class weights:", class_weights)

# -------------------------------------
# 6. TRAIN THE MODEL
# -------------------------------------

checkpoint_path = MODEL_DIR / "chest_xray_model.keras"

callbacks = [
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        filepath=str(checkpoint_path),
        monitor="val_loss",
        save_best_only=True
    )
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks
)

# -------------------------------------
# 7. SAVE TRAINING HISTORY
# -------------------------------------

history_df = pd.DataFrame(history.history)

history_df.to_csv(
    MODEL_DIR / "training_history.csv",
    index=False
)

print("\nTRAINING COMPLETE")
print("Saved model:", checkpoint_path)
print("Saved history:", MODEL_DIR / "training_history.csv")
