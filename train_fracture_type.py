"""Train a multi-class fracture *type* classifier using dataset/.

Dataset layout expected:
  dataset/<Fracture Type>/{Train,Test}/*.jpg|*.png

Outputs:
  backend/models/orthovision_type_model.keras
  backend/models/orthovision_type_model.class_indices.json

Notes:
- This model predicts *type among fractures*. It cannot predict "normal" unless
  you provide a normal dataset.
- We use tf.data + ignore_errors() to survive corrupt images.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

SEED = 1337
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (384, 384)
BATCH_SIZE = 16
EPOCHS = 30

DATASET_ROOT = Path("dataset")
OUT_MODEL = Path("backend/models/orthovision_type_model.keras")
OUT_INDICES = Path("backend/models/orthovision_type_model.class_indices.json")

AUG = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomFlip("vertical"),
        layers.RandomRotation(0.12),
        layers.RandomZoom(0.12),
        layers.RandomContrast(0.2),
        layers.RandomBrightness(0.2),
        layers.RandomTranslation(0.08, 0.08),
    ],
    name="augment",
)


@dataclass(frozen=True)
class Example:
    path: str
    label: int


def _list_image_files(folder: Path) -> list[Path]:
    return (
        list(folder.glob("*.jpg"))
        + list(folder.glob("*.JPG"))
        + list(folder.glob("*.jpeg"))
        + list(folder.glob("*.JPEG"))
        + list(folder.glob("*.png"))
        + list(folder.glob("*.PNG"))
    )


def load_dataset() -> tuple[list[str], list[int], dict]:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Missing dataset folder: {DATASET_ROOT}")

    classes = sorted([p.name for p in DATASET_ROOT.iterdir() if p.is_dir()])
    if not classes:
        raise ValueError("No fracture type folders found under dataset/")

    class_indices = {name: idx for idx, name in enumerate(classes)}

    image_paths: list[str] = []
    labels: list[int] = []

    for class_name in classes:
        for split in ["Train", "Test"]:
            split_dir = DATASET_ROOT / class_name / split
            if not split_dir.exists():
                continue
            for img_path in _list_image_files(split_dir):
                image_paths.append(str(img_path))
                labels.append(class_indices[class_name])

    if not image_paths:
        raise ValueError("No images found in dataset/<type>/{Train,Test}")

    return image_paths, labels, class_indices


def make_tf_dataset(paths: list[str], labels: list[int], *, training: bool) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _preprocess(path, label):
        data = tf.io.read_file(path)
        # Try jpeg first, then png.
        try:
            img = tf.image.decode_jpeg(
                data,
                channels=3,
                try_recover_truncated=True,
                acceptable_fraction=0.25,
            )
        except Exception:
            img = tf.image.decode_png(data, channels=3)
        img = tf.image.resize(img, IMG_SIZE)
        img = tf.cast(img, tf.float32)
        if training:
            img = AUG(img, training=True)
        img = tf.keras.applications.efficientnet.preprocess_input(img)
        return img, tf.cast(label, tf.int32)

    # `decode_jpeg` may still fail for some files -> drop those elements.
    ds = ds.map(_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    if training:
        ds = ds.shuffle(buffer_size=min(5000, len(paths)), seed=SEED)

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    base = EfficientNetB3(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    model = models.Sequential(
        [
            base,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.4),
            layers.Dense(128, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model, base


def main():
    print("=" * 80)
    print("TRAINING: Fracture Type Classifier (dataset/)")
    print("=" * 80)

    image_paths, labels, class_indices = load_dataset()
    num_classes = len(class_indices)

    # Write indices immediately so backend can map predictions even while training.
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUT_INDICES.write_text(json.dumps(class_indices, indent=2), encoding="utf-8")

    print(f"Classes: {num_classes}")
    for k, v in class_indices.items():
        print(f"  {v}: {k}")
    print(f"Total images: {len(image_paths)}")

    # Split train/val/test (70/15/15) stratified.
    train_p, temp_p, train_y, temp_y = train_test_split(
        image_paths, labels, test_size=0.3, random_state=SEED, stratify=labels
    )
    val_p, test_p, val_y, test_y = train_test_split(
        temp_p, temp_y, test_size=0.5, random_state=SEED, stratify=temp_y
    )

    print("\nSplit:")
    print(f"  Train: {len(train_p)}")
    print(f"  Val:   {len(val_p)}")
    print(f"  Test:  {len(test_p)}")

    train_ds = make_tf_dataset(train_p, train_y, training=True)
    val_ds = make_tf_dataset(val_p, val_y, training=False)
    test_ds = make_tf_dataset(test_p, test_y, training=False)

    model, base = build_model(num_classes)
    print(f"\nTotal params: {model.count_params():,}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            str(OUT_MODEL), monitor="val_accuracy", save_best_only=True, verbose=0
        ),
    ]

    print("\nPhase 1: frozen base")
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks, verbose=1)

    # Fine-tune top half of the base.
    print("\nPhase 2: fine-tune")
    n = len(base.layers)
    for layer in base.layers[n // 2 :]:
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(train_ds, validation_data=val_ds, epochs=max(5, EPOCHS // 2), callbacks=callbacks, verbose=1)

    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")

    print(f"Saved model: {OUT_MODEL}")
    print(f"Saved class indices: {OUT_INDICES}")


if __name__ == "__main__":
    main()
