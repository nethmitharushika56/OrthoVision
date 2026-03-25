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
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.get_logger().setLevel("ERROR")

SEED = 1337
np.random.seed(SEED)
tf.random.set_seed(SEED)

IMG_SIZE = (384, 384)
BATCH_SIZE = 16
EPOCHS_HEAD = 24
EPOCHS_FINE_TUNE = 18

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


def load_dataset() -> tuple[list[str], list[int], list[str], list[int], dict]:
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"Missing dataset folder: {DATASET_ROOT}")

    classes: list[str] = []
    for class_dir in sorted([p for p in DATASET_ROOT.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        train_dir = class_dir / "Train"
        test_dir = class_dir / "Test"
        count = len(_list_image_files(train_dir) if train_dir.exists() else []) + len(
            _list_image_files(test_dir) if test_dir.exists() else []
        )
        if count > 0:
            classes.append(class_dir.name)
        else:
            print(f"[DATA] Skipping empty class folder: {class_dir.name}")

    if not classes:
        raise ValueError("No non-empty fracture type folders found under dataset/")

    class_indices = {name: idx for idx, name in enumerate(classes)}

    train_paths: list[str] = []
    train_labels: list[int] = []
    test_paths: list[str] = []
    test_labels: list[int] = []

    rng = np.random.default_rng(SEED)

    for class_name in classes:
        class_idx = class_indices[class_name]

        train_dir = DATASET_ROOT / class_name / "Train"
        class_train_files = _list_image_files(train_dir) if train_dir.exists() else []
        for img_path in class_train_files:
            train_paths.append(str(img_path))
            train_labels.append(class_idx)

        test_dir = DATASET_ROOT / class_name / "Test"
        class_test_files = _list_image_files(test_dir) if test_dir.exists() else []

        # Some classes may ship with empty Train folders. Promote most Test
        # samples to Train so the class can still be learned, while keeping
        # a small holdout subset for final evaluation.
        if not class_train_files and class_test_files:
            shuffled = list(rng.permutation(class_test_files))
            if len(shuffled) == 1:
                promoted = shuffled
                retained_test = []
            else:
                n_test_keep = max(1, int(round(len(shuffled) * 0.2)))
                n_test_keep = min(n_test_keep, len(shuffled) - 1)
                retained_test = shuffled[:n_test_keep]
                promoted = shuffled[n_test_keep:]

            class_train_files = promoted
            class_test_files = retained_test
            print(
                f"[DATA] Promoted {len(promoted)} samples from Test->Train for '{class_name}' "
                f"(kept {len(retained_test)} for Test)."
            )

        for img_path in class_test_files:
            test_paths.append(str(img_path))
            test_labels.append(class_idx)

    if not train_paths:
        raise ValueError("No training images found in dataset/<type>/Train")
    if not test_paths:
        raise ValueError("No test images found in dataset/<type>/Test")

    return train_paths, train_labels, test_paths, test_labels, class_indices


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
    if not training:
        ds = ds.cache()
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
            layers.BatchNormalization(),
            layers.Dropout(0.35),
            layers.Dense(512, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.35),
            layers.Dense(256, activation="relu"),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    return model, base


def compile_model(model: tf.keras.Model, lr: float):
    def sparse_cce_with_label_smoothing(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
        y_true_one_hot = tf.one_hot(y_true, depth=tf.shape(y_pred)[-1])
        return tf.keras.losses.categorical_crossentropy(
            y_true_one_hot,
            y_pred,
            label_smoothing=0.03,
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=sparse_cce_with_label_smoothing,
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name="top_3_accuracy"),
        ],
    )


def compute_class_weights_from_labels(labels: list[int], num_classes: int) -> dict[int, float]:
    y = np.asarray(labels)
    present_classes = np.unique(y)
    balanced = compute_class_weight(class_weight="balanced", classes=present_classes, y=y)

    # Default unseen classes to 1.0 so training can proceed without crashing.
    weights = {int(i): 1.0 for i in range(num_classes)}
    for class_id, w in zip(present_classes.tolist(), balanced.tolist()):
        weights[int(class_id)] = float(w)
    return weights


def main():
    print("=" * 80)
    print("TRAINING: Fracture Type Classifier (dataset/)")
    print("=" * 80)

    train_all_p, train_all_y, test_p, test_y, class_indices = load_dataset()
    num_classes = len(class_indices)

    # Write indices immediately so backend can map predictions even while training.
    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    OUT_INDICES.write_text(json.dumps(class_indices, indent=2), encoding="utf-8")

    print(f"Classes: {num_classes}")
    for k, v in class_indices.items():
        print(f"  {v}: {k}")
    print(f"Training images: {len(train_all_p)}")
    print(f"Test images:     {len(test_p)}")
    print("Training class distribution:")
    for class_name, idx in class_indices.items():
        print(f"  {class_name:25s}: {Counter(train_all_y)[idx]}")

    # Keep dataset/Test fully isolated and carve validation from Train only.
    val_ratio = 0.15
    rng = np.random.default_rng(SEED)
    train_idx: list[int] = []
    val_idx: list[int] = []
    train_labels_arr = np.asarray(train_all_y)

    for class_id in range(num_classes):
        idxs = np.where(train_labels_arr == class_id)[0]
        if idxs.size == 0:
            continue
        shuffled = rng.permutation(idxs)
        n_val = max(1, int(round(idxs.size * val_ratio))) if idxs.size > 1 else 0
        val_idx.extend(shuffled[:n_val].tolist())
        train_idx.extend(shuffled[n_val:].tolist())

    if not train_idx or not val_idx:
        raise ValueError("Could not create train/val split from Train set")

    train_p = [train_all_p[i] for i in train_idx]
    train_y = [train_all_y[i] for i in train_idx]
    val_p = [train_all_p[i] for i in val_idx]
    val_y = [train_all_y[i] for i in val_idx]

    print("\nSplit:")
    print(f"  Train: {len(train_p)}")
    print(f"  Val:   {len(val_p)}")
    print(f"  Test:  {len(test_p)}")

    class_weights = compute_class_weights_from_labels(train_y, num_classes)
    print("Class weights:")
    print("  " + ", ".join([f"{k}:{v:.3f}" for k, v in class_weights.items()]))

    train_ds = make_tf_dataset(train_p, train_y, training=True)
    val_ds = make_tf_dataset(val_p, val_y, training=False)
    test_ds = make_tf_dataset(test_p, test_y, training=False)

    model, base = build_model(num_classes)
    print(f"\nTotal params: {model.count_params():,}")

    compile_model(model, lr=8e-4)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            str(OUT_MODEL), monitor="val_accuracy", save_best_only=True, verbose=0
        ),
    ]

    print("\nPhase 1: frozen base")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_HEAD,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    # Fine-tune top 35% of the base.
    print("\nPhase 2: fine-tune")
    n = len(base.layers)
    for layer in base.layers[int(n * 0.65) :]:
        layer.trainable = True

    compile_model(model, lr=8e-5)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE_TUNE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )

    test_loss, test_acc, test_top3 = model.evaluate(test_ds, verbose=0)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test top-3 accuracy: {test_top3:.4f}")

    # Detailed diagnostics to guide data improvements.
    y_true_batches: list[np.ndarray] = []
    pred_batches: list[np.ndarray] = []
    for xb, yb in test_ds:
        pb = model.predict(xb, verbose=0)
        pred_batches.append(pb)
        y_true_batches.append(yb.numpy())

    y_true = np.concatenate(y_true_batches, axis=0).astype(np.int32)
    pred_probs = np.concatenate(pred_batches, axis=0)
    y_pred = np.argmax(pred_probs, axis=1)
    idx_to_class = {v: k for k, v in class_indices.items()}
    target_names = [idx_to_class[i] for i in range(num_classes)]

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))

    print(f"Saved model: {OUT_MODEL}")
    print(f"Saved class indices: {OUT_INDICES}")


if __name__ == "__main__":
    main()
