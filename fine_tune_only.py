import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import classification_report, confusion_matrix

# Import configuration and helper functions from train_fracture_type
import train_fracture_type

def main():
    print("=" * 80)
    print("FINE-TUNING ONLY: Running Phase 2 from the saved Phase 1 checkpoint")
    print("=" * 80)
    
    # Load dataset splits and weights
    train_all_p, train_all_y, test_p, test_y, class_indices = train_fracture_type.load_dataset()
    num_classes = len(class_indices)
    
    # Split
    val_ratio = 0.15
    rng = np.random.default_rng(train_fracture_type.SEED)
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

    train_p = [train_all_p[i] for i in train_idx]
    train_y = [train_all_y[i] for i in train_idx]
    val_p = [train_all_p[i] for i in val_idx]
    val_y = [train_all_y[i] for i in val_idx]

    print(f"Dataset split - Train: {len(train_p)}, Val: {len(val_p)}, Test: {len(test_y)}")
    
    class_weights = train_fracture_type.compute_class_weights_from_labels(train_y, num_classes)
    
    train_ds = train_fracture_type.make_tf_dataset(train_p, train_y, training=True, num_classes=num_classes)
    val_ds = train_fracture_type.make_tf_dataset(val_p, val_y, training=False, num_classes=num_classes)
    test_ds = train_fracture_type.make_tf_dataset(test_p, test_y, training=False, num_classes=num_classes)
    
    # Load saved model from Phase 1
    model_path = train_fracture_type.OUT_MODEL
    print(f"Loading Phase 1 model from {model_path}...")
    model = tf.keras.models.load_model(str(model_path), compile=False)
    
    # Configure Phase 2 fine-tuning layers
    print("Unfreezing top 35% layers of the base model...")
    base = model.layers[0]
    base.trainable = True
    n = len(base.layers)
    fine_tune_at = int(n * 0.65)
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base.layers[fine_tune_at:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            
    # Compile
    train_fracture_type.compile_model(model, lr=8e-5)
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.ModelCheckpoint(
            str(model_path), monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
        ),
    ]
    
    print("\nPhase 2: fine-tune (8 epochs)")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=8,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1,
    )
    
    print("\nEvaluating on Test dataset...")
    test_loss, test_acc, test_top3 = model.evaluate(test_ds, verbose=0)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test top-3 accuracy: {test_top3:.4f}")
    
    # Detailed diagnostics
    y_true_batches = []
    pred_batches = []
    for xb, yb in test_ds:
        pb = model.predict(xb, verbose=0)
        pred_batches.append(pb)
        y_true_batches.append(np.argmax(yb.numpy(), axis=1))

    y_true = np.concatenate(y_true_batches, axis=0).astype(np.int32)
    pred_probs = np.concatenate(pred_batches, axis=0)
    y_pred = np.argmax(pred_probs, axis=1)
    
    idx_to_class = {v: k for k, v in class_indices.items()}
    target_names = [idx_to_class[i] for i in range(num_classes)]

    print("\nClassification report:")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print(f"Saved fine-tuned model: {model_path}")

if __name__ == "__main__":
    main()
