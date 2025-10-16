"""Train the BISINDO MLP model."""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from bisindo.utils import load_label_encoder, load_scaler, project_root

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "train_mlp.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(ch)
    logger.addHandler(fh)


def load_datasets(processed_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_X = np.load(processed_dir / "train_X.npy")
    train_y = np.load(processed_dir / "train_y.npy")
    val_X = np.load(processed_dir / "val_X.npy")
    val_y = np.load(processed_dir / "val_y.npy")
    return train_X, train_y, val_X, val_y


def build_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(input_dim,)),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="categorical_accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top_k_categorical_accuracy"),
        ],
    )
    return model


def plot_history(history: tf.keras.callbacks.History, output_path: Path) -> None:
    history_dict = history.history
    epochs = range(1, len(history_dict["loss"]) + 1)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_dict["loss"], label="Train Loss")
    plt.plot(epochs, history_dict["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_dict["categorical_accuracy"], label="Train Acc")
    plt.plot(epochs, history_dict["val_categorical_accuracy"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def save_confusion_matrix(cm: np.ndarray, classes: list[str], output_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Validation Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    project_dir = project_root()
    processed_dir = project_dir / "data" / "processed"
    models_dir = project_dir / "models"
    runs_dir = project_dir / "runs"

    setup_logging(runs_dir)

    logging.info("TensorFlow version: %s", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        logging.info("GPUs available: %s", gpus)
    else:
        logging.info("No GPU detected; training on CPU")

    scaler = load_scaler(models_dir / "scaler.joblib")
    label_encoder = load_label_encoder(models_dir / "label_encoder.joblib")

    train_X, train_y, val_X, val_y = load_datasets(processed_dir)

    assert train_X.shape[1] == scaler.mean_.shape[0], "Feature size mismatch with scaler"
    assert val_X.shape[1] == scaler.mean_.shape[0], "Validation feature size mismatch"

    num_classes = len(label_encoder.classes_)
    logging.info("Number of classes: %d", num_classes)

    y_train_cat = tf.keras.utils.to_categorical(train_y, num_classes=num_classes)
    y_val_cat = tf.keras.utils.to_categorical(val_y, num_classes=num_classes)

    model = build_model(train_X.shape[1], num_classes)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=5,
            factor=0.5,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_X,
        y_train_cat,
        validation_data=(val_X, y_val_cat),
        epochs=200,
        batch_size=64,
        callbacks=callbacks,
        verbose=1,
    )

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "mlp.h5"
    model.save(model_path)
    logging.info("Saved model to %s", model_path)

    config = {"min_confidence": 0.75, "two_hands_policy": "reject"}
    with open(models_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    history_plot = runs_dir / "training_history.png"
    plot_history(history, history_plot)
    logging.info("Saved training history plot to %s", history_plot)

    val_pred_probs = model.predict(val_X)
    val_preds = np.argmax(val_pred_probs, axis=1)

    report = classification_report(
        val_y,
        val_preds,
        target_names=label_encoder.inverse_transform(np.arange(num_classes)),
        zero_division=0,
    )
    logging.info("Validation classification report:\n%s", report)

    cm = confusion_matrix(val_y, val_preds)
    cm_path = runs_dir / "val_confusion_matrix.png"
    save_confusion_matrix(cm, list(label_encoder.classes_), cm_path)
    logging.info("Saved validation confusion matrix to %s", cm_path)


if __name__ == "__main__":
    main()
