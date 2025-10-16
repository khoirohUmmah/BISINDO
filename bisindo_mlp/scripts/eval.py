"""Evaluate the BISINDO MLP model on the test set."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from bisindo.utils import load_label_encoder, load_scaler, project_root

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "eval.log"

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


def load_test_set(processed_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    X_test = np.load(processed_dir / "test_X.npy")
    y_test = np.load(processed_dir / "test_y.npy")
    return X_test, y_test


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], output_path: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Test Confusion Matrix")
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

    scaler = load_scaler(models_dir / "scaler.joblib")
    label_encoder = load_label_encoder(models_dir / "label_encoder.joblib")

    X_test, y_test = load_test_set(processed_dir)
    assert X_test.shape[1] == scaler.mean_.shape[0], "Test feature size mismatch"

    model_path = models_dir / "mlp.h5"
    if not model_path.exists():
        logging.error("Model not found at %s", model_path)
        sys.exit(1)

    model = tf.keras.models.load_model(model_path)
    logging.info("Loaded model from %s", model_path)

    probs = model.predict(X_test)
    preds = np.argmax(probs, axis=1)

    accuracy = accuracy_score(y_test, preds)
    precision = precision_score(y_test, preds, average="macro", zero_division=0)
    recall = recall_score(y_test, preds, average="macro", zero_division=0)
    f1 = f1_score(y_test, preds, average="macro", zero_division=0)

    logging.info("Test Accuracy: %.4f", accuracy)
    logging.info("Macro Precision: %.4f", precision)
    logging.info("Macro Recall: %.4f", recall)
    logging.info("Macro F1: %.4f", f1)

    class_names = list(label_encoder.classes_)
    report = classification_report(
        y_test,
        preds,
        target_names=class_names,
        zero_division=0,
    )
    logging.info("Classification report:\n%s", report)

    cm = confusion_matrix(y_test, preds)
    cm_path = runs_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_path)
    logging.info("Saved confusion matrix to %s", cm_path)

    confidences = probs[np.arange(len(preds)), preds]
    mistakes = []
    for true_idx, pred_idx, conf in zip(y_test, preds, confidences):
        if true_idx != pred_idx:
            mistakes.append(
                (
                    conf,
                    class_names[int(true_idx)],
                    class_names[int(pred_idx)],
                )
            )
    mistakes.sort(reverse=True)
    top_mistakes = mistakes[:10]
    if top_mistakes:
        logging.info("Top mispredictions (confidence, true -> pred):")
        for conf, true_label, pred_label in top_mistakes:
            logging.info("%.3f : %s -> %s", conf, true_label, pred_label)
    else:
        logging.info("No mispredictions found on test set.")


if __name__ == "__main__":
    main()
