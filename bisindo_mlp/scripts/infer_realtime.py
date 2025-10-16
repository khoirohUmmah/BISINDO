"""Real-time inference for BISINDO gestures using webcam."""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

from bisindo.utils import (
    draw_label,
    load_config,
    load_label_encoder,
    load_scaler,
    make_feature_vector,
    project_root,
    validate_hand_count,
)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


class GracefulExit:
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        signal.signal(signal.SIGINT, self._handler)
        signal.signal(signal.SIGTERM, self._handler)

    def _handler(self, *_):
        logging.info("Received termination signal. Cleaning up.")
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "infer_realtime.log"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time BISINDO inference")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index")
    parser.add_argument("--mirror", action="store_true", help="Mirror video output")
    parser.add_argument("--show-prob", action="store_true", help="Show max probability")
    parser.add_argument("--fps", action="store_true", help="Display FPS overlay")
    parser.add_argument("--tts", action="store_true", help="Enable text-to-speech")
    return parser.parse_args()


def extract_hand_features(hand_landmarks: mp.framework.formats.landmark_pb2.NormalizedLandmarkList) -> np.ndarray:
    lm = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
    features = make_feature_vector(lm)
    return features


def predict_label(
    model: tf.keras.Model,
    scaler,
    label_encoder,
    features: np.ndarray,
    min_confidence: float,
) -> Tuple[str, float]:
    scaled = scaler.transform(features.reshape(1, -1))
    probs = model.predict(scaled, verbose=0)[0]
    max_idx = int(np.argmax(probs))
    max_prob = float(probs[max_idx])
    if max_prob < min_confidence:
        return "", max_prob
    label = label_encoder.inverse_transform([max_idx])[0]
    return label, max_prob


def main() -> None:
    args = parse_args()

    project_dir = project_root()
    models_dir = project_dir / "models"
    runs_dir = project_dir / "runs"

    setup_logging(runs_dir)

    scaler = load_scaler(models_dir / "scaler.joblib")
    label_encoder = load_label_encoder(models_dir / "label_encoder.joblib")
    config = load_config(models_dir / "config.json")
    min_confidence = float(config.get("min_confidence", 0.75))
    policy = config.get("two_hands_policy", "reject")

    model_path = models_dir / "mlp.h5"
    if not model_path.exists():
        logging.error("Model not found at %s", model_path)
        sys.exit(1)
    model = tf.keras.models.load_model(model_path)
    logging.info("Loaded model from %s", model_path)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        logging.error("Failed to open camera index %s", args.camera_index)
        sys.exit(1)

    GracefulExit(cap)

    tts_engine = None
    if args.tts:
        import pyttsx3

        tts_engine = pyttsx3.init()
        logging.info("TTS enabled")

    last_label = ""
    prev_time = time.time()
    fps = 0.0

    mp_hands = mp.solutions.hands
    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.warning("Failed to read frame from camera")
                break

            if args.mirror:
                frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)
            multi_hand_landmarks = results.multi_hand_landmarks or []
            hand_count = len(multi_hand_landmarks)

            valid_hand = validate_hand_count(hand_count, policy)
            display_label = ""
            max_prob = 0.0

            box: Optional[Tuple[int, int, int, int]] = None

            if valid_hand and hand_count == 1:
                features = extract_hand_features(multi_hand_landmarks[0])
                assert features.shape[0] == scaler.mean_.shape[0], "Feature dimension mismatch"
                display_label, max_prob = predict_label(
                    model,
                    scaler,
                    label_encoder,
                    features,
                    min_confidence,
                )

                lm_points = np.array(
                    [
                        [lm.x * frame.shape[1], lm.y * frame.shape[0]]
                        for lm in multi_hand_landmarks[0].landmark
                    ],
                    dtype=np.float32,
                )
                x_min, y_min = np.min(lm_points, axis=0).astype(int)
                x_max, y_max = np.max(lm_points, axis=0).astype(int)
                box = (x_min, y_min, x_max, y_max)
            else:
                display_label = ""

            overlay_text = display_label
            if args.show_prob and display_label:
                overlay_text = f"{display_label} ({max_prob:.2f})"
            draw_label(frame, overlay_text, origin=(10, 40), box=box)

            if args.fps:
                current_time = time.time()
                fps = 0.9 * fps + 0.1 * (1.0 / max(current_time - prev_time, 1e-6))
                prev_time = current_time
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            cv2.imshow("BISINDO Real-time Inference", frame)

            if args.tts and tts_engine is not None and display_label and display_label != last_label:
                tts_engine.say(display_label)
                tts_engine.runAndWait()

            last_label = display_label

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logging.info("Exiting on user request.")
                break

    cap.release()
    cv2.destroyAllWindows()
    if tts_engine is not None:
        tts_engine.stop()
    logging.info("Inference ended.")


if __name__ == "__main__":
    main()
