"""Collect BISINDO hand gesture data using MediaPipe."""
from __future__ import annotations

import argparse
import csv
import logging
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional

import cv2
import mediapipe as mp
import numpy as np

from bisindo.utils import make_feature_vector, project_root

SEED = 42
np.random.seed(SEED)


def setup_logging(log_dir: Path) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "data_collect.log"

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


class GracefulKiller:
    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *_):
        logging.info("Received termination signal. Releasing camera.")
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        sys.exit(0)


def write_header_if_needed(csv_path: Path, feature_dim: int) -> None:
    if csv_path.exists() and csv_path.stat().st_size > 0:
        return
    header = ["label"] + [f"f{i}" for i in range(feature_dim)] + ["ts"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)


def append_sample(csv_path: Path, row: List[float]) -> None:
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect BISINDO hand gesture data")
    parser.add_argument("--label", required=True, help="Label for captured gesture, e.g. 'A'")
    parser.add_argument("--samples", type=int, default=500, help="Number of frames to save")
    parser.add_argument(
        "--out",
        type=Path,
        default=project_root() / "data" / "raw",
        help="Output CSV file or directory",
    )
    parser.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--mirror", action="store_true", help="Mirror the webcam feed")
    return parser.parse_args()


def get_output_path(out: Path, label: str) -> Path:
    if out.is_dir() or not out.suffix:
        out.mkdir(parents=True, exist_ok=True)
        return out / f"{label}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def main() -> None:
    args = parse_args()
    project_dir = project_root()
    runs_dir = project_dir / "runs"
    setup_logging(runs_dir)

    csv_path = get_output_path(args.out, args.label)
    logging.info("Saving samples for label '%s' to %s", args.label, csv_path)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        logging.error("Failed to open camera index %s", args.camera_index)
        sys.exit(1)

    killer = GracefulKiller(cap)

    mp_hands = mp.solutions.hands
    saved = 0
    required = args.samples
    feature_dim: Optional[int] = None

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        logging.info("Press 's' to save frame, 'q' to quit.")
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

            overlay_text = f"Label: {args.label} | Saved: {saved}/{required}"
            cv2.putText(
                frame,
                overlay_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if hand_count == 1:
                hand_landmarks = multi_hand_landmarks[0]
                lm = np.array(
                    [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                    dtype=np.float32,
                )
                features = make_feature_vector(lm)
                feature_dim = feature_dim or len(features)
                assert len(features) == feature_dim, "Feature dimension mismatch detected."

                for landmark in hand_landmarks.landmark:
                    cx, cy = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
            else:
                features = None

            cv2.imshow("BISINDO Data Collection", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                logging.info("Exiting on user request.")
                break
            if key == ord("s") and hand_count == 1 and features is not None:
                if saved >= required:
                    logging.info("Target number of samples reached. Press 'q' to exit.")
                    continue
                if feature_dim is None:
                    feature_dim = len(features)
                write_header_if_needed(csv_path, feature_dim)
                row: List[float] = [args.label] + features.tolist() + [time.time()]
                append_sample(csv_path, row)
                saved += 1
                if saved % 10 == 0 or saved == required:
                    logging.info("Saved %s/%s samples", saved, required)
            elif key == ord("s") and hand_count != 1:
                logging.info("Frame skipped: detected %s hands", hand_count)

    cap.release()
    cv2.destroyAllWindows()
    logging.info("Finished data collection. Saved %s samples.", saved)


if __name__ == "__main__":
    main()
