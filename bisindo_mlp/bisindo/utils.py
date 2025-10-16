"""Utility functions for BISINDO MLP pipeline."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from joblib import load

LOGGER = logging.getLogger(__name__)


def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parent.parent


def _default_models_dir() -> Path:
    return project_root() / "models"


def load_scaler(path: Optional[Path] = None):
    """Load the fitted StandardScaler."""
    scaler_path = path or _default_models_dir() / "scaler.joblib"
    LOGGER.debug("Loading scaler from %s", scaler_path)
    return load(scaler_path)


def load_label_encoder(path: Optional[Path] = None):
    """Load the fitted LabelEncoder."""
    encoder_path = path or _default_models_dir() / "label_encoder.joblib"
    LOGGER.debug("Loading label encoder from %s", encoder_path)
    return load(encoder_path)


def load_config(path: Optional[Path] = None) -> dict:
    """Load the runtime configuration for inference."""
    config_path = path or _default_models_dir() / "config.json"
    LOGGER.debug("Loading config from %s", config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_landmarks(landmarks: Sequence[Sequence[float]]) -> np.ndarray:
    """Normalize landmarks relative to the wrist and scale invariantly.

    Parameters
    ----------
    landmarks: iterable with shape (21, 3)

    Returns
    -------
    np.ndarray
        Normalized landmarks of shape (21, 3).
    """
    lm_array = np.asarray(landmarks, dtype=np.float32)
    assert lm_array.shape == (21, 3), "Expected 21 hand landmarks with 3 coordinates each."

    wrist = lm_array[0].copy()
    lm_array -= wrist

    middle_mcp = lm_array[9]
    scale = float(np.linalg.norm(middle_mcp))
    if scale < 1e-6:
        scale = 1.0
    lm_array /= scale
    return lm_array


def _angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
    dot = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return float(np.arccos(dot))


def _compute_joint_angles(norm_landmarks: np.ndarray) -> List[float]:
    fingers: Iterable[Tuple[int, int, int, int]] = (
        (5, 6, 7, 8),  # index
        (9, 10, 11, 12),  # middle
        (13, 14, 15, 16),  # ring
        (17, 18, 19, 20),  # pinky
    )
    angles: List[float] = []
    for base, p1, p2, tip in fingers:
        v1 = norm_landmarks[p1] - norm_landmarks[base]
        v2 = norm_landmarks[p2] - norm_landmarks[p1]
        v3 = norm_landmarks[tip] - norm_landmarks[p2]
        angles.append(_angle_between(v1, v2))
        angles.append(_angle_between(v2, v3))
    return angles


def make_feature_vector(landmarks: Sequence[Sequence[float]]) -> np.ndarray:
    """Create a feature vector from landmarks using normalization and angles."""
    normalized = normalize_landmarks(landmarks)
    flat = normalized.flatten()
    angles = _compute_joint_angles(normalized)
    if angles:
        features = np.concatenate([flat, np.asarray(angles, dtype=np.float32)], axis=0)
    else:
        features = flat
    return features.astype(np.float32)


def draw_label(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int] = (10, 30),
    color: Tuple[int, int, int] = (0, 255, 0),
    font_scale: float = 1.0,
    thickness: int = 2,
    box: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """Draw a label and optional bounding box on the frame."""
    if box is not None:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if text:
        cv2.putText(
            frame,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    return frame


def validate_hand_count(hand_count: int, policy: str = "reject") -> bool:
    """Validate the number of detected hands based on policy."""
    if policy == "reject":
        return hand_count == 1
    if policy == "allow":
        return hand_count >= 1
    if policy == "at_least_one":
        return hand_count >= 1
    LOGGER.warning("Unknown two-hand policy %s, defaulting to reject", policy)
    return hand_count == 1


__all__ = [
    "project_root",
    "load_scaler",
    "load_label_encoder",
    "load_config",
    "normalize_landmarks",
    "make_feature_vector",
    "draw_label",
    "validate_hand_count",
]
