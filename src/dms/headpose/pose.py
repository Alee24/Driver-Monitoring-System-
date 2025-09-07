"""Head pose estimation using OpenCV's solvePnP.

This module provides a single function, :func:`estimate_head_pose`, that
estimates yaw, pitch and roll angles from facial landmarks.  The function uses
a simple 3D face model (defined in millimetres) and maps selected 2D
landmarks to the model via the Perspective‑n‑Point (PnP) algorithm.  It
returns Euler angles in degrees.
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# Landmark indices to use for head pose estimation (MediaPipe Face Mesh)
POSE_LANDMARKS = {
    "nose": 1,
    "chin": 152,
    "eye_left_outer": 33,
    "eye_right_outer": 263,
    "mouth_left": 61,
    "mouth_right": 291,
}

# Approximate 3D model points (in millimetres).  These are relative
# coordinates and do not correspond to physical units; scaling is arbitrary.
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),        # nose tip
    (0.0, -63.6, -12.5),    # chin
    (-43.3, 32.7, -26.0),   # left eye outer
    (43.3, 32.7, -26.0),    # right eye outer
    (-28.9, -28.9, -24.1),  # mouth left
    (28.9, -28.9, -24.1),   # mouth right
], dtype=np.float64)


def estimate_head_pose(
    landmarks: Dict[int, Tuple[float, float]],
    frame_width: int,
    frame_height: int,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Estimate yaw, pitch and roll angles from face landmarks.

    Parameters
    ----------
    landmarks:
        Mapping from landmark index to ``(x, y)`` pixel coordinates.
    frame_width:
        Width of the current frame in pixels.  Used to build the camera matrix.
    frame_height:
        Height of the current frame in pixels.

    Returns
    -------
    tuple of (yaw, pitch, roll)
        Euler angles in degrees.  If estimation fails, all values are ``None``.
    """
    try:
        image_points = np.array([
            landmarks[POSE_LANDMARKS["nose"]],
            landmarks[POSE_LANDMARKS["chin"]],
            landmarks[POSE_LANDMARKS["eye_left_outer"]],
            landmarks[POSE_LANDMARKS["eye_right_outer"]],
            landmarks[POSE_LANDMARKS["mouth_left"]],
            landmarks[POSE_LANDMARKS["mouth_right"]],
        ], dtype=np.float64)
    except KeyError:
        return None, None, None

    # Camera matrix: assume fx = fy = frame_width, and principal point at centre
    focal_length = frame_width
    centre = (frame_width / 2.0, frame_height / 2.0)
    camera_matrix = np.array([
        [focal_length, 0, centre[0]],
        [0, focal_length, centre[1]],
        [0, 0, 1],
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))  # assume no lens distortion
    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None, None, None
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    # Compute Euler angles
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        pitch = math.degrees(math.atan2(-rotation_matrix[2, 0], sy))
        yaw = math.degrees(math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))
        roll = math.degrees(math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2]))
    else:
        # Gimbal lock: pitch = 0
        pitch = 0.0
        yaw = math.degrees(math.atan2(-rotation_matrix[0, 2], rotation_matrix[1, 1]))
        roll = 0.0
    return yaw, pitch, roll