"""Facial metrics for driver monitoring.

This module defines functions to compute facial aspect ratios used in
driver‑monitoring heuristics.  The functions operate on a dictionary of
landmark coordinates indexed by MediaPipe Face Mesh landmark indices.

Functions:
    * :func:`compute_ear` – Eye Aspect Ratio for one eye.
    * :func:`compute_mar` – Mouth Aspect Ratio.
    * :func:`update_perclos` – Maintain a PERCLOS rolling window.

Index constants are provided for convenience.  See the MediaPipe Face Mesh
documentation for the meaning of each landmark index.

"""
from __future__ import annotations

import numpy as np
from collections import deque
from typing import Dict, Iterable, List, MutableSequence, Tuple

# Landmark indices for the eyes and mouth (MediaPipe Face Mesh)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 13, 14]  # left, right, top inner, bottom inner


def _euclidean(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def compute_ear(landmarks: Dict[int, Tuple[float, float]], eye_indices: Iterable[int]) -> float:
    """Compute the Eye Aspect Ratio (EAR) for one eye.

    Parameters
    ----------
    landmarks:
        A mapping from landmark indices to ``(x, y)`` pixel coordinates.
    eye_indices:
        An iterable of six landmark indices in the order (p1, p2, p3, p4, p5, p6).

    Returns
    -------
    float
        The eye aspect ratio.  If any landmark is missing, returns ``0.0``.
    """
    try:
        p1, p2, p3, p4, p5, p6 = [landmarks[i] for i in eye_indices]
    except KeyError:
        return 0.0
    # EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    numerator = _euclidean(p2, p6) + _euclidean(p3, p5)
    denominator = 2.0 * _euclidean(p1, p4)
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def compute_mar(landmarks: Dict[int, Tuple[float, float]], mouth_indices: Iterable[int] = MOUTH) -> float:
    """Compute the Mouth Aspect Ratio (MAR).

    Parameters
    ----------
    landmarks:
        A mapping from landmark indices to ``(x, y)`` pixel coordinates.
    mouth_indices:
        Indices specifying (left corner, right corner, top inner, bottom inner).

    Returns
    -------
    float
        The mouth aspect ratio.  If any landmark is missing, returns ``0.0``.
    """
    try:
        left, right, top, bottom = [landmarks[i] for i in mouth_indices]
    except KeyError:
        return 0.0
    horiz = _euclidean(left, right)
    vert = _euclidean(top, bottom)
    if horiz <= 0:
        return 0.0
    return vert / horiz


def update_perclos(window: MutableSequence[int], eye_closed: bool, maxlen: int) -> float:
    """Update a PERCLOS window with the current eye state and return the ratio.

    A simple sliding window of fixed length stores ``1`` when the eye is
    considered closed (``True``) and ``0`` when open (``False``).  PERCLOS is
    defined as the mean of the values in the window.

    Parameters
    ----------
    window:
        Mutable sequence representing the current window.  Will be modified
        in place.  Should behave like a deque or list.
    eye_closed:
        Boolean indicating whether the eye is closed in the current frame.
    maxlen:
        Maximum window length.  If the window exceeds this length, the oldest
        entry is removed.

    Returns
    -------
    float
        The PERCLOS value (fraction of frames with closed eyes).
    """
    window.append(1 if eye_closed else 0)
    # Enforce maximum length
    while len(window) > maxlen:
        window.pop(0)
    if not window:
        return 0.0
    return sum(window) / len(window)