import builtins
import math
from collections import deque

import pytest

from dms.face.metrics import (compute_ear, compute_mar, update_perclos, LEFT_EYE, RIGHT_EYE, MOUTH)


def test_compute_ear_simple():
    # Define simple eye landmarks forming a rectangle: width=4, height=2
    # Indices correspond to p1..p6 for the formula
    landmarks = {
        0: (0.0, 0.0),  # p1 (will not be used directly)
    }
    # Map indices in LEFT_EYE to coordinates
    # p1=(0,0), p2=(0,2), p3=(0,2), p4=(4,0), p5=(0,2), p6=(0,2)
    coords = [
        (0.0, 0.0),
        (0.0, 2.0),
        (0.0, 2.0),
        (4.0, 0.0),
        (0.0, 2.0),
        (0.0, 2.0),
    ]
    for idx, coord in zip(LEFT_EYE, coords):
        landmarks[idx] = coord
    ear = compute_ear(landmarks, LEFT_EYE)
    # EAR = (2+2) / (2*4) = 4/8 = 0.5
    assert math.isclose(ear, 0.5, rel_tol=1e-2)


def test_compute_mar_simple():
    # Define a mouth with width=4 and height=2 -> MAR=2/4=0.5
    landmarks = {
        MOUTH[0]: (0.0, 0.0),      # left
        MOUTH[1]: (4.0, 0.0),      # right
        MOUTH[2]: (2.0, -1.0),     # top
        MOUTH[3]: (2.0, 1.0),      # bottom
    }
    mar = compute_mar(landmarks)
    assert math.isclose(mar, 0.5, rel_tol=1e-2)


def test_update_perclos_window():
    window: deque[int] = deque()
    # Add 5 entries (closed, open, closed, closed, open)
    values = [True, False, True, True, False]
    for v in values:
        ratio = update_perclos(window, v, maxlen=5)
    # There are 3 closed out of 5 -> ratio = 0.6
    assert math.isclose(ratio, 3/5, rel_tol=1e-3)