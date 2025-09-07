"""Heads‑up display rendering.

The functions in this module draw diagnostic information and alert banners
directly onto video frames using OpenCV.  They are pure functions that
operate on ``numpy`` arrays in place.
"""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import cv2


def draw_metrics(
    frame: 'cv2.typing.MatLike',
    ear: float,
    perclos: float,
    mar: float,
    yaw: Optional[float],
    pitch: Optional[float],
    colour: Tuple[int, int, int] = (255, 255, 255),
    position: Tuple[int, int] = (10, 20),
) -> None:
    """Draw metric values onto the frame.

    Parameters
    ----------
    frame:
        The BGR image to draw on.
    ear, perclos, mar:
        Aspect ratio values to display.
    yaw, pitch:
        Head pose angles in degrees.  ``None`` values are shown as ``--``.
    colour:
        Text colour in BGR.
    position:
        Top‑left corner for the text.
    """
    x, y = position
    dy = 20  # line spacing
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 1
    cv2.putText(frame, f"EAR: {ear:.2f}", (x, y), font, scale, colour, thickness, cv2.LINE_AA)
    cv2.putText(frame, f"PERCLOS: {perclos:.2f}", (x, y + dy), font, scale, colour, thickness, cv2.LINE_AA)
    cv2.putText(frame, f"MAR: {mar:.2f}", (x, y + 2 * dy), font, scale, colour, thickness, cv2.LINE_AA)
    yaw_str = f"{yaw:.1f}" if yaw is not None else "--"
    pitch_str = f"{pitch:.1f}" if pitch is not None else "--"
    cv2.putText(frame, f"Yaw: {yaw_str}°", (x, y + 3 * dy), font, scale, colour, thickness, cv2.LINE_AA)
    cv2.putText(frame, f"Pitch: {pitch_str}°", (x, y + 4 * dy), font, scale, colour, thickness, cv2.LINE_AA)


def draw_status_banner(
    frame: 'cv2.typing.MatLike',
    status: Optional[str],
    colour: Tuple[int, int, int],
    height: int = 30,
) -> None:
    """Draw a coloured banner across the top of the frame with the status text.

    Parameters
    ----------
    frame:
        The BGR image to draw on.
    status:
        Status string to display.  If ``None`` or empty, no banner is drawn.
    colour:
        Banner colour in BGR.
    height:
        Banner height in pixels.
    """
    if not status:
        return
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, height), colour, -1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thickness = 2
    text_size, _ = cv2.getTextSize(status, font, scale, thickness)
    text_x = (w - text_size[0]) // 2
    text_y = height // 2 + text_size[1] // 2 - 2
    cv2.putText(frame, status, (text_x, text_y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


def draw_phone_boxes(
    frame: 'cv2.typing.MatLike',
    boxes: Iterable[Tuple[int, int, int, int]],
    colour: Tuple[int, int, int] = (0, 0, 255),
) -> None:
    """Draw phone bounding boxes and labels.

    Parameters
    ----------
    frame:
        The BGR image to draw on.
    boxes:
        Iterable of bounding boxes in (x1, y1, x2, y2) format.
    colour:
        Box colour in BGR.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.5
    thickness = 2
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        # Draw label above the box
        label = "PHONE"
        label_size, _ = cv2.getTextSize(label, font, scale, thickness)
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 4), (x1 + label_size[0] + 4, y1), colour, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)