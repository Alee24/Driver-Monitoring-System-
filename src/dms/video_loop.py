"""
Main video processing loop for the driver monitoring system.

This module orchestrates the capture, processing and rendering of video
frames.  It pulls configuration values from a :class:`dms.utils.config.Config`
instance, performs facial landmark detection via MediaPipe, computes
heuristics for drowsiness and distraction, optionally detects phones via
YOLOv8n, renders heads-up diagnostics and alerts, and outputs spoken
warnings when appropriate.

The core functionality is provided by the :func:`run` function.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Optional, Dict, Tuple

import cv2
import mediapipe as mp
from loguru import logger

from dms.face.metrics import (
    LEFT_EYE,
    RIGHT_EYE,
    MOUTH,
    compute_ear,
    compute_mar,
    update_perclos,
)
from dms.headpose.pose import estimate_head_pose
from dms.detectors.phone import PhoneDetector
from dms.audio.alerts import AlertSpeaker
from dms.hud.overlay import draw_metrics, draw_phone_boxes, draw_status_banner
from dms.utils.config import Config


def run(config: Config) -> None:
    """Run the driver monitoring loop with the given configuration.

    Parameters
    ----------
    config:
        Configuration object defining thresholds, video source and behaviour.
    """
    logger.info("Starting driver monitoring system with configuration: {}", config.to_dict())

    # Initialise phone detector if enabled (gracefully degrade if model missing)
    phone_detector = PhoneDetector(config.yolo_model) if config.enable_phone else PhoneDetector()
    if config.enable_phone and not phone_detector.available():
        logger.warning("Phone detection enabled but model could not be loaded. Detection disabled.")

    # -------------------------
    # Initialise audio alerts
    # -------------------------
    # Backward-compat: prefer new 'audio_repeat_interval', fall back to 'audio_cooldown'
    repeat = getattr(config, "audio_repeat_interval", None)
    if repeat is None:
        repeat = getattr(config, "audio_cooldown", 1.0)
    try:
        repeat = float(repeat)
    except Exception:
        repeat = 1.0

    speaker = AlertSpeaker(speak_rate=config.speak_rate, repeat_interval=repeat)

    # Initialise PERCLOS window (store 1 for closed, 0 for open)
    perclos_window: deque[int] = deque(maxlen=config.perclos_window)

    # Track off-road gaze duration
    gaze_off_since: Optional[float] = None

    # Track frames without a detected face
    no_face_frames = 0
    no_face_threshold = int(max(1, config.target_fps) * 2)  # allow ~2 seconds without a face

    # Open video capture
    cap = cv2.VideoCapture(config.source)
    if not cap.isOpened():
        logger.error("Failed to open camera or video source: {}", config.source)
        return

    # Set desired resolution/FPS if possible (best effort)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    cap.set(cv2.CAP_PROP_FPS, config.target_fps)

    # Initialise MediaPipe Face Mesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Frame counter for throttling phone detection
    frame_counter = 0

    window_title = "Driver Monitoring System"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("End of stream or camera read failure.")
                break

            # Resize frame to desired resolution (if reading from file or camera does not support)
            h, w = frame.shape[:2]
            if (w != config.width) or (h != config.height):
                frame = cv2.resize(frame, (config.width, config.height))
            # Flip horizontally for a mirror view
            frame = cv2.flip(frame, 1)

            # Prepare for MediaPipe: convert to RGB
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            landmarks: Dict[int, Tuple[float, float]] = {}
            if results.multi_face_landmarks:
                # Extract landmarks for the first face
                face_landmarks = results.multi_face_landmarks[0]
                # Use the configured width/height for pixel coords
                for idx, landmark in enumerate(face_landmarks.landmark):
                    landmarks[idx] = (landmark.x * config.width, landmark.y * config.height)

            status: Optional[str] = None
            message: Optional[str] = None

            # Detection results
            phone_boxes = []
            ear_val = 0.0
            perclos_val = 0.0
            mar_val = 0.0
            yaw_val: Optional[float] = None
            pitch_val: Optional[float] = None

            if not landmarks:
                # No face detected
                no_face_frames += 1
                if no_face_frames >= no_face_threshold:
                    status = "NO FACE"  # display banner after threshold frames
                # Skip metrics update and alerts
                gaze_off_since = None

            else:
                no_face_frames = 0

                # Compute metrics
                left_ear = compute_ear(landmarks, LEFT_EYE)
                right_ear = compute_ear(landmarks, RIGHT_EYE)
                ear_val = (left_ear + right_ear) / 2.0

                eye_closed = ear_val < config.ear_thresh
                perclos_val = update_perclos(perclos_window, eye_closed, config.perclos_window)
                mar_val = compute_mar(landmarks, MOUTH)

                yaw_val, pitch_val, _ = estimate_head_pose(landmarks, config.width, config.height)

                # Check distraction (off-road)
                off_road = False
                if yaw_val is not None and pitch_val is not None:
                    if abs(yaw_val) > config.yaw_abs_deg or pitch_val > config.down_pitch_deg:
                        off_road = True

                # Update gaze timer
                now = time.monotonic()
                if off_road:
                    if gaze_off_since is None:
                        gaze_off_since = now
                else:
                    gaze_off_since = None

                # Throttle phone detection to ~2 Hz based on target FPS
                phone_alert = False
                if config.enable_phone and phone_detector.available():
                    step = max(1, int(max(config.target_fps, 1) / 2))  # every ~0.5s at target FPS
                    if frame_counter % step == 0:
                        phone_boxes = phone_detector.detect(
                            frame, phone_zone_ratio=config.phone_zone_ratio
                        )
                        phone_alert = bool(phone_boxes)
                else:
                    phone_alert = False
                    phone_boxes = []

                # Determine alert conditions
                drowsy = perclos_val > config.perclos_thresh
                yawn = mar_val > config.mar_yawn
                distracted = False
                if gaze_off_since is not None and (now - gaze_off_since) > config.gaze_secs:
                    distracted = True

                # Prioritise alerts
                if phone_alert:
                    status = "PHONE"
                    message = "Put the phone down. Eyes on the road."
                elif drowsy:
                    status = "DROWSY"
                    message = "Drowsiness detected. Please rest."
                elif yawn:
                    status = "YAWN"
                    message = "You are yawning. Take a short break."
                elif distracted:
                    status = "DISTRACTED"
                    message = "Please look at the road."
                else:
                    message = None

            # --- Speak continuously while active; clear when not ---
            if message:
                speaker.set_active_message(message)
            else:
                speaker.set_active_message(None)

            # Draw HUD
            draw_metrics(frame, ear_val, perclos_val, mar_val, yaw_val, pitch_val)
            if phone_boxes:
                draw_phone_boxes(frame, phone_boxes)

            # Banner colours
            banner_colour = (0, 0, 0)
            if status == "PHONE" or status == "DROWSY":
                banner_colour = (0, 0, 255)  # red
            elif status == "YAWN":
                banner_colour = (0, 165, 255)  # orange
            elif status == "DISTRACTED":
                banner_colour = (0, 255, 255)  # yellow
            elif status == "NO FACE":
                banner_colour = (128, 128, 128)  # grey
            draw_status_banner(frame, status, banner_colour)

            # Show window unless headless
            if not config.headless:
                cv2.imshow(window_title, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    break

            frame_counter += 1

    finally:
        cap.release()
        if not config.headless:
            cv2.destroyAllWindows()
        face_mesh.close()
        logger.info("Driver monitoring system stopped.")
