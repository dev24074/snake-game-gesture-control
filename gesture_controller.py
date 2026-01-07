"""
Gesture Recognition Module (Calibrated OpenCV-only)
Python 3.12 compatible – NO MediaPipe
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from collections import deque
import time


class GestureController:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=400, varThreshold=20, detectShadows=False
        )

        self.positions = deque(maxlen=6)

        self.current_direction = None
        self.lock_frames = 0
        self.max_lock = 10

        # Adaptive (will be calibrated)
        self.gesture_threshold = 45
        self.area_threshold = 3000
        self.boost_area_jump = 2000

        self.prev_area = None
        self.boost_counter = 0

        # Calibration
        self.calibrating = False
        self.calibration_data = []
        self.calibration_start = None

    def _skin_mask(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower = np.array([0, 20, 70], dtype=np.uint8)
        upper = np.array([20, 255, 255], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower, upper)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        mask = cv2.dilate(mask, kernel, iterations=2)

        return mask

    def start_calibration(self):
        self.calibrating = True
        self.calibration_data.clear()
        self.calibration_start = time.time()

    def _finish_calibration(self):
        areas = np.array(self.calibration_data)

        avg_area = np.mean(areas)
        noise = np.std(areas)

        self.area_threshold = max(2000, int(avg_area * 0.6))
        self.boost_area_jump = int(noise * 6 + 1200)
        self.gesture_threshold = int(35 + noise * 2)

        self.calibrating = False

    def detect_gestures(
        self, frame: np.ndarray
    ) -> Tuple[Optional[str], bool, np.ndarray]:

        annotated = frame.copy()

        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            self.start_calibration()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        motion = self.bg_subtractor.apply(blur)
        _, motion = cv2.threshold(motion, 200, 255, cv2.THRESH_BINARY)

        skin = self._skin_mask(frame)
        combined = cv2.bitwise_and(motion, skin)

        contours, _ = cv2.findContours(
            combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        direction = self.current_direction
        is_boost = False

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > self.area_threshold:
                x, y, w, h = cv2.boundingRect(largest)
                cx, cy = x + w // 2, y + h // 2
                self.positions.append(np.array([cx, cy]))

                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(annotated, (cx, cy), 6, (0, 0, 255), -1)

                if self.calibrating:
                    self.calibration_data.append(area)
                    cv2.putText(
                        annotated,
                        "CALIBRATING... HOLD HAND STILL",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (0, 255, 255),
                        2,
                    )

                    if time.time() - self.calibration_start > 2.0:
                        self._finish_calibration()

                elif len(self.positions) >= 2 and self.lock_frames == 0:
                    delta = self.positions[-1] - self.positions[0]

                    if np.linalg.norm(delta) > self.gesture_threshold:
                        if abs(delta[0]) > abs(delta[1]):
                            direction = "RIGHT" if delta[0] > 0 else "LEFT"
                        else:
                            direction = "DOWN" if delta[1] > 0 else "UP"

                        if direction != self.current_direction:
                            self.current_direction = direction
                            self.lock_frames = self.max_lock

                if self.prev_area is not None:
                    if area - self.prev_area > self.boost_area_jump:
                        self.boost_counter += 1
                    else:
                        self.boost_counter = max(0, self.boost_counter - 1)

                if self.boost_counter >= 3:
                    is_boost = True
                    cv2.putText(
                        annotated,
                        "SPEED BOOST!",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 255),
                        2,
                    )

                self.prev_area = area

        if self.lock_frames > 0:
            self.lock_frames -= 1

        if self.current_direction:
            cv2.putText(
                annotated,
                f"Direction: {self.current_direction}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )

        return self.current_direction, is_boost, annotated

    def reset_gesture_state(self):
        self.positions.clear()
        self.current_direction = None
        self.lock_frames = 0
        self.prev_area = None
        self.boost_counter = 0
