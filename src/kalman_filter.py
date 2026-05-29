"""
Aerial-tuned Kalman filter for bounding-box state estimation.

State vector (8-dim):  [cx, cy, a, h, vx, vy, va, vh]
  cx, cy  -- bounding-box centre
  a       -- aspect ratio (width / height)
  h       -- bounding-box height
  vx, vy  -- translational velocity
  va, vh  -- aspect and height velocity

Measurement vector (4-dim):  [cx, cy, a, h]

Drone-specific tuning:
  - Process noise Q is scaled UP relative to the standard ByteTrack values.
    Drone platforms have erratic micro-movements that make the constant-velocity
    assumption break down quickly; a larger Q lets the filter adapt faster.
  - Measurement noise R is slightly relaxed to absorb SAHI tile-boundary jitter
    where the same person can have marginally different box coordinates depending
    on which tile it was detected in.
"""
from __future__ import annotations

import numpy as np


class KalmanFilter:
    """
    Constant-velocity Kalman filter for [cx, cy, aspect, height] bounding boxes.
    """

    _ndim = 4  # measurement dims
    _dt = 1.0  # time step (frames)

    def __init__(self, q_xy_scale: float = 1.0, q_vel_scale: float = 0.01, r_scale: float = 1.0) -> None:
        # --- Transition matrix F (8x8) ---
        self.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = self._dt

        # --- Measurement matrix H (4x8) ---
        self.H = np.eye(4, 8, dtype=np.float64)

        # --- Process noise Q (8x8) ---
        # Diagonal values tuned for aerial dynamics
        q_pos = 1.0 * q_xy_scale
        q_vel = 1.0 * q_vel_scale
        self.Q = np.diag([
            q_pos, q_pos, 1e-2, 1e-2,   # position noise: cx, cy, a, h
            q_vel, q_vel, 1e-5, 1e-5,   # velocity noise: vx, vy, va, vh
        ]).astype(np.float64)

        # --- Measurement noise R (4x4) ---
        r_diag = np.array([1.0, 1.0, 1e-1, 1e-1], dtype=np.float64) * r_scale
        self.R = np.diag(r_diag)

        # --- Initial covariance scale factors ---
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Create a new track from a raw [x1, y1, x2, y2] detection.

        Returns:
            mean (8,), covariance (8, 8)
        """
        cx = (measurement[0] + measurement[2]) / 2
        cy = (measurement[1] + measurement[3]) / 2
        h = measurement[3] - measurement[1]
        a = (measurement[2] - measurement[0]) / max(h, 1e-6)

        mean = np.array([cx, cy, a, h, 0, 0, 0, 0], dtype=np.float64)

        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            1e-2,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            1e-5,
            10 * self._std_weight_velocity * h,
        ]
        covariance = np.diag(np.square(std)).astype(np.float64)
        return mean, covariance

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Kalman predict step."""
        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + self.Q
        return mean, covariance

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Kalman update step.

        Args:
            measurement: [x1, y1, x2, y2] detection box
        """
        meas = self._xyxy_to_state(measurement)

        S = self.H @ covariance @ self.H.T + self.R
        K = covariance @ self.H.T @ np.linalg.inv(S)

        innovation = meas - self.H @ mean
        mean = mean + K @ innovation
        covariance = (np.eye(8) - K @ self.H) @ covariance
        return mean, covariance

    def gating_distance(
        self, mean: np.ndarray, covariance: np.ndarray, measurements: np.ndarray
    ) -> np.ndarray:
        """Mahalanobis distance from predicted state to each measurement."""
        projected_mean = self.H @ mean
        projected_cov = self.H @ covariance @ self.H.T + self.R

        meas_states = np.array([self._xyxy_to_state(m) for m in measurements])
        diff = meas_states - projected_mean
        try:
            chol = np.linalg.cholesky(projected_cov)
            z = np.linalg.solve(chol, diff.T)
            return np.sum(z ** 2, axis=0)
        except np.linalg.LinAlgError:
            return np.full(len(measurements), 1e9)

    @staticmethod
    def _xyxy_to_state(box: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] -> [cx, cy, aspect, height]."""
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        h = box[3] - box[1]
        a = (box[2] - box[0]) / max(h, 1e-6)
        return np.array([cx, cy, a, h], dtype=np.float64)

    @staticmethod
    def state_to_xyxy(mean: np.ndarray) -> np.ndarray:
        """Convert Kalman mean -> [x1, y1, x2, y2]."""
        cx, cy, a, h = mean[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)
