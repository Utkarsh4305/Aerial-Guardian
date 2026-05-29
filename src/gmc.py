"""
Global Motion Compensation (GMC) for drone ego-motion.

Problem:
  A drone camera moves continuously. Without compensation, the Kalman filter's
  velocity estimate (vx, vy) is polluted by camera motion, causing predicted
  track positions to diverge from actual detections -> cascading ID switches.

Approach:
  1. Extract sparse feature points on the previous grayscale frame (Shi-Tomasi).
  2. Track them to the current frame with Lucas-Kanade optical flow.
  3. Estimate a 3x3 homography (perspective transform) between matched point sets
     using RANSAC to reject moving-object outliers.
  4. Apply the homography to each track's center point so that all track positions
     are expressed in the current frame's coordinate system before association.
"""
from __future__ import annotations

import cv2
import numpy as np


class GMC:
    def __init__(self, cfg: dict) -> None:
        gmc_cfg = cfg["gmc"]
        self.max_corners: int = gmc_cfg["max_corners"]
        self.quality_level: float = gmc_cfg["quality_level"]
        self.min_distance: float = gmc_cfg["min_distance"]
        self.ransac_thresh: float = gmc_cfg["ransac_reproj_threshold"]

        self._prev_gray: np.ndarray | None = None
        self._prev_pts: np.ndarray | None = None

        # Lucas-Kanade optical flow params
        self._lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )

    def apply(
        self,
        frame: np.ndarray,
        track_means: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate camera motion and compensate track positions.

        Args:
            frame:        Current BGR frame.
            track_means:  (N, 4) array of track state [cx, cy, aspect, height].

        Returns:
            (compensated_means, H): warped track means and 3x3 homography.
                                    H is identity if estimation fails.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        H = np.eye(3, dtype=np.float64)

        if self._prev_gray is None or self._prev_pts is None or len(self._prev_pts) == 0:
            self._update_keypoints(gray)
            return track_means.copy(), H

        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, gray, self._prev_pts, None, **self._lk_params
        )

        if curr_pts is None or status is None:
            self._update_keypoints(gray)
            return track_means.copy(), H

        good_prev = self._prev_pts[status.ravel() == 1]
        good_curr = curr_pts[status.ravel() == 1]

        if len(good_prev) >= 4:
            H, inlier_mask = cv2.findHomography(
                good_prev, good_curr, cv2.RANSAC, self.ransac_thresh
            )
            if H is None:
                H = np.eye(3, dtype=np.float64)

        self._update_keypoints(gray)

        if track_means.shape[0] == 0:
            return track_means.copy(), H

        compensated = self._warp_means(track_means, H)
        return compensated, H

    def _warp_means(self, means: np.ndarray, H: np.ndarray) -> np.ndarray:
        """Apply homography to [cx, cy] of each track, leave aspect/height unchanged."""
        out = means.copy()
        pts = means[:, :2].reshape(-1, 1, 2).astype(np.float32)
        warped = cv2.perspectiveTransform(pts, H.astype(np.float32))
        if warped is not None:
            out[:, :2] = warped.reshape(-1, 2)
        return out

    def _update_keypoints(self, gray: np.ndarray) -> None:
        self._prev_gray = gray
        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
        )
        self._prev_pts = pts if pts is not None else np.empty((0, 1, 2), dtype=np.float32)

    def reset(self) -> None:
        self._prev_gray = None
        self._prev_pts = None
