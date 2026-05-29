"""
ByteTrack with Global Motion Compensation -- drone-adapted MOT tracker.

ByteTrack two-step association strategy:
  Round 1: high-confidence detections (conf > high_thresh) matched to
           existing Tracked+Lost tracks via IoU.
  Round 2: low-confidence detections (low_thresh < conf <= high_thresh) matched
           to the remaining unmatched Tracked tracks.

This prevents low-confidence detections (common for small aerial objects)
from being discarded, which is a key strength over simpler trackers.

GMC integration:
  Before Round 1, each track's Kalman-predicted position is warped using the
  current frame's homography so that camera movement doesn't count as object
  movement during association.
"""
from __future__ import annotations

from enum import IntEnum

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.kalman_filter import KalmanFilter
from src.gmc import GMC


class TrackState(IntEnum):
    Tracked = 0
    Lost = 1
    Removed = 2


class Track:
    _next_id = 1

    def __init__(self, det: np.ndarray, kf: KalmanFilter) -> None:
        self.track_id = Track._next_id
        Track._next_id += 1

        self.kf = kf
        self.mean, self.covariance = kf.initiate(det[:4])
        self.conf = float(det[4])
        self.state = TrackState.Tracked
        self.time_since_update = 0
        self.hits = 1
        # Store centre history for trajectory visualisation
        self.history: list[tuple[int, int]] = [self._centre()]

    def predict(self) -> None:
        self.mean, self.covariance = self.kf.predict(self.mean, self.covariance)
        self.time_since_update += 1

    def update(self, det: np.ndarray) -> None:
        self.mean, self.covariance = self.kf.update(self.mean, self.covariance, det[:4])
        self.conf = float(det[4])
        self.state = TrackState.Tracked
        self.time_since_update = 0
        self.hits += 1
        self.history.append(self._centre())

    def apply_homography(self, H: np.ndarray) -> None:
        """Warp the track's [cx, cy] using homography H (drone motion compensation)."""
        import cv2
        pts = self.mean[:2].reshape(1, 1, 2).astype(np.float32)
        warped = cv2.perspectiveTransform(pts, H.astype(np.float32))
        if warped is not None:
            self.mean[:2] = warped.reshape(2)

    def to_xyxy(self) -> np.ndarray:
        return KalmanFilter.state_to_xyxy(self.mean)

    def _centre(self) -> tuple[int, int]:
        cx, cy = self.mean[0], self.mean[1]
        return (int(cx), int(cy))

    @classmethod
    def reset_id(cls) -> None:
        cls._next_id = 1


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Compute IoU matrix between two sets of [x1, y1, x2, y2] boxes."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = boxes_a[:, None, :]  # (A, 1, 4)
    b = boxes_b[None, :, :]  # (1, B, 4)

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])

    union = area_a + area_b - inter_area
    return (inter_area / np.maximum(union, 1e-6)).astype(np.float32)


def _linear_assignment(cost_matrix: np.ndarray, thresh: float):
    """Hungarian assignment; returns (matched, unmatched_rows, unmatched_cols)."""
    if cost_matrix.size == 0:
        rows = list(range(cost_matrix.shape[0]))
        cols = list(range(cost_matrix.shape[1]))
        return [], rows, cols

    row_idx, col_idx = linear_sum_assignment(cost_matrix)
    matched, unmatched_rows, unmatched_cols = [], [], []

    matched_set = set()
    for r, c in zip(row_idx, col_idx):
        if cost_matrix[r, c] <= thresh:
            matched.append((r, c))
            matched_set.add((r, c))

    for r in range(cost_matrix.shape[0]):
        if not any(r == m[0] for m in matched):
            unmatched_rows.append(r)
    for c in range(cost_matrix.shape[1]):
        if not any(c == m[1] for m in matched):
            unmatched_cols.append(c)

    return matched, unmatched_rows, unmatched_cols


class ByteTracker:
    def __init__(self, cfg: dict) -> None:
        tracker_cfg = cfg["tracker"]
        kf_cfg = cfg["kalman"]

        self.high_thresh = tracker_cfg["high_conf_thresh"]
        self.low_thresh = tracker_cfg["low_conf_thresh"]
        self.match_thresh = tracker_cfg["match_thresh"]
        self.second_match_thresh = tracker_cfg["second_match_thresh"]
        self.max_time_lost = tracker_cfg["max_time_lost"]

        self.kf = KalmanFilter(
            q_xy_scale=kf_cfg["q_xy_scale"],
            q_vel_scale=kf_cfg["q_vel_scale"],
            r_scale=kf_cfg["r_scale"],
        )
        self._gmc_enabled: bool = cfg["gmc"]["enabled"]
        self.gmc = GMC(cfg)

        self.tracked_tracks: list[Track] = []
        self.lost_tracks: list[Track] = []
        self.frame_id = 0

    def update(self, detections: np.ndarray, frame: np.ndarray) -> list[Track]:
        """
        Process one frame.

        Args:
            detections: (N, 5) [x1, y1, x2, y2, conf]
            frame:      BGR frame (used for GMC)

        Returns:
            List of active Track objects.
        """
        self.frame_id += 1

        # --- Predict all tracks ---
        all_tracks = self.tracked_tracks + self.lost_tracks
        for t in all_tracks:
            t.predict()

        # --- GMC: compensate drone motion on predicted positions ---
        if self._gmc_enabled:
            if len(all_tracks) > 0:
                means = np.array([t.mean[:4] for t in all_tracks])
                compensated, H = self.gmc.apply(frame, means)
                for i, t in enumerate(all_tracks):
                    t.mean[:4] = compensated[i]
            else:
                self.gmc.apply(frame, np.empty((0, 4)))

        # --- Split detections by confidence ---
        if len(detections) == 0:
            high_dets = np.empty((0, 5), dtype=np.float32)
            low_dets = np.empty((0, 5), dtype=np.float32)
        else:
            high_mask = detections[:, 4] >= self.high_thresh
            high_dets = detections[high_mask]
            low_mask = (detections[:, 4] >= self.low_thresh) & ~high_mask
            low_dets = detections[low_mask]

        # --- Round 1: high-conf dets <-> tracked+lost tracks ---
        active = [t for t in all_tracks if t.state == TrackState.Tracked]
        lost = [t for t in all_tracks if t.state == TrackState.Lost]

        matched1, unmatched_tracks1, unmatched_dets1 = self._associate(
            active + lost, high_dets, self.match_thresh
        )

        for ti, di in matched1:
            (active + lost)[ti].update(high_dets[di])

        unmatched_tracks_after1 = [active[i] for i in unmatched_tracks1 if i < len(active)]

        # --- Round 2: low-conf dets <-> unmatched active tracks ---
        matched2, unmatched_tracks2, _ = self._associate(
            unmatched_tracks_after1, low_dets, self.second_match_thresh
        )
        for ti, di in matched2:
            unmatched_tracks_after1[ti].update(low_dets[di])

        still_unmatched = [unmatched_tracks_after1[i] for i in unmatched_tracks2]
        for t in still_unmatched:
            t.state = TrackState.Lost

        # --- New tracks from unmatched high-conf detections ---
        matched_det_ids = {di for _, di in matched1}
        for di in unmatched_dets1:
            if di not in matched_det_ids and high_dets[di][4] >= self.high_thresh:
                new_track = Track(high_dets[di], self.kf)
                self.tracked_tracks.append(new_track)

        # --- Rebuild track lists ---
        # Keep only un-recovered lost tracks (state is still Lost)
        self.lost_tracks = [
            t for t in (lost + still_unmatched)
            if t.state == TrackState.Lost and t.time_since_update <= self.max_time_lost
        ]
        # Active confirmed tracks + new single-hit tentatives from this frame
        self.tracked_tracks = [
            t for t in (active + lost)
            if t.state == TrackState.Tracked
        ] + [t for t in self.tracked_tracks if t.hits == 1 and t.state == TrackState.Tracked]

        return [t for t in self.tracked_tracks if t.state == TrackState.Tracked]

    def _associate(
        self,
        tracks: list[Track],
        detections: np.ndarray,
        iou_thresh: float,
    ):
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(tracks))), list(range(len(detections)))

        track_boxes = np.array([t.to_xyxy() for t in tracks])
        det_boxes = detections[:, :4]

        iou_mat = iou_batch(track_boxes, det_boxes)
        cost_mat = 1.0 - iou_mat

        return _linear_assignment(cost_mat, 1.0 - iou_thresh)

    def reset(self) -> None:
        Track.reset_id()
        self.tracked_tracks.clear()
        self.lost_tracks.clear()
        self.frame_id = 0
        self.gmc.reset()
