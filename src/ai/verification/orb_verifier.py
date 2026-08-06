"""
ORB local-feature geometric verification for near-duplicate tile disambiguation.

Runs only on the short candidate list after hybrid scoring. Does not touch the
FAISS index or TileFeatures. Pure OpenCV, CPU-only, cross-platform.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger("tilevision.ai.verification.orb_verifier")


class OrbVerifier:
    """ORB keypoint + RANSAC inlier ratio for geometric near-duplicate checks."""

    def __init__(
        self,
        n_features: int = 500,
        ratio_thresh: float = 0.75,
        min_inliers: int = 8,
    ) -> None:
        self.n_features = max(8, int(n_features))
        self.ratio_thresh = float(ratio_thresh)
        self.min_inliers = max(1, int(min_inliers))

    def score(self, query_gray: np.ndarray, candidate_gray: np.ndarray) -> float:
        """
        Return a bounded [0, 1] geometric match score.

        Uses ORB descriptors, Lowe ratio test, and RANSAC homography inliers.
        Any OpenCV / shape failure returns 0.0 (never raises).
        """
        try:
            q = self._as_gray_u8(query_gray)
            c = self._as_gray_u8(candidate_gray)
            if q is None or c is None:
                return 0.0

            orb = cv2.ORB_create(nfeatures=self.n_features)
            kp_q, des_q = orb.detectAndCompute(q, None)
            kp_c, des_c = orb.detectAndCompute(c, None)

            if (
                des_q is None
                or des_c is None
                or len(kp_q) == 0
                or len(kp_c) == 0
            ):
                return 0.0

            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            raw_pairs = matcher.knnMatch(des_q, des_c, k=2)

            good: list = []
            for pair in raw_pairs:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < self.ratio_thresh * n.distance:
                    good.append(m)

            if len(good) < 4:
                return 0.0

            src = np.float32([kp_q[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst = np.float32([kp_c[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
            _H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
            if mask is None:
                return 0.0

            inliers = int(mask.ravel().sum())
            if inliers <= 0:
                return 0.0

            denom = max(self.min_inliers, min(len(kp_q), len(kp_c)))
            return float(max(0.0, min(1.0, inliers / float(denom))))
        except Exception as exc:
            logger.debug("ORB verification failed: %s", exc, exc_info=True)
            return 0.0

    @staticmethod
    def _as_gray_u8(image: np.ndarray) -> np.ndarray | None:
        if image is None:
            return None
        arr = np.asarray(image)
        if arr.size == 0:
            return None
        if arr.ndim == 3:
            if arr.shape[2] == 3:
                arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
            else:
                arr = arr[:, :, 0]
        if arr.ndim != 2:
            return None
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        # Tiny images rarely produce useful keypoints.
        if min(arr.shape[:2]) < 16:
            return None
        return arr
