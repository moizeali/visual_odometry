"""
Enhanced Visual Odometry Implementation
Supports stereo and monocular visual odometry with multiple algorithms
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Optional
import logging
from dataclasses import dataclass
from pathlib import Path

@dataclass
class CameraParams:
    """Camera calibration parameters"""
    fx: float  # Focal length x
    fy: float  # Focal length y
    cx: float  # Principal point x
    cy: float  # Principal point y
    baseline: float = 0.0  # Stereo baseline (for stereo VO)

    @property
    def K(self) -> np.ndarray:
        """Camera intrinsic matrix"""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])

@dataclass
class Pose:
    """Camera pose representation"""
    R: np.ndarray  # Rotation matrix (3x3)
    t: np.ndarray  # Translation vector (3x1)
    timestamp: float = 0.0

    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix"""
        T = np.eye(4)
        T[:3, :3] = self.R
        T[:3, 3] = self.t.flatten()
        return T

    def inverse(self) -> 'Pose':
        """Get inverse pose"""
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return Pose(R_inv, t_inv, self.timestamp)

class FeatureDetector:
    """Feature detection and description"""

    def __init__(self, detector_type: str = 'ORB'):
        self.detector_type = detector_type
        self.detector = self._create_detector()

    def _create_detector(self):
        """Create feature detector based on type"""
        if self.detector_type == 'ORB':
            return cv2.ORB_create(nfeatures=1000)
        elif self.detector_type == 'SIFT':
            return cv2.SIFT_create(nfeatures=1000)
        elif self.detector_type == 'SURF':
            return cv2.xfeatures2d.SURF_create(hessianThreshold=400)
        else:
            raise ValueError(f"Unsupported detector type: {self.detector_type}")

    def detect_and_compute(self, image: np.ndarray) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """Detect keypoints and compute descriptors"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors

class FeatureMatcher:
    """Feature matching between frames"""

    def __init__(self, matcher_type: str = 'BF'):
        self.matcher_type = matcher_type
        self.matcher = self._create_matcher()

    def _create_matcher(self):
        """Create feature matcher"""
        if self.matcher_type == 'BF':
            return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        elif self.matcher_type == 'FLANN':
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            return cv2.FlannBasedMatcher(index_params, search_params)
        else:
            raise ValueError(f"Unsupported matcher type: {self.matcher_type}")

    def match_features(self, desc1: np.ndarray, desc2: np.ndarray,
                      ratio_threshold: float = 0.7) -> List[cv2.DMatch]:
        """Match features between two frames"""
        if desc1 is None or desc2 is None:
            return []

        if self.matcher_type == 'BF':
            matches = self.matcher.match(desc1, desc2)
            # Sort by distance
            matches = sorted(matches, key=lambda x: x.distance)
        else:  # FLANN
            matches = self.matcher.knnMatch(desc1, desc2, k=2)
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < ratio_threshold * n.distance:
                        good_matches.append(m)
            matches = good_matches

        return matches

class MonocularVO:
    """Monocular Visual Odometry"""

    def __init__(self, camera_params: CameraParams,
                 detector_type: str = 'ORB',
                 matcher_type: str = 'BF'):
        self.camera_params = camera_params
        self.detector = FeatureDetector(detector_type)
        self.matcher = FeatureMatcher(matcher_type)

        self.trajectory = []
        self.current_pose = Pose(np.eye(3), np.zeros((3, 1)))
        self.previous_frame = None
        self.previous_keypoints = None
        self.previous_descriptors = None

        self.logger = logging.getLogger(__name__)

    def process_frame(self, frame: np.ndarray) -> Tuple[Pose, Dict]:
        """Process a single frame and estimate motion"""
        # Detect features
        keypoints, descriptors = self.detector.detect_and_compute(frame)

        info = {
            'keypoints_count': len(keypoints),
            'matches_count': 0,
            'inliers_count': 0
        }

        if self.previous_frame is not None:
            # Match features with previous frame
            matches = self.matcher.match_features(
                self.previous_descriptors, descriptors
            )
            info['matches_count'] = len(matches)

            if len(matches) > 8:  # Minimum for essential matrix
                # Extract matched points
                pts1 = np.float32([self.previous_keypoints[m.queryIdx].pt for m in matches])
                pts2 = np.float32([keypoints[m.trainIdx].pt for m in matches])

                # Estimate essential matrix
                E, mask = cv2.findEssentialMat(
                    pts1, pts2, self.camera_params.K,
                    method=cv2.RANSAC,
                    prob=0.999,
                    threshold=1.0
                )

                if E is not None:
                    inliers = mask.ravel().astype(bool)
                    info['inliers_count'] = np.sum(inliers)

                    # Recover pose
                    _, R, t, _ = cv2.recoverPose(
                        E, pts1[inliers], pts2[inliers], self.camera_params.K
                    )

                    # Update current pose
                    self.current_pose = Pose(
                        self.current_pose.R @ R,
                        self.current_pose.t + self.current_pose.R @ t,
                        len(self.trajectory)
                    )

        # Store for next iteration
        self.previous_frame = frame.copy()
        self.previous_keypoints = keypoints
        self.previous_descriptors = descriptors

        self.trajectory.append(self.current_pose)

        return self.current_pose, info

    def get_trajectory(self) -> np.ndarray:
        """Get trajectory as array of positions"""
        if not self.trajectory:
            return np.array([]).reshape(0, 3)

        positions = []
        for pose in self.trajectory:
            positions.append(pose.t.flatten())

        return np.array(positions)

    def reset(self):
        """Reset the visual odometry system"""
        self.trajectory = []
        self.current_pose = Pose(np.eye(3), np.zeros((3, 1)))
        self.previous_frame = None
        self.previous_keypoints = None
        self.previous_descriptors = None

class StereoVO:
    """Stereo Visual Odometry"""

    def __init__(self, camera_params: CameraParams,
                 detector_type: str = 'ORB',
                 matcher_type: str = 'BF'):
        self.camera_params = camera_params
        self.detector = FeatureDetector(detector_type)
        self.matcher = FeatureMatcher(matcher_type)

        self.trajectory = []
        self.current_pose = Pose(np.eye(3), np.zeros((3, 1)))

        self.logger = logging.getLogger(__name__)

    def triangulate_points(self, pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
        """Triangulate 3D points from stereo correspondences"""
        # Projection matrices for stereo setup
        P1 = self.camera_params.K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = self.camera_params.K @ np.hstack([
            np.eye(3),
            np.array([[-self.camera_params.baseline], [0], [0]])
        ])

        # Triangulate
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_4d[:3] / points_4d[3]

        return points_3d.T

    def process_stereo_pair(self, left_frame: np.ndarray,
                           right_frame: np.ndarray) -> Tuple[Pose, Dict]:
        """Process stereo image pair"""
        # Detect features in left image
        left_kp, left_desc = self.detector.detect_and_compute(left_frame)
        right_kp, right_desc = self.detector.detect_and_compute(right_frame)

        info = {
            'left_keypoints': len(left_kp),
            'right_keypoints': len(right_kp),
            'stereo_matches': 0,
            'triangulated_points': 0
        }

        # Match features between left and right images
        stereo_matches = self.matcher.match_features(left_desc, right_desc)
        info['stereo_matches'] = len(stereo_matches)

        if len(stereo_matches) > 8:
            # Extract matched points
            left_pts = np.float32([left_kp[m.queryIdx].pt for m in stereo_matches])
            right_pts = np.float32([right_kp[m.trainIdx].pt for m in stereo_matches])

            # Filter matches by epipolar constraint (y-coordinates should be similar)
            y_diff = np.abs(left_pts[:, 1] - right_pts[:, 1])
            valid_matches = y_diff < 2.0  # pixel threshold

            if np.sum(valid_matches) > 8:
                left_pts = left_pts[valid_matches]
                right_pts = right_pts[valid_matches]

                # Triangulate 3D points
                points_3d = self.triangulate_points(left_pts, right_pts)

                # Filter points by depth
                valid_depth = points_3d[:, 2] > 0
                points_3d = points_3d[valid_depth]

                info['triangulated_points'] = len(points_3d)

                # For now, just update pose counter (full stereo VO requires temporal matching)
                self.current_pose = Pose(
                    np.eye(3),
                    np.array([[len(self.trajectory)], [0], [0]]),
                    len(self.trajectory)
                )

        self.trajectory.append(self.current_pose)
        return self.current_pose, info

class VisualOdometryPipeline:
    """Main pipeline for visual odometry processing"""

    def __init__(self, camera_params: CameraParams,
                 mode: str = 'mono',
                 detector_type: str = 'ORB'):
        self.camera_params = camera_params
        self.mode = mode

        if mode == 'mono':
            self.vo = MonocularVO(camera_params, detector_type)
        elif mode == 'stereo':
            self.vo = StereoVO(camera_params, detector_type)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        self.processing_stats = []

    def process_sequence(self, image_paths: List[str]) -> Dict:
        """Process a sequence of images"""
        results = {
            'trajectory': [],
            'stats': [],
            'success': True,
            'error': None
        }

        try:
            for i, path in enumerate(image_paths):
                img = cv2.imread(path)
                if img is None:
                    continue

                if self.mode == 'mono':
                    pose, stats = self.vo.process_frame(img)
                else:  # stereo - would need left/right pairs
                    # For demo, process as mono
                    pose, stats = self.vo.process_frame(img)

                results['trajectory'].append({
                    'frame': i,
                    'position': pose.t.flatten().tolist(),
                    'rotation': pose.R.tolist()
                })
                results['stats'].append(stats)

        except Exception as e:
            results['success'] = False
            results['error'] = str(e)
            self.logger.error(f"Error processing sequence: {e}")

        return results

    def get_trajectory_array(self) -> np.ndarray:
        """Get trajectory as numpy array"""
        return self.vo.get_trajectory()

    def reset(self):
        """Reset the pipeline"""
        self.vo.reset()
        self.processing_stats = []