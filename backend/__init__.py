"""
Enhanced Visual Odometry System Backend
"""

__version__ = "1.0.0"
__author__ = "Syed Moiz Ali"
__email__ = "moizeali@gmail.com"

from .core.visual_odometry import (
    VisualOdometryPipeline,
    MonocularVO,
    StereoVO,
    CameraParams,
    Pose
)

from .data.dataset_loader import DatasetManager

__all__ = [
    'VisualOdometryPipeline',
    'MonocularVO',
    'StereoVO',
    'CameraParams',
    'Pose',
    'DatasetManager'
]