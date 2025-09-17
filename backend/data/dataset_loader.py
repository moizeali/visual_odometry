"""
Dataset loading utilities for various visual odometry datasets
Supports KITTI, TUM RGB-D, EuRoC, and custom datasets
"""

import os
import cv2
import numpy as np
import pandas as pd
import requests
import zipfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator
from dataclasses import dataclass
import json
import logging
from tqdm import tqdm

@dataclass
class DatasetInfo:
    """Dataset information structure"""
    name: str
    type: str  # 'stereo', 'mono', 'rgbd'
    sequences: List[str]
    download_url: str
    size_mb: int
    description: str

# Available datasets for download
AVAILABLE_DATASETS = {
    'kitti_odometry': DatasetInfo(
        name="KITTI Odometry",
        type="stereo",
        sequences=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10'],
        download_url="http://www.cvlibs.net/datasets/kitti/eval_odometry.php",
        size_mb=22000,
        description="KITTI Vision benchmark suite - Odometry sequences"
    ),
    'tum_rgbd': DatasetInfo(
        name="TUM RGB-D",
        type="rgbd",
        sequences=['freiburg1_xyz', 'freiburg1_rpy', 'freiburg2_xyz', 'freiburg3_long'],
        download_url="https://vision.in.tum.de/data/datasets/rgbd-dataset/download",
        size_mb=2000,
        description="TUM RGB-D SLAM Dataset and Benchmark"
    ),
    'euroc_mav': DatasetInfo(
        name="EuRoC MAV",
        type="stereo",
        sequences=['MH_01', 'MH_02', 'MH_03', 'MH_04', 'MH_05', 'V1_01', 'V1_02', 'V1_03', 'V2_01', 'V2_02'],
        download_url="http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets",
        size_mb=1500,
        description="EuRoC Micro Aerial Vehicle datasets"
    )
}

class DatasetDownloader:
    """Download and manage datasets"""

    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def list_available_datasets(self) -> Dict[str, DatasetInfo]:
        """List all available datasets"""
        return AVAILABLE_DATASETS

    def download_sample_data(self) -> str:
        """Download sample data for testing (creates synthetic data)"""
        sample_dir = self.data_dir / "sample"
        sample_dir.mkdir(exist_ok=True)

        # Create sample images
        self._create_sample_images(sample_dir)

        # Create sample calibration
        self._create_sample_calibration(sample_dir)

        self.logger.info(f"Sample data created in {sample_dir}")
        return str(sample_dir)

    def _create_sample_images(self, output_dir: Path):
        """Create synthetic images for testing"""
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        # Create 50 sample images with moving pattern
        for i in range(50):
            # Create a simple moving pattern
            img = np.zeros((480, 640, 3), dtype=np.uint8)

            # Add some features (rectangles, circles)
            cv2.rectangle(img, (100 + i*2, 100), (200 + i*2, 200), (255, 0, 0), -1)
            cv2.circle(img, (300 + i, 300), 50, (0, 255, 0), -1)
            cv2.rectangle(img, (400 - i, 150 + i), (500 - i, 250 + i), (0, 0, 255), -1)

            # Add some noise
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)

            cv2.imwrite(str(images_dir / f"frame_{i:06d}.png"), img)

    def _create_sample_calibration(self, output_dir: Path):
        """Create sample camera calibration"""
        calibration = {
            "camera_matrix": {
                "fx": 525.0,
                "fy": 525.0,
                "cx": 319.5,
                "cy": 239.5
            },
            "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
            "image_size": [640, 480],
            "baseline": 0.075  # For stereo (if applicable)
        }

        with open(output_dir / "calibration.json", 'w') as f:
            json.dump(calibration, f, indent=2)

    def download_from_url(self, url: str, filename: str) -> bool:
        """Download file from URL with progress bar"""
        try:
            response = requests.get(url, stream=True)
            total_size = int(response.headers.get('content-length', 0))

            output_path = self.data_dir / filename
            with open(output_path, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            self.logger.info(f"Downloaded {filename} to {output_path}")
            return True

        except Exception as e:
            self.logger.error(f"Error downloading {filename}: {e}")
            return False

class KITTILoader:
    """KITTI dataset loader"""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)
        self.sequences_path = self.data_path / "sequences"
        self.poses_path = self.data_path / "poses"

    def load_sequence(self, sequence: str) -> Dict:
        """Load a KITTI sequence"""
        seq_path = self.sequences_path / sequence

        if not seq_path.exists():
            raise FileNotFoundError(f"Sequence {sequence} not found at {seq_path}")

        # Load calibration
        calib_file = seq_path / "calib.txt"
        calibration = self._load_calibration(calib_file)

        # Get image paths
        image_0_path = seq_path / "image_0"  # Left camera
        image_1_path = seq_path / "image_1"  # Right camera

        left_images = sorted(list(image_0_path.glob("*.png")))
        right_images = sorted(list(image_1_path.glob("*.png")))

        # Load ground truth poses (if available)
        poses_file = self.poses_path / f"{sequence}.txt"
        ground_truth = None
        if poses_file.exists():
            ground_truth = self._load_poses(poses_file)

        return {
            'sequence': sequence,
            'calibration': calibration,
            'left_images': [str(p) for p in left_images],
            'right_images': [str(p) for p in right_images],
            'ground_truth': ground_truth,
            'num_frames': len(left_images)
        }

    def _load_calibration(self, calib_file: Path) -> Dict:
        """Load KITTI calibration file"""
        calibration = {}

        if not calib_file.exists():
            # Return default calibration if file doesn't exist
            return {
                'P0': np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, 0.000000000000e+00],
                               [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00],
                               [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]]),
                'P1': np.array([[7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02, -3.861448000000e+02],
                               [0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02, 0.000000000000e+00],
                               [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 0.000000000000e+00]])
            }

        with open(calib_file, 'r') as f:
            for line in f:
                if line.strip():
                    key, values = line.strip().split(':', 1)
                    calibration[key] = np.fromstring(values, sep=' ').reshape(3, 4)

        return calibration

    def _load_poses(self, poses_file: Path) -> np.ndarray:
        """Load ground truth poses"""
        poses = []
        with open(poses_file, 'r') as f:
            for line in f:
                pose = np.fromstring(line, sep=' ').reshape(3, 4)
                # Convert to 4x4 matrix
                pose_4x4 = np.eye(4)
                pose_4x4[:3, :] = pose
                poses.append(pose_4x4)
        return np.array(poses)

class CustomDatasetLoader:
    """Loader for custom datasets"""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load_image_sequence(self, pattern: str = "*.png") -> List[str]:
        """Load sequence of images matching pattern"""
        image_paths = sorted(list(self.data_path.glob(pattern)))
        return [str(p) for p in image_paths]

    def load_calibration(self, calib_file: str = "calibration.json") -> Dict:
        """Load calibration from JSON file"""
        calib_path = self.data_path / calib_file

        if not calib_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_path}")

        with open(calib_path, 'r') as f:
            calibration = json.load(f)

        return calibration

class DatasetManager:
    """Main dataset management class"""

    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.downloader = DatasetDownloader(data_dir)
        self.logger = logging.getLogger(__name__)

    def list_datasets(self) -> Dict[str, DatasetInfo]:
        """List available datasets"""
        return self.downloader.list_available_datasets()

    def prepare_sample_data(self) -> str:
        """Prepare sample data for testing"""
        return self.downloader.download_sample_data()

    def load_dataset(self, dataset_name: str, sequence: str = None) -> Dict:
        """Load a specific dataset"""
        if dataset_name == 'sample':
            # Load sample data
            sample_dir = self.data_dir / "sample"
            if not sample_dir.exists():
                self.prepare_sample_data()

            loader = CustomDatasetLoader(sample_dir)
            images = loader.load_image_sequence("images/*.png")
            calibration = loader.load_calibration()

            return {
                'type': 'mono',
                'images': images,
                'calibration': calibration,
                'sequence': 'sample',
                'num_frames': len(images)
            }

        elif dataset_name == 'kitti':
            # Load KITTI data
            kitti_dir = self.data_dir / "kitti"
            if not kitti_dir.exists():
                raise FileNotFoundError(f"KITTI dataset not found. Please download to {kitti_dir}")

            loader = KITTILoader(kitti_dir)
            if sequence is None:
                sequence = "00"  # Default sequence

            return loader.load_sequence(sequence)

        elif dataset_name == 'custom':
            # Load custom data
            custom_dir = self.data_dir / "custom"
            if not custom_dir.exists():
                raise FileNotFoundError(f"Custom dataset directory not found: {custom_dir}")

            loader = CustomDatasetLoader(custom_dir)
            images = loader.load_image_sequence()

            try:
                calibration = loader.load_calibration()
            except FileNotFoundError:
                # Use default calibration
                calibration = {
                    "camera_matrix": {"fx": 525.0, "fy": 525.0, "cx": 319.5, "cy": 239.5},
                    "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
                    "image_size": [640, 480]
                }

            return {
                'type': 'mono',
                'images': images,
                'calibration': calibration,
                'sequence': 'custom',
                'num_frames': len(images)
            }

        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

    def get_dataset_info(self, dataset_name: str) -> Optional[DatasetInfo]:
        """Get information about a dataset"""
        return AVAILABLE_DATASETS.get(dataset_name)

    def upload_custom_images(self, image_files: List[str]) -> str:
        """Upload custom images to the dataset directory"""
        custom_dir = self.data_dir / "custom" / "images"
        custom_dir.mkdir(parents=True, exist_ok=True)

        uploaded_files = []
        for i, image_file in enumerate(image_files):
            if isinstance(image_file, str) and os.path.exists(image_file):
                # Copy file to custom directory
                filename = f"frame_{i:06d}.png"
                output_path = custom_dir / filename

                img = cv2.imread(image_file)
                if img is not None:
                    cv2.imwrite(str(output_path), img)
                    uploaded_files.append(str(output_path))

        self.logger.info(f"Uploaded {len(uploaded_files)} images to {custom_dir}")
        return str(custom_dir.parent)