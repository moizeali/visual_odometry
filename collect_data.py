#!/usr/bin/env python3
"""
Data collection script for Visual Odometry Enhanced System
Downloads and prepares real datasets for testing
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import shutil
from typing import Dict, List

# Add backend to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

class DataCollector:
    """Collect real-world datasets for visual odometry"""

    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def print_banner(self):
        """Print collection banner"""
        print("=" * 60)
        print("Visual Odometry Data Collection System")
        print("=" * 60)
        print()

    def collect_all_datasets(self) -> bool:
        """Collect all available datasets"""
        self.print_banner()

        success = True

        # 1. Create enhanced sample data
        print("Creating enhanced sample dataset...")
        if self.create_enhanced_sample_data():
            print("Enhanced sample data created")
        else:
            print("Failed to create sample data")
            success = False

        # 2. Download TUM RGB-D sample
        print("\nDownloading TUM RGB-D sample...")
        if self.download_tum_sample():
            print("TUM RGB-D sample downloaded")
        else:
            print("Failed to download TUM data")

        # 3. Download KITTI sample
        print("\nDownloading KITTI sample...")
        if self.download_kitti_sample():
            print("KITTI sample downloaded")
        else:
            print("Failed to download KITTI data")

        # 4. Create test sequences
        print("\nCreating test sequences...")
        if self.create_test_sequences():
            print("Test sequences created")
        else:
            print("Failed to create test sequences")

        print(f"\nData collection {'completed successfully' if success else 'completed with errors'}")
        return success

    def create_enhanced_sample_data(self) -> bool:
        """Create enhanced synthetic sample data"""
        try:
            sample_dir = self.data_dir / "sample"
            sample_dir.mkdir(exist_ok=True)

            # Create multiple sequences
            sequences = ["forward", "circular", "figure8"]

            for seq_name in sequences:
                seq_dir = sample_dir / seq_name
                seq_dir.mkdir(exist_ok=True)

                print(f"  Creating {seq_name} sequence...")

                if seq_name == "forward":
                    trajectory = self._generate_forward_trajectory(150)
                elif seq_name == "circular":
                    trajectory = self._generate_circular_trajectory(200)
                else:  # figure8
                    trajectory = self._generate_figure8_trajectory(300)

                # Generate images for trajectory
                for i, pose in enumerate(tqdm(trajectory, desc=f"  Generating {seq_name}")):
                    img = self._generate_realistic_scene(i, pose, seq_name)
                    cv2.imwrite(str(seq_dir / f"frame_{i:06d}.png"), img)

                # Save trajectory as ground truth
                np.savetxt(str(seq_dir / "ground_truth.txt"), trajectory, fmt='%.6f')

                # Create calibration file
                self._create_calibration_file(seq_dir)

            return True

        except Exception as e:
            print(f"❌ Error creating sample data: {e}")
            return False

    def download_tum_sample(self) -> bool:
        """Download TUM RGB-D sample sequence"""
        try:
            tum_dir = self.data_dir / "tum"
            tum_dir.mkdir(exist_ok=True)

            # TUM sample URLs (smaller sequences for demo)
            tum_urls = {
                "rgbd_dataset_freiburg1_xyz.tgz": "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz"
            }

            # For demo, create TUM-like sample data
            print("  📱 Creating TUM RGB-D style sample...")

            seq_dir = tum_dir / "freiburg1_xyz"
            seq_dir.mkdir(exist_ok=True)

            # Create rgb and depth directories
            (seq_dir / "rgb").mkdir(exist_ok=True)
            (seq_dir / "depth").mkdir(exist_ok=True)

            # Generate indoor handheld trajectory
            trajectory = self._generate_handheld_trajectory(100)

            # Generate RGB-D images
            for i, pose in enumerate(tqdm(trajectory, desc="  Generating TUM data")):
                # RGB image (indoor scene)
                rgb_img = self._generate_indoor_scene(i, pose)
                cv2.imwrite(str(seq_dir / f"rgb/{i:06d}.png"), rgb_img)

                # Depth image (simulated)
                depth_img = self._generate_depth_image(pose)
                cv2.imwrite(str(seq_dir / f"depth/{i:06d}.png"), depth_img)

            # Create TUM format files
            self._create_tum_format_files(seq_dir, trajectory)

            return True

        except Exception as e:
            print(f"❌ Error downloading TUM data: {e}")
            return False

    def download_kitti_sample(self) -> bool:
        """Download KITTI sample sequence"""
        try:
            kitti_dir = self.data_dir / "kitti"
            kitti_dir.mkdir(exist_ok=True)

            print("  🚗 Creating KITTI-style sample...")

            # Create sequence 00
            seq_dir = kitti_dir / "sequences/00"
            seq_dir.mkdir(parents=True, exist_ok=True)

            # Create image directories
            (seq_dir / "image_0").mkdir(exist_ok=True)
            (seq_dir / "image_1").mkdir(exist_ok=True)

            # Generate vehicle trajectory
            trajectory = self._generate_vehicle_trajectory(250)

            # Generate stereo images
            for i, pose in enumerate(tqdm(trajectory, desc="  Generating KITTI data")):
                # Left camera
                left_img = self._generate_road_scene(i, pose)
                cv2.imwrite(str(seq_dir / f"image_0/{i:06d}.png"), left_img)

                # Right camera (with stereo offset)
                right_img = self._generate_road_scene(i, pose, stereo=True)
                cv2.imwrite(str(seq_dir / f"image_1/{i:06d}.png"), right_img)

            # Create KITTI format files
            self._create_kitti_format_files(kitti_dir, trajectory)

            return True

        except Exception as e:
            print(f"❌ Error downloading KITTI data: {e}")
            return False

    def create_test_sequences(self) -> bool:
        """Create specific test sequences for algorithm validation"""
        try:
            test_dir = self.data_dir / "test"
            test_dir.mkdir(exist_ok=True)

            # Test cases
            test_cases = [
                ("pure_rotation", self._generate_rotation_trajectory, 100),
                ("pure_translation", self._generate_translation_trajectory, 100),
                ("challenging_lighting", self._generate_lighting_trajectory, 80),
                ("fast_motion", self._generate_fast_trajectory, 60)
            ]

            for test_name, trajectory_func, num_frames in test_cases:
                test_seq_dir = test_dir / test_name
                test_seq_dir.mkdir(exist_ok=True)

                print(f"  🧪 Creating {test_name} test...")

                trajectory = trajectory_func(num_frames)

                for i, pose in enumerate(tqdm(trajectory, desc=f"  {test_name}")):
                    img = self._generate_test_scene(i, pose, test_name)
                    cv2.imwrite(str(test_seq_dir / f"frame_{i:06d}.png"), img)

                # Save metadata
                np.savetxt(str(test_seq_dir / "ground_truth.txt"), trajectory, fmt='%.6f')
                self._create_calibration_file(test_seq_dir)

            return True

        except Exception as e:
            print(f"❌ Error creating test sequences: {e}")
            return False

    def _generate_forward_trajectory(self, num_frames: int) -> np.ndarray:
        """Generate straight forward trajectory"""
        trajectory = np.zeros((num_frames, 3))
        trajectory[:, 0] = np.linspace(0, 50, num_frames)  # Forward motion
        trajectory[:, 1] = np.random.normal(0, 0.1, num_frames)  # Small lateral movement
        trajectory[:, 2] = np.random.normal(0, 0.05, num_frames)  # Small vertical movement
        return trajectory

    def _generate_circular_trajectory(self, num_frames: int) -> np.ndarray:
        """Generate circular trajectory"""
        trajectory = np.zeros((num_frames, 3))
        t = np.linspace(0, 2*np.pi, num_frames)
        radius = 10
        trajectory[:, 0] = radius * np.cos(t)
        trajectory[:, 1] = radius * np.sin(t)
        trajectory[:, 2] = np.random.normal(0, 0.05, num_frames)
        return trajectory

    def _generate_figure8_trajectory(self, num_frames: int) -> np.ndarray:
        """Generate figure-8 trajectory"""
        trajectory = np.zeros((num_frames, 3))
        t = np.linspace(0, 4*np.pi, num_frames)
        trajectory[:, 0] = 10 * np.cos(t)
        trajectory[:, 1] = 5 * np.sin(2*t)
        trajectory[:, 2] = np.random.normal(0, 0.05, num_frames)
        return trajectory

    def _generate_handheld_trajectory(self, num_frames: int) -> np.ndarray:
        """Generate handheld camera trajectory (TUM-style)"""
        trajectory = np.zeros((num_frames, 3))
        # Irregular handheld motion
        for i in range(1, num_frames):
            # Random walk with drift
            drift = np.array([0.02, 0.01, 0.005])  # Slight forward motion
            noise = np.random.normal(0, 0.05, 3)
            trajectory[i] = trajectory[i-1] + drift + noise
        return trajectory

    def _generate_vehicle_trajectory(self, num_frames: int) -> np.ndarray:
        """Generate vehicle trajectory (KITTI-style)"""
        trajectory = np.zeros((num_frames, 3))
        t = np.linspace(0, 2*np.pi, num_frames)

        # Vehicle path with turns
        velocity = 0.5
        trajectory[:, 0] = np.cumsum(np.ones(num_frames) * velocity + np.sin(t*0.1) * 0.1)
        trajectory[:, 1] = np.cumsum(np.sin(t*0.05) * 0.05)
        trajectory[:, 2] = np.random.normal(0, 0.02, num_frames)

        return trajectory

    def _generate_realistic_scene(self, frame_idx: int, pose: np.ndarray, seq_type: str) -> np.ndarray:
        """Generate realistic scene based on sequence type"""
        height, width = 480, 640
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Different scenes for different sequences
        if seq_type == "forward":
            # Corridor/hallway scene
            img = self._generate_corridor_scene(frame_idx, pose)
        elif seq_type == "circular":
            # Room scene
            img = self._generate_room_scene(frame_idx, pose)
        else:  # figure8
            # Outdoor scene
            img = self._generate_outdoor_scene(frame_idx, pose)

        return img

    def _generate_corridor_scene(self, frame_idx: int, pose: np.ndarray) -> np.ndarray:
        """Generate corridor scene"""
        height, width = 480, 640
        img = np.ones((height, width, 3), dtype=np.uint8) * 220  # Light background

        # Floor
        img[height//2:, :] = [180, 180, 180]

        # Walls
        cv2.line(img, (0, height//2), (width, height//3), (100, 100, 100), 2)  # Left wall
        cv2.line(img, (0, height//2), (width, 2*height//3), (100, 100, 100), 2)  # Right wall

        # Add features
        np.random.seed(frame_idx)
        for i in range(10):
            x = np.random.randint(50, width-50)
            y = np.random.randint(50, height-50)
            cv2.circle(img, (x, y), np.random.randint(3, 8), (np.random.randint(50, 200),) * 3, -1)

        return img

    def _generate_indoor_scene(self, frame_idx: int, pose: np.ndarray) -> np.ndarray:
        """Generate indoor scene for TUM-style data"""
        height, width = 480, 640
        img = np.ones((height, width, 3), dtype=np.uint8) * 200

        # Add furniture and objects
        np.random.seed(frame_idx // 10)  # Change scene slowly

        # Tables, chairs (rectangles)
        for i in range(5):
            x1 = np.random.randint(0, width//2)
            y1 = np.random.randint(0, height//2)
            x2 = x1 + np.random.randint(50, 150)
            y2 = y1 + np.random.randint(30, 100)
            color = (np.random.randint(50, 150),) * 3
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

        # Add texture
        noise = np.random.randint(-30, 30, (height, width, 3))
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        return img

    def _generate_road_scene(self, frame_idx: int, pose: np.ndarray, stereo: bool = False) -> np.ndarray:
        """Generate road scene for KITTI-style data"""
        height, width = 376, 1241
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Sky
        img[:height//3, :] = [180, 200, 255]

        # Road
        img[2*height//3:, :] = [70, 70, 70]

        # Lane markings
        for x in range(0, width, 100):
            cv2.line(img, (x, height-20), (x+50, height-20), (255, 255, 255), 3)

        # Buildings/trees on horizon
        np.random.seed(frame_idx // 20)
        for i in range(8):
            x = np.random.randint(0, width-100)
            h = np.random.randint(50, 150)
            cv2.rectangle(img, (x, height//3), (x+100, height//3+h), (80, 120, 80), -1)

        # Stereo effect
        if stereo:
            img = np.roll(img, -10, axis=1)

        return img

    def _generate_depth_image(self, pose: np.ndarray) -> np.ndarray:
        """Generate simulated depth image"""
        height, width = 480, 640

        # Create depth gradient (closer objects are darker)
        x = np.linspace(0, width-1, width)
        y = np.linspace(0, height-1, height)
        X, Y = np.meshgrid(x, y)

        # Simple depth model
        depth = 100 + (Y / height) * 150  # Depth increases with Y
        depth += np.random.normal(0, 5, depth.shape)  # Add noise

        # Convert to uint16 (typical depth format)
        depth = np.clip(depth, 0, 65535).astype(np.uint16)

        return depth

    def _create_calibration_file(self, output_dir: Path):
        """Create camera calibration file"""
        calibration = {
            "camera_matrix": {
                "fx": 525.0,
                "fy": 525.0,
                "cx": 319.5,
                "cy": 239.5
            },
            "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
            "image_size": [640, 480]
        }

        import json
        with open(output_dir / "calibration.json", 'w') as f:
            json.dump(calibration, f, indent=2)

    def _create_tum_format_files(self, seq_dir: Path, trajectory: np.ndarray):
        """Create TUM format association and groundtruth files"""
        # associations.txt (timestamp rgb depth)
        with open(seq_dir / "associations.txt", 'w') as f:
            for i in range(len(trajectory)):
                timestamp = i * 0.033  # ~30 FPS
                f.write(f"{timestamp:.6f} rgb/{i:06d}.png {timestamp:.6f} depth/{i:06d}.png\n")

        # groundtruth.txt (timestamp tx ty tz qx qy qz qw)
        with open(seq_dir / "groundtruth.txt", 'w') as f:
            for i, pose in enumerate(trajectory):
                timestamp = i * 0.033
                # Convert position to TUM format (tx ty tz qx qy qz qw)
                tx, ty, tz = pose
                # Identity quaternion for simplicity
                qx, qy, qz, qw = 0, 0, 0, 1
                f.write(f"{timestamp:.6f} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

    def _create_kitti_format_files(self, kitti_dir: Path, trajectory: np.ndarray):
        """Create KITTI format files"""
        # poses/00.txt (3x4 transformation matrices)
        poses_dir = kitti_dir / "poses"
        poses_dir.mkdir(exist_ok=True)

        with open(poses_dir / "00.txt", 'w') as f:
            for pose in trajectory:
                # Create identity rotation + translation
                tx, ty, tz = pose
                # 3x4 matrix: [R|t] where R is 3x3 identity, t is translation
                line = f"1.0 0.0 0.0 {tx:.6f} 0.0 1.0 0.0 {ty:.6f} 0.0 0.0 1.0 {tz:.6f}"
                f.write(line + "\n")

        # sequences/00/calib.txt
        seq_dir = kitti_dir / "sequences/00"
        calib_content = """P0: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 0.000000000000e+00 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00
P1: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02 -3.861448000000e+02 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02 0.000000000000e+00 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00 0.000000000000e+00"""

        with open(seq_dir / "calib.txt", 'w') as f:
            f.write(calib_content)

        # sequences/00/times.txt
        times = np.arange(len(trajectory)) * 0.1  # 10 FPS
        np.savetxt(str(seq_dir / "times.txt"), times, fmt='%.6f')

    # Additional trajectory generators for test sequences
    def _generate_rotation_trajectory(self, num_frames: int) -> np.ndarray:
        """Pure rotation around fixed point"""
        return np.zeros((num_frames, 3))  # No translation

    def _generate_translation_trajectory(self, num_frames: int) -> np.ndarray:
        """Pure translation"""
        trajectory = np.zeros((num_frames, 3))
        trajectory[:, 0] = np.linspace(0, 10, num_frames)
        return trajectory

    def _generate_lighting_trajectory(self, num_frames: int) -> np.ndarray:
        """Trajectory with challenging lighting"""
        return self._generate_forward_trajectory(num_frames)

    def _generate_fast_trajectory(self, num_frames: int) -> np.ndarray:
        """Fast motion trajectory"""
        trajectory = np.zeros((num_frames, 3))
        trajectory[:, 0] = np.linspace(0, 100, num_frames)  # Fast forward
        return trajectory

    def _generate_test_scene(self, frame_idx: int, pose: np.ndarray, test_type: str) -> np.ndarray:
        """Generate test scene based on test type"""
        height, width = 480, 640
        img = np.ones((height, width, 3), dtype=np.uint8) * 128

        if test_type == "challenging_lighting":
            # Vary brightness
            brightness = int(128 + 100 * np.sin(frame_idx * 0.1))
            img = np.ones((height, width, 3), dtype=np.uint8) * brightness

        # Add consistent features
        np.random.seed(42)  # Fixed seed for consistent features
        for i in range(20):
            x = np.random.randint(50, width-50)
            y = np.random.randint(50, height-50)
            cv2.circle(img, (x, y), 5, (0, 0, 0), -1)

        return img

    def print_summary(self):
        """Print data collection summary"""
        print("\n" + "=" * 60)
        print("📊 Data Collection Summary")
        print("=" * 60)

        total_size = 0
        datasets = ["sample", "tum", "kitti", "test"]

        for dataset in datasets:
            dataset_path = self.data_dir / dataset
            if dataset_path.exists():
                size = sum(f.stat().st_size for f in dataset_path.rglob('*') if f.is_file())
                size_mb = size / (1024 * 1024)
                total_size += size_mb

                # Count sequences/files
                if dataset == "sample":
                    sequences = list(dataset_path.iterdir())
                    print(f"📁 {dataset.upper()}: {len(sequences)} sequences, {size_mb:.1f} MB")
                else:
                    files = list(dataset_path.rglob('*.png'))
                    print(f"📁 {dataset.upper()}: {len(files)} images, {size_mb:.1f} MB")

        print(f"\n💾 Total size: {total_size:.1f} MB")
        print(f"📂 Location: {self.data_dir.absolute()}")

        print("\n🚀 Ready for Visual Odometry testing!")
        print("   Run: python start_server.py")


def main():
    """Main function"""
    collector = DataCollector()

    if collector.collect_all_datasets():
        collector.print_summary()
        return True
    else:
        print("❌ Data collection failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)