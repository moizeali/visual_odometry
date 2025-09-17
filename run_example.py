#!/usr/bin/env python3
"""
Example script to run visual odometry processing without the web interface
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add backend to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from core.visual_odometry import MonocularVO, CameraParams
from data.dataset_loader import DatasetManager
from utils.metrics import MetricsCollector
from utils.visualization import TrajectoryPlotter

def run_example():
    """Run a complete visual odometry example"""
    print("Visual Odometry Example")
    print("=" * 40)

    # Initialize components
    dataset_manager = DatasetManager()
    metrics_collector = MetricsCollector()
    plotter = TrajectoryPlotter()

    # Camera parameters (typical values)
    camera_params = CameraParams(
        fx=718.856,
        fy=718.856,
        cx=607.1928,
        cy=185.2157,
        baseline=0.54
    )

    # Initialize VO system
    vo = MonocularVO(camera_params, detector_type='ORB')
    print("Visual Odometry system initialized successfully")

    # Load or generate sample data
    try:
        print("Loading sample dataset...")
        dataset_path = dataset_manager.prepare_sample_data()

        # Use existing test data
        sample_dir = Path("datasets/sample/forward")
        if sample_dir.exists():
            image_files = sorted(list(sample_dir.glob("frame_*.png")))
            dataset = {
                'frames': [str(f) for f in image_files],
                'name': 'sample',
                'calibration': {
                    'fx': 525.0, 'fy': 525.0, 'cx': 319.5, 'cy': 239.5
                }
            }
        else:
            raise FileNotFoundError("Sample dataset not found. Run python test_data_collection.py first.")

        if dataset['frames']:
            print(f"Loaded {len(dataset['frames'])} frames")

            # Start metrics collection
            session_id = metrics_collector.start_session(
                dataset="sample",
                sequence="test",
                algorithm="ORB_MonoVO"
            )
            print(f"Started metrics session: {session_id}")

            # Process frames
            trajectory = []
            for i, frame_path in enumerate(dataset['frames'][:10]):  # Process first 10 frames
                print(f"Processing frame {i+1}/10", end="")

                # Load image
                frame = cv2.imread(frame_path)
                if frame is None:
                    print(" [FAILED - Cannot load image]")
                    continue

                try:
                    pose, metrics = vo.process_frame(frame)
                    if pose is not None:
                        trajectory.append(pose.t.flatten())

                        # Record metrics
                        metrics_collector.record_frame(
                            frame_number=i,
                            processing_time=metrics.get('processing_time', 0.01),
                            keypoints=metrics.get('keypoints_count', 500),
                            matches=metrics.get('matches_count', 300),
                            inliers=metrics.get('inliers_count', 200),
                            position=pose.t.flatten().tolist(),
                            rotation=pose.R.tolist()
                        )
                        print(" [SUCCESS]")
                    else:
                        print(" [FAILED - No pose estimated]")
                except Exception as e:
                    print(f" [ERROR - {str(e)}]")

            # End session and get summary
            completed_session = metrics_collector.end_session(success=True)
            summary = metrics_collector.get_session_summary()

            print("\nProcessing Summary:")
            print(f"   Total frames: {summary['total_frames']}")
            print(f"   Average FPS: {summary['fps']:.2f}")
            print(f"   Trajectory length: {summary['trajectory_length']:.2f}m")

            # Generate visualizations
            if len(trajectory) > 1:
                print("\nGenerating trajectory plot...")
                trajectory_array = np.array(trajectory)

                # Save 2D plot
                plot_2d = plotter.plot_trajectory_2d(trajectory_array, title="Example Trajectory (2D)")
                print("2D trajectory plot generated successfully")

                # Save 3D plot
                plot_3d = plotter.plot_trajectory_3d(trajectory_array, title="Example Trajectory (3D)")
                print("3D trajectory plot generated successfully")

                print("\nExample completed successfully!")
                print("To see interactive visualizations, run: python start_server.py")
            else:
                print("WARNING: Not enough trajectory points for visualization")

        else:
            print("WARNING: No sample frames available")

    except Exception as e:
        print(f"ERROR: Error during processing: {e}")
        return False

    return True

if __name__ == "__main__":
    success = run_example()
    if success:
        print("\nReady to start the web server!")
        print("   Run: python start_server.py")
    else:
        print("\nExample failed. Check error messages above.")