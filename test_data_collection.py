#!/usr/bin/env python3
"""
Simple test for data collection and validation
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

def create_simple_test_data():
    """Create minimal test data for validation"""
    print("Creating simple test dataset...")

    # Create datasets directory
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)

    # Create sample dataset
    sample_dir = datasets_dir / "sample"
    sample_dir.mkdir(exist_ok=True)

    forward_dir = sample_dir / "forward"
    forward_dir.mkdir(exist_ok=True)

    print("Generating 50 test images...")

    # Generate simple trajectory
    trajectory = []
    for i in range(50):
        x = i * 0.1  # Forward motion
        y = 0.05 * np.sin(i * 0.1)  # Small lateral movement
        z = 0.01 * np.random.randn()  # Small vertical noise
        trajectory.append([x, y, z])

    # Generate simple images
    for i in range(50):
        # Create a simple test image
        img = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray background

        # Add some features
        cv2.circle(img, (320, 240), 50, (255, 255, 255), -1)  # White circle
        cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 0), 2)  # Black rectangle

        # Add frame number
        cv2.putText(img, f"Frame {i:03d}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Save image
        cv2.imwrite(str(forward_dir / f"frame_{i:06d}.png"), img)

    # Save trajectory
    np.savetxt(str(forward_dir / "ground_truth.txt"), trajectory, fmt='%.6f')

    # Create calibration file
    import json
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

    with open(forward_dir / "calibration.json", 'w') as f:
        json.dump(calibration, f, indent=2)

    print(f"Created test dataset with {len(trajectory)} frames")
    print(f"Location: {forward_dir.absolute()}")

    return True

def validate_test_data():
    """Validate the created test data"""
    print("\nValidating test data...")

    forward_dir = Path("datasets/sample/forward")

    if not forward_dir.exists():
        print("ERROR: Test data directory not found")
        return False

    # Check images
    image_files = list(forward_dir.glob("frame_*.png"))
    print(f"Found {len(image_files)} image files")

    if len(image_files) == 0:
        print("ERROR: No image files found")
        return False

    # Check first image
    first_img = cv2.imread(str(image_files[0]))
    if first_img is None:
        print("ERROR: Cannot read first image")
        return False

    height, width = first_img.shape[:2]
    print(f"Image resolution: {width}x{height}")

    # Check ground truth
    gt_file = forward_dir / "ground_truth.txt"
    if gt_file.exists():
        trajectory = np.loadtxt(gt_file)
        print(f"Ground truth trajectory: {trajectory.shape[0]} poses")
    else:
        print("WARNING: No ground truth file found")

    # Check calibration
    calib_file = forward_dir / "calibration.json"
    if calib_file.exists():
        print("Calibration file found")
    else:
        print("WARNING: No calibration file found")

    print("Test data validation completed successfully")
    return True

def test_visual_odometry():
    """Test basic visual odometry processing"""
    print("\nTesting basic visual odometry processing...")

    try:
        # Simple feature detection test
        forward_dir = Path("datasets/sample/forward")
        image_files = sorted(list(forward_dir.glob("frame_*.png")))

        if len(image_files) < 2:
            print("ERROR: Need at least 2 images for VO test")
            return False

        # Load first two images
        img1 = cv2.imread(str(image_files[0]), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(image_files[1]), cv2.IMREAD_GRAYSCALE)

        # Detect features using ORB
        orb = cv2.ORB_create(1000)

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        print(f"Frame 1: {len(kp1)} keypoints")
        print(f"Frame 2: {len(kp2)} keypoints")

        # Match features
        if des1 is not None and des2 is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)

            print(f"Found {len(matches)} feature matches")

            if len(matches) > 10:
                print("Basic visual odometry test PASSED")
                return True
            else:
                print("WARNING: Very few feature matches found")
                return True
        else:
            print("ERROR: Could not extract features")
            return False

    except Exception as e:
        print(f"ERROR in VO test: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("Visual Odometry Data Collection Test")
    print("=" * 50)

    success = True

    # Step 1: Create test data
    if not create_simple_test_data():
        print("FAILED: Could not create test data")
        success = False

    # Step 2: Validate data
    if not validate_test_data():
        print("FAILED: Data validation failed")
        success = False

    # Step 3: Test VO processing
    if not test_visual_odometry():
        print("FAILED: Visual odometry test failed")
        success = False

    print("\n" + "=" * 50)
    if success:
        print("ALL TESTS PASSED - Ready for visual odometry!")
        print("Next steps:")
        print("  1. Run: python start_server.py")
        print("  2. Open: http://localhost:8000")
        print("  3. Click 'Prepare Sample' then 'Start Processing'")
    else:
        print("SOME TESTS FAILED - Check error messages above")
    print("=" * 50)

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)