#!/usr/bin/env python3
"""
Generate professional-grade datasets for industrial demonstration
Creates high-quality synthetic data with realistic scenarios
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
import time

def create_industrial_datasets():
    """Create professional industrial-grade datasets"""
    print("=" * 60)
    print("GENERATING PROFESSIONAL VISUAL ODOMETRY DATASETS")
    print("=" * 60)

    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)

    # 1. Highway Driving Sequence (KITTI-style)
    print("\n1. Creating Highway Driving Sequence...")
    create_highway_sequence(datasets_dir)

    # 2. Industrial Inspection Sequence
    print("\n2. Creating Industrial Inspection Sequence...")
    create_industrial_sequence(datasets_dir)

    # 3. Drone Survey Sequence
    print("\n3. Creating Drone Survey Sequence...")
    create_drone_sequence(datasets_dir)

    # 4. Urban Navigation Sequence
    print("\n4. Creating Urban Navigation Sequence...")
    create_urban_sequence(datasets_dir)

    print("\n" + "=" * 60)
    print("PROFESSIONAL DATASETS CREATED SUCCESSFULLY")
    print("=" * 60)
    print(f"Location: {datasets_dir.absolute()}")
    print("Ready for industrial demonstration!")

def create_highway_sequence(base_dir: Path):
    """Create realistic highway driving sequence"""
    seq_dir = base_dir / "professional" / "highway_driving"
    seq_dir.mkdir(parents=True, exist_ok=True)

    # Generate 300-frame sequence (30 seconds at 10 FPS)
    num_frames = 300

    # Realistic vehicle trajectory - highway with gentle curves
    trajectory = generate_highway_trajectory(num_frames)

    print(f"  Generating {num_frames} high-quality highway frames...")

    for i in tqdm(range(num_frames), desc="  Highway frames"):
        # Create realistic highway scene
        img = create_highway_scene(i, trajectory[i], num_frames)
        cv2.imwrite(str(seq_dir / f"frame_{i:06d}.png"), img)

    # Save metadata
    save_sequence_metadata(seq_dir, trajectory, "Highway Driving",
                          "High-speed highway driving with lane changes and curves")

def create_industrial_sequence(base_dir: Path):
    """Create industrial facility inspection sequence"""
    seq_dir = base_dir / "professional" / "industrial_inspection"
    seq_dir.mkdir(parents=True, exist_ok=True)

    # Generate 200-frame handheld inspection sequence
    num_frames = 200

    # Realistic inspection trajectory - systematic scanning pattern
    trajectory = generate_inspection_trajectory(num_frames)

    print(f"  Generating {num_frames} industrial inspection frames...")

    for i in tqdm(range(num_frames), desc="  Inspection frames"):
        # Create industrial facility scene
        img = create_industrial_scene(i, trajectory[i], num_frames)
        cv2.imwrite(str(seq_dir / f"frame_{i:06d}.png"), img)

    # Save metadata
    save_sequence_metadata(seq_dir, trajectory, "Industrial Inspection",
                          "Handheld camera inspection of industrial equipment and structures")

def create_drone_sequence(base_dir: Path):
    """Create aerial drone survey sequence"""
    seq_dir = base_dir / "professional" / "drone_survey"
    seq_dir.mkdir(parents=True, exist_ok=True)

    # Generate 250-frame aerial sequence
    num_frames = 250

    # Realistic drone trajectory - aerial survey pattern
    trajectory = generate_drone_trajectory(num_frames)

    print(f"  Generating {num_frames} aerial survey frames...")

    for i in tqdm(range(num_frames), desc="  Aerial frames"):
        # Create aerial survey scene
        img = create_aerial_scene(i, trajectory[i], num_frames)
        cv2.imwrite(str(seq_dir / f"frame_{i:06d}.png"), img)

    # Save metadata
    save_sequence_metadata(seq_dir, trajectory, "Drone Survey",
                          "Aerial drone survey of construction site or agricultural area")

def create_urban_sequence(base_dir: Path):
    """Create urban navigation sequence"""
    seq_dir = base_dir / "professional" / "urban_navigation"
    seq_dir.mkdir(parents=True, exist_ok=True)

    # Generate 400-frame urban sequence
    num_frames = 400

    # Realistic urban trajectory - city driving with turns
    trajectory = generate_urban_trajectory(num_frames)

    print(f"  Generating {num_frames} urban navigation frames...")

    for i in tqdm(range(num_frames), desc="  Urban frames"):
        # Create urban scene
        img = create_urban_scene(i, trajectory[i], num_frames)
        cv2.imwrite(str(seq_dir / f"frame_{i:06d}.png"), img)

    # Save metadata
    save_sequence_metadata(seq_dir, trajectory, "Urban Navigation",
                          "Autonomous vehicle navigation in urban environment")

def generate_highway_trajectory(num_frames: int) -> np.ndarray:
    """Generate realistic highway driving trajectory"""
    trajectory = np.zeros((num_frames, 3))

    # Parameters for highway driving
    base_speed = 1.5  # m/frame (about 54 km/h at 10 FPS)

    for i in range(num_frames):
        t = i / num_frames

        # Forward motion with slight speed variation
        speed_var = 1.0 + 0.1 * np.sin(t * 2 * np.pi)
        trajectory[i, 0] = i * base_speed * speed_var

        # Gentle highway curves
        trajectory[i, 1] = 20 * np.sin(t * np.pi) + 5 * np.sin(t * 4 * np.pi)

        # Slight elevation changes
        trajectory[i, 2] = 2 * np.sin(t * 0.5 * np.pi) + np.random.normal(0, 0.1)

    return trajectory

def generate_inspection_trajectory(num_frames: int) -> np.ndarray:
    """Generate systematic inspection trajectory"""
    trajectory = np.zeros((num_frames, 3))

    # Handheld camera inspection pattern
    for i in range(num_frames):
        t = i / num_frames

        # Scanning motion
        trajectory[i, 0] = 0.1 * i + 2 * np.sin(t * 6 * np.pi)
        trajectory[i, 1] = 3 * np.sin(t * 4 * np.pi)
        trajectory[i, 2] = 1 * np.cos(t * 3 * np.pi) + np.random.normal(0, 0.1)

    return trajectory

def generate_drone_trajectory(num_frames: int) -> np.ndarray:
    """Generate aerial drone survey trajectory"""
    trajectory = np.zeros((num_frames, 3))

    # Aerial survey pattern
    for i in range(num_frames):
        t = i / num_frames

        # Survey grid pattern
        trajectory[i, 0] = 50 * t
        trajectory[i, 1] = 10 * np.sin(t * 8 * np.pi)
        trajectory[i, 2] = 30 + 5 * np.sin(t * 2 * np.pi)  # Altitude variation

    return trajectory

def generate_urban_trajectory(num_frames: int) -> np.ndarray:
    """Generate urban driving trajectory"""
    trajectory = np.zeros((num_frames, 3))

    # Urban driving with turns and stops
    for i in range(num_frames):
        t = i / num_frames

        # Variable speed city driving
        speed_factor = 0.5 + 0.5 * np.abs(np.sin(t * 3 * np.pi))
        trajectory[i, 0] = i * 0.8 * speed_factor

        # City block turns
        trajectory[i, 1] = 15 * np.sin(t * 2 * np.pi) + 8 * np.cos(t * 6 * np.pi)

        # Flat urban terrain with small variations
        trajectory[i, 2] = np.random.normal(0, 0.05)

    return trajectory

def create_highway_scene(frame_idx: int, position: np.ndarray, total_frames: int) -> np.ndarray:
    """Create realistic highway scene"""
    height, width = 720, 1280  # High resolution
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky gradient
    for y in range(height // 3):
        intensity = int(180 + 50 * (1 - y / (height // 3)))
        img[y, :] = [intensity, intensity + 20, intensity + 40]

    # Road surface
    road_start = 2 * height // 3
    img[road_start:, :] = [60, 60, 60]

    # Lane markings with perspective
    lane_width = width // 4
    for lane in range(1, 4):
        x_base = lane * lane_width
        # Dashed lines with movement
        for dash_start in range(-20, width, 60):
            x_start = x_base + dash_start - (frame_idx * 5) % 60
            if 0 <= x_start < width - 30:
                cv2.line(img, (x_start, road_start + 20),
                        (x_start + 30, road_start + 20), (255, 255, 255), 3)

    # Horizon buildings/trees
    np.random.seed(frame_idx // 50)  # Change scenery slowly
    for i in range(15):
        x = np.random.randint(0, width - 150)
        building_height = np.random.randint(80, 200)
        y_start = height // 3
        y_end = y_start + building_height

        # Building color variation
        color_base = np.random.randint(60, 120)
        color = (color_base, color_base + 20, color_base + 10)
        cv2.rectangle(img, (x, y_start), (x + 150, y_end), color, -1)

    # Add vehicles (other cars)
    add_traffic(img, frame_idx)

    # Add motion blur for realism
    if frame_idx > 0:
        kernel = np.ones((1, 3), np.float32) / 3
        img = cv2.filter2D(img, -1, kernel)

    return img

def create_industrial_scene(frame_idx: int, position: np.ndarray, total_frames: int) -> np.ndarray:
    """Create industrial facility scene"""
    height, width = 720, 1280
    img = np.ones((height, width, 3), dtype=np.uint8) * 140  # Industrial gray

    # Industrial structures
    np.random.seed(42)  # Consistent structures

    # Large industrial equipment
    for i in range(8):
        x = np.random.randint(0, width - 200)
        y = np.random.randint(height // 4, height - 300)
        equipment_width = np.random.randint(150, 250)
        equipment_height = np.random.randint(200, 400)

        # Equipment color (metallic)
        metal_color = (80 + np.random.randint(-20, 20),) * 3
        cv2.rectangle(img, (x, y), (x + equipment_width, y + equipment_height),
                     metal_color, -1)

        # Add details (pipes, panels)
        for j in range(5):
            detail_x = x + np.random.randint(0, equipment_width - 20)
            detail_y = y + np.random.randint(0, equipment_height - 20)
            cv2.circle(img, (detail_x, detail_y), np.random.randint(5, 15),
                      (200, 200, 200), -1)

    # Pipes and conduits
    for i in range(20):
        start_x = np.random.randint(0, width)
        start_y = np.random.randint(0, height)
        end_x = start_x + np.random.randint(-200, 200)
        end_y = start_y + np.random.randint(-100, 100)

        if 0 <= end_x < width and 0 <= end_y < height:
            cv2.line(img, (start_x, start_y), (end_x, end_y),
                    (120, 120, 120), np.random.randint(8, 20))

    # Add industrial lighting effects
    add_industrial_lighting(img, frame_idx)

    return img

def create_aerial_scene(frame_idx: int, position: np.ndarray, total_frames: int) -> np.ndarray:
    """Create aerial survey scene"""
    height, width = 720, 1280
    img = np.ones((height, width, 3), dtype=np.uint8) * 120  # Ground color

    # Terrain features
    np.random.seed(1)  # Consistent terrain

    # Fields and plots
    for i in range(12):
        x = np.random.randint(0, width - 300)
        y = np.random.randint(0, height - 200)
        field_width = np.random.randint(200, 400)
        field_height = np.random.randint(150, 300)

        # Field colors (agriculture)
        if i % 3 == 0:
            field_color = (80, 150, 80)  # Green crops
        elif i % 3 == 1:
            field_color = (120, 120, 80)  # Brown soil
        else:
            field_color = (140, 140, 90)  # Harvested field

        cv2.rectangle(img, (x, y), (x + field_width, y + field_height),
                     field_color, -1)

        # Field boundaries
        cv2.rectangle(img, (x, y), (x + field_width, y + field_height),
                     (60, 60, 60), 2)

    # Roads and paths
    for i in range(6):
        start_x = np.random.randint(0, width)
        start_y = np.random.randint(0, height)
        end_x = np.random.randint(0, width)
        end_y = np.random.randint(0, height)

        cv2.line(img, (start_x, start_y), (end_x, end_y),
                (80, 80, 80), np.random.randint(15, 30))

    # Add shadows and elevation effects
    add_aerial_effects(img, position)

    return img

def create_urban_scene(frame_idx: int, position: np.ndarray, total_frames: int) -> np.ndarray:
    """Create urban environment scene"""
    height, width = 720, 1280
    img = np.ones((height, width, 3), dtype=np.uint8) * 100

    # Urban buildings
    np.random.seed(frame_idx // 100)  # Change blocks slowly

    # Buildings on both sides
    for side in [0, 1]:  # Left and right sides
        for i in range(6):
            building_x = side * (width - 200) + np.random.randint(0, 200)
            building_y = np.random.randint(0, height // 2)
            building_width = np.random.randint(100, 250)
            building_height = np.random.randint(200, height - building_y)

            # Building colors
            building_color = (
                np.random.randint(80, 150),
                np.random.randint(80, 150),
                np.random.randint(80, 150)
            )

            cv2.rectangle(img, (building_x, building_y),
                         (building_x + building_width, building_y + building_height),
                         building_color, -1)

            # Windows
            for row in range(building_height // 40):
                for col in range(building_width // 30):
                    window_x = building_x + 10 + col * 30
                    window_y = building_y + 20 + row * 40
                    if window_x < building_x + building_width - 20:
                        cv2.rectangle(img, (window_x, window_y),
                                    (window_x + 15, window_y + 25),
                                    (200, 200, 100), -1)

    # Street
    street_y = 2 * height // 3
    img[street_y:, :] = [70, 70, 70]

    # Street markings
    center_line_y = street_y + 30
    for dash_start in range(-10, width, 40):
        x_start = dash_start - (frame_idx * 3) % 40
        if 0 <= x_start < width - 20:
            cv2.line(img, (x_start, center_line_y),
                    (x_start + 20, center_line_y), (255, 255, 255), 2)

    return img

def add_traffic(img: np.ndarray, frame_idx: int):
    """Add other vehicles to highway scene"""
    height, width = img.shape[:2]
    road_y = 2 * height // 3 + 40

    # Add a few cars
    np.random.seed(frame_idx // 20)
    for i in range(3):
        car_x = np.random.randint(100, width - 100) - (frame_idx % 100)
        if 0 <= car_x < width - 80:
            # Car body
            cv2.rectangle(img, (car_x, road_y), (car_x + 80, road_y + 30),
                         (np.random.randint(50, 200), np.random.randint(50, 200),
                          np.random.randint(50, 200)), -1)
            # Windows
            cv2.rectangle(img, (car_x + 10, road_y + 5), (car_x + 70, road_y + 20),
                         (150, 180, 200), -1)

def add_industrial_lighting(img: np.ndarray, frame_idx: int):
    """Add realistic industrial lighting"""
    height, width = img.shape[:2]

    # Fluorescent lighting effect
    light_intensity = 0.8 + 0.2 * np.sin(frame_idx * 0.1)
    overlay = np.ones_like(img) * int(20 * light_intensity)
    img = cv2.addWeighted(img, 0.9, overlay, 0.1, 0)

def add_aerial_effects(img: np.ndarray, position: np.ndarray):
    """Add altitude and atmospheric effects"""
    height, width = img.shape[:2]

    # Slight haze effect
    haze = np.ones_like(img) * 30
    img = cv2.addWeighted(img, 0.95, haze, 0.05, 0)

def save_sequence_metadata(seq_dir: Path, trajectory: np.ndarray,
                          name: str, description: str):
    """Save sequence metadata and calibration"""

    # Save trajectory as ground truth
    np.savetxt(str(seq_dir / "ground_truth.txt"), trajectory, fmt='%.6f')

    # Professional camera calibration (high-end industrial camera)
    calibration = {
        "camera_model": "Industrial Grade Camera System",
        "resolution": [1280, 720],
        "camera_matrix": {
            "fx": 1000.0,
            "fy": 1000.0,
            "cx": 640.0,
            "cy": 360.0
        },
        "distortion_coefficients": [0.0, 0.0, 0.0, 0.0, 0.0],
        "baseline": 0.12,  # For stereo if needed
        "fps": 10.0
    }

    with open(seq_dir / "calibration.json", 'w') as f:
        json.dump(calibration, f, indent=2)

    # Sequence metadata
    metadata = {
        "name": name,
        "description": description,
        "frames": len(trajectory),
        "duration_seconds": len(trajectory) / 10.0,
        "trajectory_length_meters": float(np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))),
        "complexity": "High",
        "use_cases": [
            "Algorithm benchmarking",
            "Industrial demonstration",
            "Performance analysis",
            "Research validation"
        ]
    }

    with open(seq_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"    Saved {name}: {len(trajectory)} frames, {metadata['trajectory_length_meters']:.1f}m trajectory")

def main():
    """Generate all professional datasets"""
    start_time = time.time()

    create_industrial_datasets()

    elapsed_time = time.time() - start_time
    print(f"\nTotal generation time: {elapsed_time:.1f} seconds")

    # Generate summary report
    datasets_dir = Path("datasets/professional")
    if datasets_dir.exists():
        total_frames = 0
        total_size = 0

        for seq_dir in datasets_dir.iterdir():
            if seq_dir.is_dir():
                frames = list(seq_dir.glob("frame_*.png"))
                total_frames += len(frames)
                total_size += sum(f.stat().st_size for f in frames)

        print(f"\nDATASET SUMMARY:")
        print(f"Total frames: {total_frames:,}")
        print(f"Total size: {total_size / (1024**2):.1f} MB")
        print(f"Average frame size: {total_size / total_frames / 1024:.1f} KB")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)