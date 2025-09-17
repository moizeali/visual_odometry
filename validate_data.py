#!/usr/bin/env python3
"""
Data validation script for Visual Odometry Enhanced System
Validates collected datasets and checks data quality
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# Add backend to Python path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

class DataValidator:
    """Validate collected datasets for visual odometry"""

    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = Path(data_dir)
        self.validation_results = {}

    def validate_all_datasets(self) -> Dict:
        """Validate all available datasets"""
        print("🔍 Visual Odometry Data Validation")
        print("=" * 50)

        results = {
            "overall_status": "success",
            "datasets": {},
            "summary": {}
        }

        # Check each dataset type
        dataset_types = ["sample", "kitti", "tum", "test"]

        for dataset_type in dataset_types:
            dataset_path = self.data_dir / dataset_type

            if dataset_path.exists():
                print(f"\n📂 Validating {dataset_type.upper()} dataset...")
                validation_result = self._validate_dataset(dataset_type, dataset_path)
                results["datasets"][dataset_type] = validation_result

                if not validation_result["valid"]:
                    results["overall_status"] = "warning"
            else:
                print(f"⚠️  {dataset_type.upper()} dataset not found")
                results["datasets"][dataset_type] = {
                    "valid": False,
                    "error": "Dataset directory not found"
                }

        # Generate summary
        results["summary"] = self._generate_summary(results["datasets"])

        self.validation_results = results
        return results

    def _validate_dataset(self, dataset_type: str, dataset_path: Path) -> Dict:
        """Validate a specific dataset"""
        result = {
            "valid": True,
            "sequences": {},
            "total_images": 0,
            "total_size_mb": 0,
            "issues": []
        }

        try:
            if dataset_type == "sample":
                result = self._validate_sample_dataset(dataset_path)
            elif dataset_type == "kitti":
                result = self._validate_kitti_dataset(dataset_path)
            elif dataset_type == "tum":
                result = self._validate_tum_dataset(dataset_path)
            elif dataset_type == "test":
                result = self._validate_test_dataset(dataset_path)

            # Calculate total size
            total_size = sum(f.stat().st_size for f in dataset_path.rglob('*') if f.is_file())
            result["total_size_mb"] = total_size / (1024 * 1024)

        except Exception as e:
            result["valid"] = False
            result["error"] = str(e)

        return result

    def _validate_sample_dataset(self, dataset_path: Path) -> Dict:
        """Validate sample dataset"""
        result = {
            "valid": True,
            "sequences": {},
            "total_images": 0,
            "issues": []
        }

        expected_sequences = ["forward", "circular", "figure8"]

        for seq_name in expected_sequences:
            seq_path = dataset_path / seq_name
            if seq_path.exists():
                seq_result = self._validate_sequence(seq_path, expected_format="sample")
                result["sequences"][seq_name] = seq_result
                result["total_images"] += seq_result.get("num_images", 0)

                if not seq_result["valid"]:
                    result["issues"].extend(seq_result.get("issues", []))
                    result["valid"] = False
            else:
                result["issues"].append(f"Missing sequence: {seq_name}")
                result["valid"] = False

        return result

    def _validate_kitti_dataset(self, dataset_path: Path) -> Dict:
        """Validate KITTI dataset"""
        result = {
            "valid": True,
            "sequences": {},
            "total_images": 0,
            "issues": []
        }

        # Check sequence 00
        seq_path = dataset_path / "sequences/00"
        if seq_path.exists():
            seq_result = self._validate_sequence(seq_path, expected_format="kitti")
            result["sequences"]["00"] = seq_result
            result["total_images"] += seq_result.get("num_images", 0)

            if not seq_result["valid"]:
                result["issues"].extend(seq_result.get("issues", []))
                result["valid"] = False

            # Validate KITTI-specific files
            self._validate_kitti_files(dataset_path, result)
        else:
            result["issues"].append("Missing KITTI sequence 00")
            result["valid"] = False

        return result

    def _validate_tum_dataset(self, dataset_path: Path) -> Dict:
        """Validate TUM RGB-D dataset"""
        result = {
            "valid": True,
            "sequences": {},
            "total_images": 0,
            "issues": []
        }

        # Check freiburg1_xyz sequence
        seq_path = dataset_path / "freiburg1_xyz"
        if seq_path.exists():
            seq_result = self._validate_sequence(seq_path, expected_format="tum")
            result["sequences"]["freiburg1_xyz"] = seq_result
            result["total_images"] += seq_result.get("num_images", 0)

            if not seq_result["valid"]:
                result["issues"].extend(seq_result.get("issues", []))
                result["valid"] = False

            # Validate TUM-specific files
            self._validate_tum_files(seq_path, result)
        else:
            result["issues"].append("Missing TUM sequence freiburg1_xyz")
            result["valid"] = False

        return result

    def _validate_test_dataset(self, dataset_path: Path) -> Dict:
        """Validate test dataset"""
        result = {
            "valid": True,
            "sequences": {},
            "total_images": 0,
            "issues": []
        }

        expected_tests = ["pure_rotation", "pure_translation", "challenging_lighting", "fast_motion"]

        for test_name in expected_tests:
            test_path = dataset_path / test_name
            if test_path.exists():
                test_result = self._validate_sequence(test_path, expected_format="test")
                result["sequences"][test_name] = test_result
                result["total_images"] += test_result.get("num_images", 0)

                if not test_result["valid"]:
                    result["issues"].extend(test_result.get("issues", []))
                    result["valid"] = False
            else:
                result["issues"].append(f"Missing test sequence: {test_name}")

        return result

    def _validate_sequence(self, seq_path: Path, expected_format: str) -> Dict:
        """Validate a single sequence"""
        result = {
            "valid": True,
            "num_images": 0,
            "image_resolution": None,
            "has_calibration": False,
            "has_ground_truth": False,
            "issues": []
        }

        try:
            # Count images
            if expected_format == "kitti":
                image_files = list((seq_path / "image_0").glob("*.png"))
                if not image_files:
                    result["issues"].append("No images found in image_0 directory")
                    result["valid"] = False
            elif expected_format == "tum":
                image_files = list((seq_path / "rgb").glob("*.png"))
                if not image_files:
                    result["issues"].append("No RGB images found")
                    result["valid"] = False
            else:
                image_files = list(seq_path.glob("frame_*.png"))
                if not image_files:
                    result["issues"].append("No frame images found")
                    result["valid"] = False

            result["num_images"] = len(image_files)

            # Check image quality
            if image_files:
                sample_img_path = image_files[0]
                img = cv2.imread(str(sample_img_path))
                if img is not None:
                    result["image_resolution"] = f"{img.shape[1]}x{img.shape[0]}"

                    # Check image quality
                    self._check_image_quality(img, result)
                else:
                    result["issues"].append("Cannot read sample image")
                    result["valid"] = False

            # Check for calibration file
            calib_files = ["calibration.json", "calib.txt"]
            for calib_file in calib_files:
                if (seq_path / calib_file).exists():
                    result["has_calibration"] = True
                    break

            # Check for ground truth
            gt_files = ["ground_truth.txt", "groundtruth.txt"]
            for gt_file in gt_files:
                if (seq_path / gt_file).exists():
                    result["has_ground_truth"] = True
                    break

            # Minimum requirements
            if result["num_images"] < 10:
                result["issues"].append(f"Too few images: {result['num_images']} (minimum 10)")
                result["valid"] = False

        except Exception as e:
            result["issues"].append(f"Validation error: {str(e)}")
            result["valid"] = False

        return result

    def _check_image_quality(self, img: np.ndarray, result: Dict):
        """Check image quality metrics"""
        # Check if image is too dark or too bright
        mean_brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

        if mean_brightness < 30:
            result["issues"].append("Images too dark (poor visibility)")
        elif mean_brightness > 225:
            result["issues"].append("Images too bright (overexposed)")

        # Check for motion blur (using Laplacian variance)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur_measure = cv2.Laplacian(gray, cv2.CV_64F).var()

        if blur_measure < 100:
            result["issues"].append("Images may be blurry")

        # Check image size
        height, width = img.shape[:2]
        if width < 320 or height < 240:
            result["issues"].append("Image resolution too low")

    def _validate_kitti_files(self, dataset_path: Path, result: Dict):
        """Validate KITTI-specific files"""
        # Check poses file
        poses_file = dataset_path / "poses/00.txt"
        if poses_file.exists():
            try:
                poses = np.loadtxt(poses_file)
                if poses.shape[1] != 12:
                    result["issues"].append("Invalid KITTI poses format")
                    result["valid"] = False
            except:
                result["issues"].append("Cannot read KITTI poses file")
                result["valid"] = False
        else:
            result["issues"].append("Missing KITTI poses file")

        # Check calibration
        calib_file = dataset_path / "sequences/00/calib.txt"
        if not calib_file.exists():
            result["issues"].append("Missing KITTI calibration file")

    def _validate_tum_files(self, seq_path: Path, result: Dict):
        """Validate TUM-specific files"""
        # Check associations file
        assoc_file = seq_path / "associations.txt"
        if not assoc_file.exists():
            result["issues"].append("Missing TUM associations file")

        # Check depth images
        depth_dir = seq_path / "depth"
        if depth_dir.exists():
            depth_files = list(depth_dir.glob("*.png"))
            rgb_files = list((seq_path / "rgb").glob("*.png"))

            if len(depth_files) != len(rgb_files):
                result["issues"].append("Mismatch between RGB and depth image counts")
        else:
            result["issues"].append("Missing TUM depth directory")

    def _generate_summary(self, datasets: Dict) -> Dict:
        """Generate validation summary"""
        summary = {
            "total_datasets": len(datasets),
            "valid_datasets": 0,
            "total_images": 0,
            "total_size_mb": 0,
            "common_issues": []
        }

        all_issues = []

        for dataset_name, dataset_result in datasets.items():
            if dataset_result.get("valid", False):
                summary["valid_datasets"] += 1

            summary["total_images"] += dataset_result.get("total_images", 0)
            summary["total_size_mb"] += dataset_result.get("total_size_mb", 0)

            # Collect issues
            all_issues.extend(dataset_result.get("issues", []))

        # Find common issues
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

        # Issues that appear in multiple datasets
        summary["common_issues"] = [issue for issue, count in issue_counts.items() if count > 1]

        return summary

    def print_validation_report(self):
        """Print detailed validation report"""
        if not self.validation_results:
            print("❌ No validation results available. Run validate_all_datasets() first.")
            return

        results = self.validation_results

        print("\n" + "=" * 60)
        print("📋 Data Validation Report")
        print("=" * 60)

        # Overall status
        status_emoji = "✅" if results["overall_status"] == "success" else "⚠️"
        print(f"\n{status_emoji} Overall Status: {results['overall_status'].upper()}")

        # Summary
        summary = results["summary"]
        print(f"\n📊 Summary:")
        print(f"   Datasets: {summary['valid_datasets']}/{summary['total_datasets']} valid")
        print(f"   Total Images: {summary['total_images']:,}")
        print(f"   Total Size: {summary['total_size_mb']:.1f} MB")

        # Dataset details
        print(f"\n📁 Dataset Details:")
        for dataset_name, dataset_result in results["datasets"].items():
            valid_emoji = "✅" if dataset_result.get("valid", False) else "❌"
            print(f"\n   {valid_emoji} {dataset_name.upper()}:")

            if "error" in dataset_result:
                print(f"      Error: {dataset_result['error']}")
                continue

            print(f"      Images: {dataset_result.get('total_images', 0):,}")
            print(f"      Size: {dataset_result.get('total_size_mb', 0):.1f} MB")

            # Sequence details
            sequences = dataset_result.get("sequences", {})
            if sequences:
                print(f"      Sequences:")
                for seq_name, seq_result in sequences.items():
                    seq_emoji = "✅" if seq_result.get("valid", False) else "❌"
                    print(f"        {seq_emoji} {seq_name}: {seq_result.get('num_images', 0)} images")

            # Issues
            issues = dataset_result.get("issues", [])
            if issues:
                print(f"      Issues:")
                for issue in issues[:3]:  # Show first 3 issues
                    print(f"        ⚠️  {issue}")
                if len(issues) > 3:
                    print(f"        ... and {len(issues) - 3} more")

        # Common issues
        if summary["common_issues"]:
            print(f"\n⚠️  Common Issues Across Datasets:")
            for issue in summary["common_issues"]:
                print(f"   • {issue}")

        # Recommendations
        print(f"\n💡 Recommendations:")
        if summary["valid_datasets"] == summary["total_datasets"]:
            print("   🎉 All datasets are valid! Ready for visual odometry processing.")
        else:
            print("   🔧 Fix validation issues before running visual odometry.")
            print("   📥 Consider re-running: python collect_data.py")

        print(f"\n🚀 Next Steps:")
        print("   • Run example: python run_example.py")
        print("   • Start web interface: python start_server.py")
        print("   • Open notebook: jupyter notebook notebooks/visual_odometry_demo.ipynb")

    def create_validation_plots(self, output_dir: str = "validation_plots"):
        """Create validation plots and visualizations"""
        if not self.validation_results:
            print("❌ No validation results available.")
            return

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Plot 1: Dataset sizes
        self._plot_dataset_sizes(output_path)

        # Plot 2: Image counts
        self._plot_image_counts(output_path)

        # Plot 3: Sample images from each dataset
        self._plot_sample_images(output_path)

        print(f"📊 Validation plots saved to: {output_path.absolute()}")

    def _plot_dataset_sizes(self, output_path: Path):
        """Plot dataset sizes"""
        datasets = self.validation_results["datasets"]
        names = []
        sizes = []

        for name, data in datasets.items():
            if data.get("valid", False):
                names.append(name.upper())
                sizes.append(data.get("total_size_mb", 0))

        if names:
            plt.figure(figsize=(10, 6))
            plt.bar(names, sizes, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
            plt.title('Dataset Sizes')
            plt.ylabel('Size (MB)')
            plt.xlabel('Dataset')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path / "dataset_sizes.png", dpi=150, bbox_inches='tight')
            plt.close()

    def _plot_image_counts(self, output_path: Path):
        """Plot image counts per dataset"""
        datasets = self.validation_results["datasets"]
        names = []
        counts = []

        for name, data in datasets.items():
            if data.get("valid", False):
                names.append(name.upper())
                counts.append(data.get("total_images", 0))

        if names:
            plt.figure(figsize=(10, 6))
            plt.bar(names, counts, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
            plt.title('Image Counts per Dataset')
            plt.ylabel('Number of Images')
            plt.xlabel('Dataset')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(output_path / "image_counts.png", dpi=150, bbox_inches='tight')
            plt.close()

    def _plot_sample_images(self, output_path: Path):
        """Plot sample images from each dataset"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        dataset_order = ["sample", "kitti", "tum", "test"]

        for i, dataset_name in enumerate(dataset_order):
            if i >= 4:
                break

            dataset_path = self.data_dir / dataset_name
            sample_img = None

            # Find a sample image
            if dataset_name == "sample":
                img_path = dataset_path / "forward" / "frame_000000.png"
            elif dataset_name == "kitti":
                img_path = dataset_path / "sequences/00/image_0/000000.png"
            elif dataset_name == "tum":
                img_path = dataset_path / "freiburg1_xyz/rgb/000000.png"
            else:  # test
                img_path = dataset_path / "pure_rotation/frame_000000.png"

            if img_path.exists():
                img = cv2.imread(str(img_path))
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    axes[i].imshow(img_rgb)
                    axes[i].set_title(f"{dataset_name.upper()} Sample")
                    axes[i].axis('off')
                else:
                    axes[i].text(0.5, 0.5, f"No {dataset_name}\nimage available",
                               ha='center', va='center', transform=axes[i].transAxes)
                    axes[i].axis('off')
            else:
                axes[i].text(0.5, 0.5, f"No {dataset_name}\nimage available",
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].axis('off')

        plt.tight_layout()
        plt.savefig(output_path / "sample_images.png", dpi=150, bbox_inches='tight')
        plt.close()


def main():
    """Main validation function"""
    validator = DataValidator()

    # Run validation
    results = validator.validate_all_datasets()

    # Print report
    validator.print_validation_report()

    # Create plots
    try:
        validator.create_validation_plots()
    except Exception as e:
        print(f"⚠️  Could not create validation plots: {e}")

    return results["overall_status"] == "success"


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)