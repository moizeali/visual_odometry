"""
Additional API routes for Visual Odometry System
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Optional
import logging
import numpy as np
from ..core.visual_odometry import CameraParams, VisualOdometryPipeline
from ..data.dataset_loader import DatasetManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["visual_odometry"])

# Dependency to get dataset manager
def get_dataset_manager():
    return DatasetManager()

# Dependency to get current pipeline
current_pipeline = None

def get_pipeline():
    global current_pipeline
    if current_pipeline is None:
        raise HTTPException(status_code=400, detail="No active pipeline")
    return current_pipeline

@router.get("/datasets/{dataset_name}/info")
async def get_dataset_info(dataset_name: str, dm: DatasetManager = Depends(get_dataset_manager)):
    """Get detailed information about a specific dataset"""
    try:
        info = dm.get_dataset_info(dataset_name)
        if info is None:
            raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")

        return {
            "name": info.name,
            "type": info.type,
            "sequences": info.sequences,
            "download_url": info.download_url,
            "size_mb": info.size_mb,
            "description": info.description
        }
    except Exception as e:
        logger.error(f"Error getting dataset info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/datasets/{dataset_name}/sequences")
async def list_sequences(dataset_name: str):
    """List available sequences for a dataset"""
    try:
        if dataset_name == "kitti":
            sequences = [f"{i:02d}" for i in range(11)]  # 00-10
        elif dataset_name == "tum_rgbd":
            sequences = ["freiburg1_xyz", "freiburg1_rpy", "freiburg2_xyz", "freiburg3_long"]
        elif dataset_name == "euroc_mav":
            sequences = ["MH_01", "MH_02", "MH_03", "MH_04", "MH_05", "V1_01", "V1_02", "V1_03", "V2_01", "V2_02"]
        else:
            sequences = ["default"]

        return {"sequences": sequences}
    except Exception as e:
        logger.error(f"Error listing sequences: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pipeline/create")
async def create_pipeline(camera_params: dict, mode: str = "mono", detector: str = "ORB"):
    """Create a new visual odometry pipeline"""
    global current_pipeline

    try:
        camera = CameraParams(
            fx=camera_params.get("fx", 525.0),
            fy=camera_params.get("fy", 525.0),
            cx=camera_params.get("cx", 319.5),
            cy=camera_params.get("cy", 239.5),
            baseline=camera_params.get("baseline", 0.075)
        )

        current_pipeline = VisualOdometryPipeline(
            camera_params=camera,
            mode=mode,
            detector_type=detector
        )

        return {
            "success": True,
            "message": "Pipeline created successfully",
            "pipeline_id": id(current_pipeline)
        }
    except Exception as e:
        logger.error(f"Error creating pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/pipeline/status")
async def get_pipeline_status():
    """Get current pipeline status"""
    global current_pipeline

    return {
        "active": current_pipeline is not None,
        "pipeline_id": id(current_pipeline) if current_pipeline else None,
        "trajectory_points": len(current_pipeline.get_trajectory_array()) if current_pipeline else 0
    }

@router.post("/pipeline/process-single")
async def process_single_frame(image_path: str, pipeline = Depends(get_pipeline)):
    """Process a single frame"""
    try:
        import cv2

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise HTTPException(status_code=400, detail="Could not load image")

        # Process frame
        if pipeline.mode == "mono":
            pose, stats = pipeline.vo.process_frame(image)
        else:
            # For stereo, we'd need both left and right images
            pose, stats = pipeline.vo.process_frame(image)

        return {
            "pose": {
                "position": pose.t.flatten().tolist(),
                "rotation": pose.R.tolist(),
                "timestamp": pose.timestamp
            },
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error processing frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trajectory/export")
async def export_trajectory(format: str = "json", pipeline = Depends(get_pipeline)):
    """Export trajectory in various formats"""
    try:
        trajectory = pipeline.get_trajectory_array()

        if format == "json":
            return {
                "trajectory": trajectory.tolist(),
                "format": "json",
                "num_points": len(trajectory)
            }
        elif format == "csv":
            import io
            import csv

            output = io.StringIO()
            writer = csv.writer(output)
            writer.writerow(["x", "y", "z"])

            for point in trajectory:
                writer.writerow([point[0], point[1], point[2]])

            return {
                "data": output.getvalue(),
                "format": "csv",
                "num_points": len(trajectory)
            }
        else:
            raise HTTPException(status_code=400, detail="Unsupported format")

    except Exception as e:
        logger.error(f"Error exporting trajectory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics/summary")
async def get_metrics_summary():
    """Get processing metrics summary"""
    # This would be implemented with actual metrics collection
    return {
        "total_frames_processed": 0,
        "average_processing_time": 0.0,
        "average_keypoints": 0,
        "average_matches": 0,
        "trajectory_length": 0.0
    }