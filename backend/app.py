"""
FastAPI backend for Visual Odometry web application
Provides REST API endpoints and WebSocket for real-time processing
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import asyncio
import logging
import cv2
import numpy as np
from pathlib import Path
import tempfile
import shutil
import uvicorn

# Import our modules
from core.visual_odometry import (
    VisualOdometryPipeline, CameraParams, MonocularVO, StereoVO
)
from data.dataset_loader import DatasetManager
from api.websocket_manager import WebSocketManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Visual Odometry System",
    description="Enhanced Visual Odometry with real-time processing and web interface",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files and templates
app.mount("/static", StaticFiles(directory="frontend/static"), name="static")
templates = Jinja2Templates(directory="frontend/templates")

# Global objects
dataset_manager = DatasetManager()
websocket_manager = WebSocketManager()
current_pipeline: Optional[VisualOdometryPipeline] = None

# Pydantic models
class CameraConfig(BaseModel):
    fx: float = 525.0
    fy: float = 525.0
    cx: float = 319.5
    cy: float = 239.5
    baseline: float = 0.075

class ProcessingConfig(BaseModel):
    camera: CameraConfig
    mode: str = "mono"  # "mono" or "stereo"
    detector: str = "ORB"  # "ORB", "SIFT", "SURF"
    dataset: str = "sample"
    sequence: Optional[str] = None

class ProcessingResult(BaseModel):
    success: bool
    message: str
    trajectory: List[Dict]
    stats: List[Dict]
    total_frames: int

# API Routes

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main dashboard"""
    return templates.TemplateResponse("index.html", {"request": {}})

@app.get("/api/datasets")
async def get_datasets():
    """Get available datasets"""
    try:
        datasets = dataset_manager.list_datasets()
        return {
            "datasets": [
                {
                    "id": key,
                    "name": dataset.name,
                    "type": dataset.type,
                    "sequences": dataset.sequences,
                    "size_mb": dataset.size_mb,
                    "description": dataset.description
                }
                for key, dataset in datasets.items()
            ]
        }
    except Exception as e:
        logger.error(f"Error getting datasets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/prepare-sample")
async def prepare_sample_data():
    """Prepare sample data for testing"""
    try:
        sample_path = dataset_manager.prepare_sample_data()
        return {
            "success": True,
            "message": "Sample data prepared successfully",
            "path": sample_path
        }
    except Exception as e:
        logger.error(f"Error preparing sample data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload-images")
async def upload_images(files: List[UploadFile] = File(...)):
    """Upload custom images"""
    try:
        # Save uploaded files temporarily
        temp_files = []
        for file in files:
            if file.content_type.startswith('image/'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                    shutil.copyfileobj(file.file, tmp)
                    temp_files.append(tmp.name)

        # Upload to dataset manager
        if temp_files:
            custom_path = dataset_manager.upload_custom_images(temp_files)

            # Clean up temp files
            for temp_file in temp_files:
                try:
                    Path(temp_file).unlink()
                except:
                    pass

            return {
                "success": True,
                "message": f"Uploaded {len(temp_files)} images",
                "path": custom_path
            }
        else:
            raise HTTPException(status_code=400, detail="No valid images uploaded")

    except Exception as e:
        logger.error(f"Error uploading images: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process")
async def process_dataset(config: ProcessingConfig):
    """Process dataset with visual odometry"""
    global current_pipeline

    try:
        # Create camera parameters
        camera_params = CameraParams(
            fx=config.camera.fx,
            fy=config.camera.fy,
            cx=config.camera.cx,
            cy=config.camera.cy,
            baseline=config.camera.baseline
        )

        # Create pipeline
        current_pipeline = VisualOdometryPipeline(
            camera_params=camera_params,
            mode=config.mode,
            detector_type=config.detector
        )

        # Load dataset
        dataset = dataset_manager.load_dataset(
            config.dataset,
            config.sequence
        )

        # Process images
        if config.mode == "mono":
            images = dataset.get('images', [])
        else:
            images = dataset.get('left_images', [])

        if not images:
            raise HTTPException(status_code=400, detail="No images found in dataset")

        # Process sequence
        results = current_pipeline.process_sequence(images)

        return ProcessingResult(
            success=results['success'],
            message="Processing completed successfully" if results['success'] else results.get('error', 'Unknown error'),
            trajectory=results['trajectory'],
            stats=results['stats'],
            total_frames=len(images)
        )

    except Exception as e:
        logger.error(f"Error processing dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/trajectory")
async def get_trajectory():
    """Get current trajectory data"""
    global current_pipeline

    if current_pipeline is None:
        raise HTTPException(status_code=400, detail="No active processing pipeline")

    try:
        trajectory = current_pipeline.get_trajectory_array()

        return {
            "trajectory": trajectory.tolist() if len(trajectory) > 0 else [],
            "num_points": len(trajectory)
        }
    except Exception as e:
        logger.error(f"Error getting trajectory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset")
async def reset_pipeline():
    """Reset the visual odometry pipeline"""
    global current_pipeline

    if current_pipeline:
        current_pipeline.reset()

    return {"success": True, "message": "Pipeline reset successfully"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and handle messages
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message.get("type") == "get_status":
                status = {
                    "type": "status",
                    "pipeline_active": current_pipeline is not None,
                    "connected_clients": len(websocket_manager.active_connections)
                }
                await websocket.send_text(json.dumps(status))

    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Visual Odometry API",
        "version": "1.0.0"
    }

if __name__ == "__main__":
    # Create directories if they don't exist
    Path("frontend/static").mkdir(parents=True, exist_ok=True)
    Path("frontend/templates").mkdir(parents=True, exist_ok=True)
    Path("datasets").mkdir(exist_ok=True)

    # Run the application
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )