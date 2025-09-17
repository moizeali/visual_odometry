"""
Performance metrics collection and analysis
"""

import time
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json


@dataclass
class FrameMetrics:
    """Metrics for a single frame processing"""
    frame_number: int
    timestamp: float
    processing_time: float
    keypoints_count: int
    matches_count: int
    inliers_count: int
    pose_position: List[float]
    pose_rotation: List[List[float]]


@dataclass
class ProcessingSession:
    """Complete processing session metrics"""
    session_id: str
    dataset: str
    sequence: str
    algorithm: str
    start_time: datetime
    end_time: Optional[datetime] = None
    frame_metrics: List[FrameMetrics] = field(default_factory=list)
    total_frames: int = 0
    success: bool = False


class MetricsCollector:
    """Collect and analyze visual odometry performance metrics"""

    def __init__(self):
        self.current_session: Optional[ProcessingSession] = None
        self.sessions: List[ProcessingSession] = []

    def start_session(self, dataset: str, sequence: str, algorithm: str) -> str:
        """Start a new processing session"""
        session_id = f"{dataset}_{sequence}_{int(time.time())}"

        self.current_session = ProcessingSession(
            session_id=session_id,
            dataset=dataset,
            sequence=sequence,
            algorithm=algorithm,
            start_time=datetime.now()
        )

        return session_id

    def record_frame(self, frame_number: int, processing_time: float,
                    keypoints: int, matches: int, inliers: int,
                    position: List[float], rotation: List[List[float]]):
        """Record metrics for a single frame"""
        if not self.current_session:
            raise ValueError("No active session. Call start_session() first.")

        metrics = FrameMetrics(
            frame_number=frame_number,
            timestamp=time.time(),
            processing_time=processing_time,
            keypoints_count=keypoints,
            matches_count=matches,
            inliers_count=inliers,
            pose_position=position,
            pose_rotation=rotation
        )

        self.current_session.frame_metrics.append(metrics)

    def end_session(self, success: bool = True):
        """End the current processing session"""
        if not self.current_session:
            raise ValueError("No active session to end.")

        self.current_session.end_time = datetime.now()
        self.current_session.success = success
        self.current_session.total_frames = len(self.current_session.frame_metrics)

        self.sessions.append(self.current_session)
        completed_session = self.current_session
        self.current_session = None

        return completed_session

    def get_session_summary(self, session_id: str = None) -> Dict:
        """Get summary statistics for a session"""
        if session_id:
            session = next((s for s in self.sessions if s.session_id == session_id), None)
            if not session:
                raise ValueError(f"Session {session_id} not found")
        else:
            session = self.current_session or self.sessions[-1] if self.sessions else None

        if not session:
            return {"error": "No session available"}

        if not session.frame_metrics:
            return {"error": "No frame metrics available"}

        # Calculate statistics
        processing_times = [m.processing_time for m in session.frame_metrics]
        keypoints_counts = [m.keypoints_count for m in session.frame_metrics]
        matches_counts = [m.matches_count for m in session.frame_metrics]
        inliers_counts = [m.inliers_count for m in session.frame_metrics]

        # Calculate trajectory length
        positions = [m.pose_position for m in session.frame_metrics]
        trajectory_length = 0.0
        if len(positions) > 1:
            for i in range(1, len(positions)):
                dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))
                trajectory_length += dist

        duration = (session.end_time - session.start_time).total_seconds() if session.end_time else 0

        return {
            "session_id": session.session_id,
            "dataset": session.dataset,
            "sequence": session.sequence,
            "algorithm": session.algorithm,
            "success": session.success,
            "total_frames": session.total_frames,
            "duration_seconds": duration,
            "fps": session.total_frames / duration if duration > 0 else 0,
            "processing_time": {
                "mean": np.mean(processing_times),
                "std": np.std(processing_times),
                "min": np.min(processing_times),
                "max": np.max(processing_times)
            },
            "keypoints": {
                "mean": np.mean(keypoints_counts),
                "std": np.std(keypoints_counts),
                "min": np.min(keypoints_counts),
                "max": np.max(keypoints_counts)
            },
            "matches": {
                "mean": np.mean(matches_counts),
                "std": np.std(matches_counts),
                "min": np.min(matches_counts),
                "max": np.max(matches_counts)
            },
            "inliers": {
                "mean": np.mean(inliers_counts),
                "std": np.std(inliers_counts),
                "min": np.min(inliers_counts),
                "max": np.max(inliers_counts)
            },
            "trajectory_length": trajectory_length
        }

    def export_session(self, session_id: str, output_file: str):
        """Export session data to JSON file"""
        session = next((s for s in self.sessions if s.session_id == session_id), None)
        if not session:
            raise ValueError(f"Session {session_id} not found")

        # Convert to serializable format
        export_data = {
            "session_info": {
                "session_id": session.session_id,
                "dataset": session.dataset,
                "sequence": session.sequence,
                "algorithm": session.algorithm,
                "start_time": session.start_time.isoformat(),
                "end_time": session.end_time.isoformat() if session.end_time else None,
                "success": session.success,
                "total_frames": session.total_frames
            },
            "frame_metrics": [
                {
                    "frame_number": m.frame_number,
                    "timestamp": m.timestamp,
                    "processing_time": m.processing_time,
                    "keypoints_count": m.keypoints_count,
                    "matches_count": m.matches_count,
                    "inliers_count": m.inliers_count,
                    "pose_position": m.pose_position,
                    "pose_rotation": m.pose_rotation
                }
                for m in session.frame_metrics
            ],
            "summary": self.get_session_summary(session_id)
        }

        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)

    def compare_sessions(self, session_ids: List[str]) -> Dict:
        """Compare multiple sessions"""
        sessions = [s for s in self.sessions if s.session_id in session_ids]

        if len(sessions) != len(session_ids):
            raise ValueError("Some sessions not found")

        comparison = {
            "sessions": session_ids,
            "comparison_metrics": {}
        }

        # Compare key metrics
        for metric in ["fps", "trajectory_length"]:
            summaries = [self.get_session_summary(sid) for sid in session_ids]
            values = [s.get(metric, 0) for s in summaries]

            comparison["comparison_metrics"][metric] = {
                "values": dict(zip(session_ids, values)),
                "best_session": session_ids[np.argmax(values)] if values else None,
                "worst_session": session_ids[np.argmin(values)] if values else None
            }

        return comparison

    def get_real_time_stats(self) -> Dict:
        """Get real-time statistics for current session"""
        if not self.current_session or not self.current_session.frame_metrics:
            return {"error": "No active session with metrics"}

        last_10_frames = self.current_session.frame_metrics[-10:]

        return {
            "current_frame": len(self.current_session.frame_metrics),
            "recent_fps": 1.0 / np.mean([m.processing_time for m in last_10_frames]) if last_10_frames else 0,
            "recent_keypoints": np.mean([m.keypoints_count for m in last_10_frames]) if last_10_frames else 0,
            "recent_matches": np.mean([m.matches_count for m in last_10_frames]) if last_10_frames else 0,
            "session_duration": (datetime.now() - self.current_session.start_time).total_seconds()
        }