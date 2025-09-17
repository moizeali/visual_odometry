"""
Logging utilities for Visual Odometry System
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
import json


def setup_logger(name: str, log_level: str = "INFO", log_dir: str = "logs") -> logging.Logger:
    """
    Set up a logger with both file and console handlers

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files

    Returns:
        Configured logger instance
    """
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )

    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # File handler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_handler = logging.FileHandler(
        log_path / f"{name}_{timestamp}.log",
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(simple_formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


class StructuredLogger:
    """Structured logging with JSON output for better parsing"""

    def __init__(self, name: str, log_dir: str = "logs"):
        self.logger = setup_logger(name, log_dir=log_dir)
        self.log_dir = Path(log_dir)

        # JSON log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_file = self.log_dir / f"{name}_structured_{timestamp}.jsonl"

    def log_event(self, event_type: str, data: dict, level: str = "INFO"):
        """Log structured event with JSON format"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "level": level,
            "data": data
        }

        # Log to JSON file
        with open(self.json_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Log to regular logger
        message = f"{event_type}: {json.dumps(data, indent=None)}"
        getattr(self.logger, level.lower())(message)

    def log_processing_start(self, dataset: str, sequence: str, total_frames: int):
        """Log processing start event"""
        self.log_event("processing_start", {
            "dataset": dataset,
            "sequence": sequence,
            "total_frames": total_frames
        })

    def log_frame_processed(self, frame_num: int, stats: dict):
        """Log frame processing event"""
        self.log_event("frame_processed", {
            "frame_number": frame_num,
            "keypoints": stats.get("keypoints_count", 0),
            "matches": stats.get("matches_count", 0),
            "inliers": stats.get("inliers_count", 0)
        })

    def log_processing_complete(self, total_time: float, total_frames: int):
        """Log processing completion event"""
        self.log_event("processing_complete", {
            "total_time_seconds": total_time,
            "total_frames": total_frames,
            "fps": total_frames / total_time if total_time > 0 else 0
        })

    def log_error(self, error_type: str, error_message: str, context: dict = None):
        """Log error event"""
        self.log_event("error", {
            "error_type": error_type,
            "error_message": error_message,
            "context": context or {}
        }, level="ERROR")