"""
Visualization utilities for trajectory and metrics plotting
"""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional, Tuple
import io
import base64


class TrajectoryPlotter:
    """Create various trajectory visualizations"""

    def __init__(self, style: str = "seaborn"):
        plt.style.use(style if style in plt.style.available else 'default')

    def plot_trajectory_2d(self, trajectory: np.ndarray,
                          ground_truth: Optional[np.ndarray] = None,
                          title: str = "2D Trajectory") -> str:
        """
        Plot 2D trajectory (X-Z plane)

        Returns base64 encoded image string
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        if len(trajectory) > 0:
            ax.plot(trajectory[:, 0], trajectory[:, 2], 'b-', linewidth=2, label='Estimated')
            ax.scatter(trajectory[0, 0], trajectory[0, 2], c='green', s=100,
                      marker='o', label='Start', zorder=5)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 2], c='red', s=100,
                      marker='s', label='End', zorder=5)

        if ground_truth is not None and len(ground_truth) > 0:
            ax.plot(ground_truth[:, 0], ground_truth[:, 2], 'r--', linewidth=2,
                   alpha=0.7, label='Ground Truth')

        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Z (meters)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return image_base64

    def plot_trajectory_3d(self, trajectory: np.ndarray,
                          ground_truth: Optional[np.ndarray] = None,
                          title: str = "3D Trajectory") -> str:
        """
        Plot 3D trajectory

        Returns base64 encoded image string
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        if len(trajectory) > 0:
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2],
                   'b-', linewidth=2, label='Estimated')
            ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2],
                      c='green', s=100, marker='o', label='Start')
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                      c='red', s=100, marker='s', label='End')

        if ground_truth is not None and len(ground_truth) > 0:
            ax.plot(ground_truth[:, 0], ground_truth[:, 1], ground_truth[:, 2],
                   'r--', linewidth=2, alpha=0.7, label='Ground Truth')

        ax.set_xlabel('X (meters)')
        ax.set_ylabel('Y (meters)')
        ax.set_zlabel('Z (meters)')
        ax.set_title(title)
        ax.legend()

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return image_base64

    def create_interactive_3d(self, trajectory: np.ndarray,
                             ground_truth: Optional[np.ndarray] = None,
                             title: str = "Interactive 3D Trajectory") -> str:
        """
        Create interactive 3D plot using Plotly

        Returns HTML string
        """
        fig = go.Figure()

        if len(trajectory) > 0:
            # Estimated trajectory
            fig.add_trace(go.Scatter3d(
                x=trajectory[:, 0],
                y=trajectory[:, 1],
                z=trajectory[:, 2],
                mode='lines+markers',
                name='Estimated',
                line=dict(color='blue', width=4),
                marker=dict(size=2)
            ))

            # Start point
            fig.add_trace(go.Scatter3d(
                x=[trajectory[0, 0]],
                y=[trajectory[0, 1]],
                z=[trajectory[0, 2]],
                mode='markers',
                name='Start',
                marker=dict(color='green', size=10, symbol='circle')
            ))

            # End point
            fig.add_trace(go.Scatter3d(
                x=[trajectory[-1, 0]],
                y=[trajectory[-1, 1]],
                z=[trajectory[-1, 2]],
                mode='markers',
                name='End',
                marker=dict(color='red', size=10, symbol='square')
            ))

        if ground_truth is not None and len(ground_truth) > 0:
            fig.add_trace(go.Scatter3d(
                x=ground_truth[:, 0],
                y=ground_truth[:, 1],
                z=ground_truth[:, 2],
                mode='lines',
                name='Ground Truth',
                line=dict(color='red', width=3, dash='dash'),
                opacity=0.7
            ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X (meters)',
                yaxis_title='Y (meters)',
                zaxis_title='Z (meters)',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )

        return fig.to_html(include_plotlyjs=True)

    def plot_metrics_timeline(self, metrics: List[Dict]) -> str:
        """
        Plot processing metrics over time

        Returns base64 encoded image string
        """
        if not metrics:
            return ""

        frames = [m.get('frame_number', i) for i, m in enumerate(metrics)]
        keypoints = [m.get('keypoints_count', 0) for m in metrics]
        matches = [m.get('matches_count', 0) for m in metrics]
        processing_times = [m.get('processing_time', 0) * 1000 for m in metrics]  # Convert to ms

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Processing Metrics Timeline')

        # Keypoints
        axes[0, 0].plot(frames, keypoints, 'b-', linewidth=2)
        axes[0, 0].set_title('Keypoints Detected')
        axes[0, 0].set_xlabel('Frame')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].grid(True, alpha=0.3)

        # Matches
        axes[0, 1].plot(frames, matches, 'g-', linewidth=2)
        axes[0, 1].set_title('Feature Matches')
        axes[0, 1].set_xlabel('Frame')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].grid(True, alpha=0.3)

        # Processing time
        axes[1, 0].plot(frames, processing_times, 'r-', linewidth=2)
        axes[1, 0].set_title('Processing Time')
        axes[1, 0].set_xlabel('Frame')
        axes[1, 0].set_ylabel('Time (ms)')
        axes[1, 0].grid(True, alpha=0.3)

        # Match ratio
        match_ratios = [m/k if k > 0 else 0 for k, m in zip(keypoints, matches)]
        axes[1, 1].plot(frames, match_ratios, 'm-', linewidth=2)
        axes[1, 1].set_title('Match Ratio')
        axes[1, 1].set_xlabel('Frame')
        axes[1, 1].set_ylabel('Matches / Keypoints')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()

        return image_base64

    def create_metrics_dashboard(self, metrics: List[Dict]) -> str:
        """
        Create interactive metrics dashboard using Plotly

        Returns HTML string
        """
        if not metrics:
            return "<div>No metrics data available</div>"

        frames = list(range(len(metrics)))
        keypoints = [m.get('keypoints_count', 0) for m in metrics]
        matches = [m.get('matches_count', 0) for m in metrics]
        processing_times = [m.get('processing_time', 0) * 1000 for m in metrics]

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Keypoints Detected', 'Feature Matches',
                          'Processing Time (ms)', 'Match Efficiency'),
            vertical_spacing=0.08
        )

        # Keypoints
        fig.add_trace(
            go.Scatter(x=frames, y=keypoints, mode='lines', name='Keypoints',
                      line=dict(color='blue')),
            row=1, col=1
        )

        # Matches
        fig.add_trace(
            go.Scatter(x=frames, y=matches, mode='lines', name='Matches',
                      line=dict(color='green')),
            row=1, col=2
        )

        # Processing time
        fig.add_trace(
            go.Scatter(x=frames, y=processing_times, mode='lines', name='Processing Time',
                      line=dict(color='red')),
            row=2, col=1
        )

        # Match efficiency
        efficiency = [m/k if k > 0 else 0 for k, m in zip(keypoints, matches)]
        fig.add_trace(
            go.Scatter(x=frames, y=efficiency, mode='lines', name='Match Ratio',
                      line=dict(color='purple')),
            row=2, col=2
        )

        fig.update_layout(
            height=600,
            showlegend=False,
            title_text="Visual Odometry Performance Metrics"
        )

        # Update x-axis labels
        fig.update_xaxes(title_text="Frame Number")
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=2)
        fig.update_yaxes(title_text="Milliseconds", row=2, col=1)
        fig.update_yaxes(title_text="Ratio", row=2, col=2)

        return fig.to_html(include_plotlyjs=True)

    def calculate_trajectory_error(self, estimated: np.ndarray,
                                 ground_truth: np.ndarray) -> Dict:
        """
        Calculate trajectory error metrics
        """
        if len(estimated) != len(ground_truth):
            # Interpolate to match lengths
            min_len = min(len(estimated), len(ground_truth))
            estimated = estimated[:min_len]
            ground_truth = ground_truth[:min_len]

        # Absolute Trajectory Error (ATE)
        ate_per_frame = np.linalg.norm(estimated - ground_truth, axis=1)
        ate_rmse = np.sqrt(np.mean(ate_per_frame ** 2))

        # Relative Pose Error (RPE) - simplified version
        rpe_trans = []
        rpe_rot = []

        for i in range(1, len(estimated)):
            # Translation difference
            est_diff = estimated[i] - estimated[i-1]
            gt_diff = ground_truth[i] - ground_truth[i-1]
            trans_error = np.linalg.norm(est_diff - gt_diff)
            rpe_trans.append(trans_error)

        rpe_trans_rmse = np.sqrt(np.mean(np.array(rpe_trans) ** 2)) if rpe_trans else 0

        return {
            "ate_rmse": float(ate_rmse),
            "ate_mean": float(np.mean(ate_per_frame)),
            "ate_std": float(np.std(ate_per_frame)),
            "ate_max": float(np.max(ate_per_frame)),
            "rpe_trans_rmse": float(rpe_trans_rmse),
            "trajectory_length_estimated": float(np.sum(np.linalg.norm(np.diff(estimated, axis=0), axis=1))),
            "trajectory_length_ground_truth": float(np.sum(np.linalg.norm(np.diff(ground_truth, axis=0), axis=1)))
        }