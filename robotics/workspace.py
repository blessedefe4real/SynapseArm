"""Reachable workspace by Monte-Carlo sampling of the joint space.

Includes xy- and yz-plane projections.
"""
from __future__ import annotations

import numpy as np

from .kinematics import DHRobot


def sample_workspace(robot: DHRobot, n: int = 50_000, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return (Q, P): sampled joint vectors and the corresponding end-effector positions (mm)."""
    Q = robot.random_q(n, rng=seed)
    P = robot.fk_batch(Q)[:, :3, 3]
    return Q, P


def plot_workspace(P: np.ndarray, path: str | None = None, color=None):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=0.3, c=color, alpha=0.4)
    ax.set(title="Reachable workspace", xlabel="x (mm)", ylabel="y (mm)", zlabel="z (mm)")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.scatter(P[:, 0], P[:, 1], s=0.3, c=color, alpha=0.3)
    ax2.set(title="Projection onto xy-plane", xlabel="x (mm)", ylabel="y (mm)", aspect="equal")

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.scatter(P[:, 1], P[:, 2], s=0.3, c=color, alpha=0.3)
    ax3.set(title="Projection onto yz-plane", xlabel="y (mm)", ylabel="z (mm)", aspect="equal")
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig
