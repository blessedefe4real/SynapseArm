"""Plot the reachable workspace and its xy / yz projections.

Points are coloured by Yoshikawa manipulability so near-singular regions show up dark.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robotics.kinematics import DHRobot  # noqa: E402
from robotics.workspace import plot_workspace, sample_workspace  # noqa: E402

robot = DHRobot()
Q, P = sample_workspace(robot, n=20_000)
w = np.array([robot.manipulability(q) for q in Q])
(ROOT / "results").mkdir(exist_ok=True)
plot_workspace(P, ROOT / "results" / "workspace.png", color=np.log10(w + 1))
print("saved results/workspace.png")
