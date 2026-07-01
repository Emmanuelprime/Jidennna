"""
Perception Layer for Autonomous Delivery Robot

Modular perception system providing environmental understanding
for local planning and MPC control.
"""

from .interfaces.core_interfaces import (
    Detection,
    TrackedObject,
    Obstacle,
    RobotPose,
    PerceptionOutput,
    FrameID
)

__version__ = "1.0.0"
__author__ = "Emmanuel Prime CEO Prime Robotics"