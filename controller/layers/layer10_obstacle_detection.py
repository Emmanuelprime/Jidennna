"""
Layer 10: Obstacle detection and reactive safety filtering.
"""
import numpy as np

from config.robot_params import RobotParams


class ObstacleDetection:
    """
    Process three forward distance sensors and adjust velocity commands.

    Expected feedback keys:
        distance_left, distance_center, distance_right

    Distances are in meters. Missing or invalid readings are treated as clear.
    """

    SENSOR_KEYS = ("distance_left", "distance_center", "distance_right")

    def __init__(
        self,
        stop_distance=RobotParams.OBSTACLE_STOP_DISTANCE,
        slow_distance=RobotParams.OBSTACLE_SLOW_DISTANCE,
        clear_distance=RobotParams.OBSTACLE_CLEAR_DISTANCE,
        max_range=RobotParams.OBSTACLE_SENSOR_MAX_RANGE,
    ):
        if not 0 < stop_distance < slow_distance <= clear_distance:
            raise ValueError("Expected stop_distance < slow_distance <= clear_distance")

        self.stop_distance = stop_distance
        self.slow_distance = slow_distance
        self.clear_distance = clear_distance
        self.max_range = max_range
        self.latest_reading = self._sanitize_distances(None, None, None)
        self.latest_status = self._build_status(self.latest_reading, "clear")

    def update_from_feedback(self, feedback):
        """Read sensor distances from a feedback dictionary."""
        distances = self._sanitize_distances(
            feedback.get("distance_left"),
            feedback.get("distance_center", feedback.get("distance_centre", feedback.get("distance_centr"))),
            feedback.get("distance_right"),
        )
        self.latest_reading = distances
        self.latest_status = self._classify(distances)
        return self.latest_status

    def filter_control(self, control, feedback=None):
        """
        Apply obstacle safety behavior to [v, omega].

        Returns:
            safe_control, status
        """
        if feedback is not None:
            status = self.update_from_feedback(feedback)
        else:
            status = self.latest_status

        v_cmd, omega_cmd = np.asarray(control, dtype=float)
        left, center, right = status["distances"].values()

        if status["state"] == "blocked":
            v_cmd = min(v_cmd, 0.0)
            omega_cmd = self._avoidance_turn(left, right, omega_cmd)
        elif status["state"] == "slow":
            scale = np.clip(
                (center - self.stop_distance) / (self.slow_distance - self.stop_distance),
                0.0,
                1.0,
            )
            v_cmd = min(v_cmd, RobotParams.MPC_MAX_VELOCITY * scale)
            omega_cmd += self._side_bias(left, right)
        else:
            omega_cmd += self._side_bias(left, right)

        safe_control = np.array([
            np.clip(v_cmd, -RobotParams.MPC_MAX_VELOCITY, RobotParams.MPC_MAX_VELOCITY),
            np.clip(omega_cmd, -RobotParams.MPC_MAX_ANGULAR_VEL, RobotParams.MPC_MAX_ANGULAR_VEL),
        ])

        status = status.copy()
        status["control_modified"] = not np.allclose(safe_control, control)
        return safe_control, status

    def get_status(self):
        return self.latest_status.copy()

    def _classify(self, distances):
        left, center, right = distances.values()
        nearest = min(left, center, right)

        if center <= self.stop_distance:
            state = "blocked"
        elif center <= self.slow_distance or nearest <= self.stop_distance:
            state = "slow"
        elif nearest < self.clear_distance:
            state = "caution"
        else:
            state = "clear"

        return self._build_status(distances, state)

    def _build_status(self, distances, state):
        nearest_sensor = min(distances, key=distances.get)
        return {
            "state": state,
            "distances": distances.copy(),
            "nearest_sensor": nearest_sensor,
            "nearest_distance": distances[nearest_sensor],
        }

    def _sanitize_distances(self, left, center, right):
        values = {
            "left": left,
            "center": center,
            "right": right,
        }

        for key, value in values.items():
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = self.max_range

            if not np.isfinite(value) or value < 0:
                value = self.max_range

            values[key] = min(value, self.max_range)

        return values

    def _avoidance_turn(self, left, right, current_omega):
        if left >= right:
            preferred_turn = RobotParams.OBSTACLE_AVOID_TURN_RATE
        else:
            preferred_turn = -RobotParams.OBSTACLE_AVOID_TURN_RATE

        if abs(current_omega) > abs(preferred_turn) and np.sign(current_omega) == np.sign(preferred_turn):
            return current_omega

        return preferred_turn

    def _side_bias(self, left, right):
        imbalance = left - right
        if abs(imbalance) < 1e-3:
            return 0.0

        return RobotParams.OBSTACLE_SIDE_TURN_GAIN * imbalance
