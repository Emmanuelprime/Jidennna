"""
Layer 9: Waypoint generation helpers.
"""
import numpy as np


def generate_sine_waypoints(amplitude=5.0, frequency=0.2, x_start=0.0, x_end=20.0, step=0.5):
    """
    Generate waypoints along y = amplitude * sin(frequency * x).
    """
    if step <= 0:
        raise ValueError("step must be greater than zero")

    waypoints = []
    for x in np.arange(x_start, x_end + step, step):
        y = amplitude * np.sin(frequency * x)
        waypoints.append((float(x), float(y)))

    return waypoints


if __name__ == "__main__":
    print(generate_sine_waypoints())
