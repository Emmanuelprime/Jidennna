from mpc_controller.controller import MPCController, create_circular_trajectory
from communication.robot_interface import RobotInterface
import time

robot = RobotInterface('COM24')
robot.connect()
# Create circular trajectory
trajectory = create_circular_trajectory(0, 0, 0.5, 20)

mpc = MPCController(robot)

# Follow trajectory
for _ in range(5):  # 5 laps
    for i in range(len(trajectory.x)):
        v, w = mpc.follow_trajectory(trajectory)
        robot.set_velocity(v, w)
        time.sleep(0.05)