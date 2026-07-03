from mpc_controller.controller import MPCController, MPCParams
from communication.robot_interface import RobotInterface

robot = RobotInterface('/dev/ttyUSB0')
robot.connect()

# Create MPC
mpc = MPCController(robot)

# Go to (1.0, 1.0)
mpc.run_control_loop(target_x=1.0, target_y=1.0, duration=10.0)