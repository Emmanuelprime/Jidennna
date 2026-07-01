"""
Robot Physical Parameters Configuration
"""
import numpy as np

class RobotParams:
    WHEEL_DIAMETER = 0.17  # m
    WHEEL_RADIUS = 0.085   # m
    WHEEL_SEPARATION = 0.521  # m (track width L)
    
    # Derived parameters
    WHEEL_CIRCUMFERENCE = np.pi * WHEEL_DIAMETER
    
    # MPC parameters
    MPC_DT = 0.1           # Time step (s)
    MPC_HORIZON = 10       # Prediction horizon
    MPC_MAX_VELOCITY = 1.0  # m/s
    MPC_MAX_ANGULAR_VEL = 2.0  # rad/s
    
    # Control weights
    Q_X = 10.0
    Q_Y = 10.0
    Q_THETA = 5.0
    R_V = 0.1
    R_OMEGA = 0.1
    
    # State estimation
    ENCODER_RESOLUTION = 90  # pulses per revolution
    IMU_UPDATE_RATE = 100      # Hz
    
    # Communication
    COMM_BAUDRATE = 115200
    COMM_UPDATE_RATE = 10      # Hz

    # Obstacle detection
    OBSTACLE_SENSOR_MAX_RANGE = 4.0      # m; used when no obstacle is detected
    OBSTACLE_STOP_DISTANCE = 0.35        # m; emergency stop/turn threshold
    OBSTACLE_SLOW_DISTANCE = 0.9         # m; begin reducing forward velocity
    OBSTACLE_CLEAR_DISTANCE = 1.2        # m; considered clear for status
    OBSTACLE_AVOID_TURN_RATE = 1.0       # rad/s; minimum avoidance turn command
    OBSTACLE_SIDE_TURN_GAIN = 0.8        # rad/s per meter side imbalance
    
    @classmethod
    def get_robot_params(cls):
        return {
            'wheel_radius': cls.WHEEL_RADIUS,
            'wheel_separation': cls.WHEEL_SEPARATION,
            'dt': cls.MPC_DT,
            'horizon': cls.MPC_HORIZON,
            'max_v': cls.MPC_MAX_VELOCITY,
            'max_w': cls.MPC_MAX_ANGULAR_VEL,
            'obstacle_stop_distance': cls.OBSTACLE_STOP_DISTANCE,
            'obstacle_slow_distance': cls.OBSTACLE_SLOW_DISTANCE
        }
