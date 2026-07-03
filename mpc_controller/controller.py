#!/usr/bin/env python3
"""
MPC Controller for Differential Drive Robot
============================================
Model Predictive Control for trajectory tracking.
Supports both position tracking and velocity tracking.
"""

import numpy as np
import time
import math
from typing import Tuple, List, Optional
from dataclasses import dataclass
from scipy.optimize import minimize
from communication.robot_interface import RobotInterface, RobotState

# ─── Data Classes ─────────────────────────────────────────────────────────────

@dataclass
class MPCParams:
    """MPC parameters"""
    horizon: int = 10          # Prediction horizon
    dt: float = 0.05           # Control period (50ms)
    
    # Weights for cost function
    Q_pos: float = 10.0        # Position error weight
    Q_yaw: float = 5.0         # Heading error weight
    Q_v: float = 1.0           # Velocity error weight
    Q_w: float = 0.5           # Angular velocity error weight
    R_v: float = 0.1           # Control effort weight (linear)
    R_w: float = 0.05          # Control effort weight (angular)
    R_dv: float = 1.0          # Control change weight (linear)
    R_dw: float = 0.5          # Control change weight (angular)
    
    # Constraints
    max_v: float = 1.2         # Max linear velocity (m/s)
    max_w: float = 2.0         # Max angular velocity (rad/s)
    max_accel: float = 1.0     # Max linear acceleration (m/s²)
    max_ang_accel: float = 2.0 # Max angular acceleration (rad/s²)

@dataclass
class MPCTrajectory:
    """Trajectory reference for MPC"""
    x: List[float]   # X positions
    y: List[float]   # Y positions
    yaw: List[float] # Heading angles
    v: List[float]   # Linear velocities
    w: List[float]   # Angular velocities
    
    def get_at_index(self, idx: int) -> Tuple[float, float, float, float, float]:
        """Get trajectory point at index"""
        if idx < len(self.x):
            return (self.x[idx], self.y[idx], self.yaw[idx], self.v[idx], self.w[idx])
        return (self.x[-1], self.y[-1], self.yaw[-1], 0.0, 0.0)

# ─── MPC Controller ──────────────────────────────────────────────────────────

class MPCController:
    """
    Model Predictive Controller for differential drive robot.
    
    Supports:
    - Position tracking (go to (x, y))
    - Trajectory tracking (follow path)
    - Velocity tracking (track v, w)
    """
    
    def __init__(self, robot: RobotInterface, params: MPCParams = None):
        self.robot = robot
        self.params = params or MPCParams()
        
        # State
        self.current_state: Optional[RobotState] = None
        self.last_v = 0.0
        self.last_w = 0.0
        
        # Reference trajectory
        self.trajectory: Optional[MPCTrajectory] = None
        self.trajectory_index = 0
        
        # Running flag
        self.running = False
        
        # Register callback for state updates
        robot.register_callback(self._on_state_update)
        
        # Statistics
        self.control_time = 0.0
        self.iterations = 0
        
        print("✅ MPC Controller initialized")
    
    def _on_state_update(self, state: RobotState):
        """Store latest state"""
        self.current_state = state
    
    def _kinematic_model(self, state: np.ndarray, control: np.ndarray, dt: float) -> np.ndarray:
        """
        Kinematic model of differential drive robot.
        
        state = [x, y, yaw]
        control = [v, w]
        """
        x, y, yaw = state
        v, w = control
        
        # Differential drive kinematics
        x_next = x + v * np.cos(yaw) * dt
        y_next = y + v * np.sin(yaw) * dt
        yaw_next = yaw + w * dt
        
        return np.array([x_next, y_next, yaw_next])
    
    def _predict_trajectory(self, initial_state: np.ndarray, 
                           control_sequence: np.ndarray) -> np.ndarray:
        """
        Predict future trajectory given initial state and control sequence.
        """
        horizon = len(control_sequence)
        trajectory = np.zeros((horizon + 1, 3))
        trajectory[0] = initial_state
        
        for i in range(horizon):
            trajectory[i+1] = self._kinematic_model(
                trajectory[i], control_sequence[i], self.params.dt
            )
        
        return trajectory
    
    def _cost_function(self, control_sequence: np.ndarray, 
                       initial_state: np.ndarray, 
                       reference: np.ndarray) -> float:
        """
        Cost function for MPC optimization.
        
        Args:
            control_sequence: Flattened [v0, w0, v1, w1, ...]
            initial_state: Current state [x, y, yaw]
            reference: Reference trajectory [x_ref, y_ref, yaw_ref, v_ref, w_ref]
        """
        horizon = self.params.horizon
        
        # Reshape control sequence
        controls = control_sequence.reshape((horizon, 2))
        
        # Predict trajectory
        trajectory = self._predict_trajectory(initial_state, controls)
        
        # Cost components
        cost = 0.0
        
        # Reference trajectory
        x_ref = reference[:, 0]
        y_ref = reference[:, 1]
        yaw_ref = reference[:, 2]
        v_ref = reference[:, 3]
        w_ref = reference[:, 4]
        
        for i in range(horizon + 1):
            # Position error
            dx = trajectory[i, 0] - x_ref[i]
            dy = trajectory[i, 1] - y_ref[i]
            
            # Yaw error (normalized)
            dyaw = trajectory[i, 2] - yaw_ref[i]
            while dyaw > np.pi: dyaw -= 2*np.pi
            while dyaw < -np.pi: dyaw += 2*np.pi
            
            # State cost
            cost += (self.params.Q_pos * (dx**2 + dy**2) + 
                    self.params.Q_yaw * dyaw**2)
            
            # Velocity tracking (for i < horizon)
            if i < horizon:
                v_err = controls[i, 0] - v_ref[i]
                w_err = controls[i, 1] - w_ref[i]
                cost += (self.params.Q_v * v_err**2 + 
                        self.params.Q_w * w_err**2)
        
        # Control effort cost
        for i in range(horizon):
            cost += (self.params.R_v * controls[i, 0]**2 + 
                    self.params.R_w * controls[i, 1]**2)
        
        # Control change cost (smoothness)
        for i in range(1, horizon):
            dv = controls[i, 0] - controls[i-1, 0]
            dw = controls[i, 1] - controls[i-1, 1]
            cost += (self.params.R_dv * dv**2 + 
                    self.params.R_dw * dw**2)
        
        return cost
    
    def _solve_mpc(self, initial_state: np.ndarray, 
                   reference: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Solve the MPC optimization problem.
        
        Returns:
            optimal_controls: [v, w] for first step
            cost: Final cost
        """
        horizon = self.params.horizon
        
        # Initial guess: use previous controls
        if hasattr(self, '_last_controls'):
            x0 = self._last_controls.flatten()
        else:
            x0 = np.zeros(horizon * 2)
        
        # Bounds
        bounds = []
        for i in range(horizon):
            bounds.append((-self.params.max_v, self.params.max_v))  # v
            bounds.append((-self.params.max_w, self.params.max_w))  # w
        
        # Constraints for acceleration limits
        constraints = []
        
        def accel_constraint(controls):
            controls = controls.reshape((horizon, 2))
            v = controls[:, 0]
            w = controls[:, 1]
            
            # Linear acceleration constraint
            v_diff = v[1:] - v[:-1]
            max_accel_v = self.params.max_accel * self.params.dt
            constraints_v = max_accel_v - np.abs(v_diff)
            
            # Angular acceleration constraint
            w_diff = w[1:] - w[:-1]
            max_accel_w = self.params.max_ang_accel * self.params.dt
            constraints_w = max_accel_w - np.abs(w_diff)
            
            return np.concatenate([constraints_v, constraints_w])
        
        # Add acceleration constraints
        # Note: This is a simplified version; in practice you'd use NonlinearConstraint
        # For now, we'll rely on bounds and cost function
        
        # Optimize
        start_time = time.time()
        result = minimize(
            self._cost_function,
            x0,
            args=(initial_state, reference),
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 100, 'ftol': 1e-6}
        )
        self.control_time = time.time() - start_time
        
        if result.success:
            controls = result.x.reshape((horizon, 2))
            self._last_controls = controls
            return controls[0], result.fun
        else:
            print(f"⚠️ MPC optimization failed: {result.message}")
            return np.array([0.0, 0.0]), float('inf')
    
    def _create_reference(self, target_x: float, target_y: float, 
                          target_yaw: float = 0.0) -> np.ndarray:
        """
        Create reference trajectory for a single target.
        """
        horizon = self.params.horizon
        
        # Current state
        state = self.current_state
        if state is None:
            return np.zeros((horizon + 1, 5))
        
        # Simple approach: create straight line to target
        x_ref = np.linspace(state.x, target_x, horizon + 1)
        y_ref = np.linspace(state.y, target_y, horizon + 1)
        yaw_ref = np.full(horizon + 1, target_yaw)
        
        # Compute desired velocities (simple)
        v_ref = np.zeros(horizon + 1)
        w_ref = np.zeros(horizon + 1)
        
        # Store reference
        return np.column_stack([x_ref, y_ref, yaw_ref, v_ref, w_ref])
    
    # ─── Public Methods ──────────────────────────────────────────────────────
    
    def go_to_position(self, target_x: float, target_y: float, 
                       target_yaw: float = 0.0) -> Tuple[float, float]:
        """
        Compute MPC command to go to target position.
        
        Returns:
            (v, w) command
        """
        if self.current_state is None:
            return 0.0, 0.0
        
        # Create reference
        reference = self._create_reference(target_x, target_y, target_yaw)
        
        # Initial state
        initial_state = np.array([
            self.current_state.x,
            self.current_state.y,
            math.radians(self.current_state.yaw)
        ])
        
        # Solve MPC
        control, cost = self._solve_mpc(initial_state, reference)
        
        v = np.clip(control[0], -self.params.max_v, self.params.max_v)
        w = np.clip(control[1], -self.params.max_w, self.params.max_w)
        
        # Store last command
        self.last_v = v
        self.last_w = w
        self.iterations += 1
        
        return v, w
    
    def follow_trajectory(self, trajectory: MPCTrajectory) -> Tuple[float, float]:
        """
        Follow a pre-defined trajectory.
        """
        if self.current_state is None:
            return 0.0, 0.0
        
        horizon = self.params.horizon
        
        # Get reference for horizon
        x_ref = []
        y_ref = []
        yaw_ref = []
        v_ref = []
        w_ref = []
        
        for i in range(self.trajectory_index, 
                      min(self.trajectory_index + horizon + 1, len(trajectory.x))):
            x, y, yaw, v, w = trajectory.get_at_index(i)
            x_ref.append(x)
            y_ref.append(y)
            yaw_ref.append(yaw)
            v_ref.append(v)
            w_ref.append(w)
        
        # Pad if needed
        while len(x_ref) < horizon + 1:
            x_ref.append(x_ref[-1])
            y_ref.append(y_ref[-1])
            yaw_ref.append(yaw_ref[-1])
            v_ref.append(0.0)
            w_ref.append(0.0)
        
        reference = np.column_stack([x_ref, y_ref, yaw_ref, v_ref, w_ref])
        
        # Initial state
        initial_state = np.array([
            self.current_state.x,
            self.current_state.y,
            math.radians(self.current_state.yaw)
        ])
        
        # Solve MPC
        control, cost = self._solve_mpc(initial_state, reference)
        
        v = np.clip(control[0], -self.params.max_v, self.params.max_v)
        w = np.clip(control[1], -self.params.max_w, self.params.max_w)
        
        self.last_v = v
        self.last_w = w
        self.iterations += 1
        
        return v, w
    
    def track_velocity(self, target_v: float, target_w: float) -> Tuple[float, float]:
        """
        Track target velocities (v, w).
        """
        if self.current_state is None:
            return 0.0, 0.0
        
        # Create reference with constant velocities
        horizon = self.params.horizon
        x_ref = np.full(horizon + 1, self.current_state.x)
        y_ref = np.full(horizon + 1, self.current_state.y)
        yaw_ref = np.full(horizon + 1, math.radians(self.current_state.yaw))
        v_ref = np.full(horizon + 1, target_v)
        w_ref = np.full(horizon + 1, target_w)
        
        reference = np.column_stack([x_ref, y_ref, yaw_ref, v_ref, w_ref])
        
        initial_state = np.array([
            self.current_state.x,
            self.current_state.y,
            math.radians(self.current_state.yaw)
        ])
        
        control, cost = self._solve_mpc(initial_state, reference)
        
        v = np.clip(control[0], -self.params.max_v, self.params.max_v)
        w = np.clip(control[1], -self.params.max_w, self.params.max_w)
        
        self.last_v = v
        self.last_w = w
        self.iterations += 1
        
        return v, w
    
    def run_control_loop(self, target_x: float = 0.0, target_y: float = 0.0,
                        duration: float = 10.0, rate: float = 20.0) -> None:
        """
        Run the MPC control loop in real-time.
        """
        self.running = True
        period = 1.0 / rate
        
        print(f"🚀 MPC Control Loop Started")
        print(f"🎯 Target: ({target_x:.2f}, {target_y:.2f})")
        print(f"📊 Control Rate: {rate} Hz")
        print("-" * 60)
        
        start_time = time.time()
        
        while self.running and (time.time() - start_time < duration):
            loop_start = time.time()
            
            # Compute MPC command
            v, w = self.go_to_position(target_x, target_y)
            
            # Send to robot
            self.robot.set_velocity(v, w)
            
            # Print status
            if self.current_state:
                state = self.current_state
                dist = math.sqrt((state.x - target_x)**2 + (state.y - target_y)**2)
                print(f"\r📍 ({state.x:.2f}, {state.y:.2f}) → "
                      f"({target_x:.2f}, {target_y:.2f})  "
                      f"dist: {dist:.2f}m  "
                      f"v: {v:.2f} w: {w:.2f}  "
                      f"cost: {self.control_time*1000:.1f}ms", end='')
            
            # Maintain control rate
            elapsed = time.time() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)
        
        # Stop robot
        self.robot.stop()
        print("\n🛑 MPC Control Loop Finished")
        print(f"📊 Total iterations: {self.iterations}")
        print(f"📊 Avg control time: {self.control_time*1000/self.iterations:.1f}ms")
    
    def stop(self):
        """Stop the control loop"""
        self.running = False
        self.robot.stop()

# ─── Helper Functions ──────────────────────────────────────────────────────

def create_circular_trajectory(center_x: float, center_y: float, 
                               radius: float, num_points: int) -> MPCTrajectory:
    """Create a circular trajectory"""
    angles = np.linspace(0, 2*np.pi, num_points)
    
    x = center_x + radius * np.cos(angles)
    y = center_y + radius * np.sin(angles)
    yaw = angles + np.pi/2  # Tangent direction
    
    # Normalize yaw
    yaw = np.arctan2(np.sin(yaw), np.cos(yaw))
    
    # Constant velocity
    v = np.full(num_points, 0.3)
    w = np.full(num_points, 0.3 / radius)
    
    return MPCTrajectory(
        x=list(x), y=list(y), yaw=list(yaw),
        v=list(v), w=list(w)
    )

def create_line_trajectory(start_x: float, start_y: float,
                           end_x: float, end_y: float,
                           num_points: int) -> MPCTrajectory:
    """Create a straight line trajectory"""
    x = np.linspace(start_x, end_x, num_points)
    y = np.linspace(start_y, end_y, num_points)
    yaw = np.full(num_points, math.atan2(end_y - start_y, end_x - start_x))
    
    v = np.full(num_points, 0.3)
    w = np.zeros(num_points)
    
    return MPCTrajectory(
        x=list(x), y=list(y), yaw=list(yaw),
        v=list(v), w=list(w)
    )

# ─── Main Example ──────────────────────────────────────────────────────────

def main():
    """Example usage of MPC Controller"""
    import argparse
    
    parser = argparse.ArgumentParser(description="MPC Controller Test")
    parser.add_argument('--port', '-p', default='/dev/ttyUSB0', help='Serial port')
    parser.add_argument('--target-x', type=float, default=1.0, help='Target X')
    parser.add_argument('--target-y', type=float, default=1.0, help='Target Y')
    parser.add_argument('--duration', type=float, default=10.0, help='Duration (s)')
    args = parser.parse_args()
    
    # Connect to robot
    robot = RobotInterface(args.port)
    if not robot.connect():
        print("Failed to connect to robot")
        return
    
    try:
        # Create MPC controller with custom parameters
        params = MPCParams(
            horizon=10,
            dt=0.05,
            Q_pos=10.0,
            Q_yaw=5.0,
            Q_v=1.0,
            Q_w=0.5,
            R_v=0.1,
            R_w=0.05,
            R_dv=1.0,
            R_dw=0.5,
            max_v=1.2,
            max_w=2.0,
            max_accel=1.0,
            max_ang_accel=2.0
        )
        
        mpc = MPCController(robot, params)
        
        # Option 1: Go to position
        print(f"🎯 Going to ({args.target_x}, {args.target_y})")
        mpc.run_control_loop(
            target_x=args.target_x,
            target_y=args.target_y,
            duration=args.duration
        )
        
        # Option 2: Follow trajectory (uncomment to use)
        # trajectory = create_circular_trajectory(0, 0, 0.5, 20)
        # for _ in range(5):  # Do 5 laps
        #     for i in range(len(trajectory.x)):
        #         v, w = mpc.follow_trajectory(trajectory)
        #         robot.set_velocity(v, w)
        #         time.sleep(0.05)
        
        # Option 3: Track velocity (uncomment to use)
        # mpc.track_velocity(0.3, 0.0)  # Forward at 0.3 m/s
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
    finally:
        robot.disconnect()

if __name__ == "__main__":
    main()