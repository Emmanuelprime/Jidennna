"""
Real-time MPC Visualization Runner - FIXED VERSION 2
Shows robot movement, waypoints, MPC predictions, and actual path
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from collections import deque
import time
import threading

from config.robot_params import RobotParams
from layers.layer1_plant_model import DifferentialDriveModel
from layers.layer3_mpc_core import MPCController
from layers.layer4_control_interface import ControlInterface
from layers.layer6_state_estimation import StateEstimator
from layers.layer7_simulation import SimulationEnvironment
from layers.layer10_obstacle_detection import ObstacleDetection
from layers.layer11_data_logging import DataLogger

class RealTimeVisualizer:
    """Real-time visualization of MPC robot control"""
    
    def __init__(self, waypoints, initial_pose=(0, 0, 0), obstacles=None):
        self.controller = RobotControllerSimple(waypoints, initial_pose, obstacles=obstacles)
        self.true_path = deque(maxlen=2000)
        self.estimated_path = deque(maxlen=2000)
        self.mpc_predictions = np.array([])
        self.reference_path = np.array([])
        self.waypoints = waypoints
        self.obstacles = obstacles or []
        self.current_waypoint_idx = 0
        self.cycle_count = 0
        self.sim_time = 0
        
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.suptitle('MPC Robot Controller - Real-Time Visualization', fontsize=14, fontweight='bold')
        
    def start(self):
        self.controller.start()
        self._run_animation()
    
    def stop(self):
        self.controller.stop()
        plt.close('all')
    
    def _run_animation(self):
        running = True
        while running and self.controller.is_running:
            try:
                data = self.controller.get_viz_data()
                
                if data:
                    self.true_path.append(data['true_state'][:2].copy())
                    self.estimated_path.append(data['estimated_state'][:2].copy())
                    
                    if len(data['predicted_traj']) > 0:
                        self.mpc_predictions = np.array(data['predicted_traj'])
                    if len(data['reference_traj']) > 0:
                        self.reference_path = np.array(data['reference_traj'])
                    
                    self.current_waypoint_idx = data['current_waypoint']
                    self.cycle_count = data['cycle_count']
                    self.sim_time = data['sim_time']
                    
                    self._update_plot(data)
                    
                    if not plt.fignum_exists(self.fig.number):
                        running = False
                        break
                
                plt.pause(0.05)
                
            except Exception as e:
                print(f"Visualization error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        
        self.controller.stop()
        print("Visualization stopped")
    
    def _update_plot(self, data):
        self.ax.clear()
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect('equal')
        
        # Plot limits
        all_x, all_y = [], []
        if self.waypoints:
            wp_arr = np.array(self.waypoints)
            all_x.extend(wp_arr[:, 0])
            all_y.extend(wp_arr[:, 1])
        if self.obstacles:
            obs_arr = np.array(self.obstacles)
            all_x.extend(obs_arr[:, 0])
            all_y.extend(obs_arr[:, 1])
        if len(self.true_path) > 0:
            tp_arr = np.array(self.true_path)
            all_x.extend(tp_arr[:, 0])
            all_y.extend(tp_arr[:, 1])
        
        if all_x:
            margin = 1.5
            x_min, x_max = min(all_x) - margin, max(all_x) + margin
            y_min, y_max = min(all_y) - margin, max(all_y) + margin
            if x_max - x_min < 8:
                cx = (x_min + x_max) / 2
                x_min, x_max = cx - 4, cx + 4
            if y_max - y_min < 8:
                cy = (y_min + y_max) / 2
                y_min, y_max = cy - 4, cy + 4
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
        
        # Waypoints
        if self.waypoints:
            wp_arr = np.array(self.waypoints)
            self.ax.plot(wp_arr[:, 0], wp_arr[:, 1], 'k--', linewidth=2, alpha=0.5, label='Goal Path')
            
            for i, wp in enumerate(self.waypoints):
                if i < self.current_waypoint_idx:
                    self.ax.scatter(wp[0], wp[1], c='green', marker='o', s=150, 
                                  edgecolors='darkgreen', linewidths=2, zorder=5)
                    self.ax.text(wp[0], wp[1] + 0.3, f'WP{i+1} (done)', fontsize=8, ha='center', color='green')
                elif i == self.current_waypoint_idx:
                    self.ax.scatter(wp[0], wp[1], c='red', marker='*', s=400, 
                                  edgecolors='darkred', linewidths=2, zorder=6)
                    self.ax.text(wp[0], wp[1] + 0.4, f'WP{i+1} (target)', fontsize=9, ha='center', color='red', fontweight='bold')
                    
                    circle = Circle(wp, self.controller.arrival_threshold, 
                                  fill=False, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
                    self.ax.add_patch(circle)
                else:
                    self.ax.scatter(wp[0], wp[1], c='gray', marker='o', s=100, 
                                  edgecolors='darkgray', linewidths=1, zorder=4)
                    self.ax.text(wp[0], wp[1] + 0.3, f'WP{i+1}', fontsize=8, ha='center', color='gray')

        if self.obstacles:
            obs_arr = np.array(self.obstacles)
            self.ax.scatter(obs_arr[:, 0], obs_arr[:, 1], c='black', marker='x', s=120,
                            linewidths=2, label='Obstacles', zorder=7)
        
        # Actual path
        if len(self.true_path) > 1:
            tp_arr = np.array(self.true_path)
            self.ax.plot(tp_arr[:, 0], tp_arr[:, 1], 'b-', linewidth=2.5, label='Actual Path', alpha=0.9, zorder=3)
            self.ax.scatter(tp_arr[0, 0], tp_arr[0, 1], c='blue', marker='s', s=80, zorder=5, label='Start')
        
        # MPC predictions
        if len(self.mpc_predictions) > 1:
            self.ax.plot(self.mpc_predictions[:, 0], self.mpc_predictions[:, 1], 
                        'orange', linewidth=2.5, label='MPC Prediction', alpha=0.8, zorder=4)
            self.ax.scatter(self.mpc_predictions[1:, 0], self.mpc_predictions[1:, 1], 
                          c='orange', marker='.', s=40, alpha=0.6, zorder=4)
        
        # Reference
        if len(self.reference_path) > 1:
            self.ax.plot(self.reference_path[:, 0], self.reference_path[:, 1], 
                        'g:', linewidth=1.5, label='Reference', alpha=0.5, zorder=2)
        
        # Estimated path
        if len(self.estimated_path) > 1:
            ep_arr = np.array(self.estimated_path)
            self.ax.plot(ep_arr[:, 0], ep_arr[:, 1], 'c--', linewidth=1, label='Estimated', alpha=0.4, zorder=2)
        
        # Robot
        if len(self.true_path) > 0 and data:
            pos = self.true_path[-1]
            theta = data['true_state'][2]
            self._draw_distance_sensors(pos[0], pos[1], theta, data.get('obstacle_status'))
            self._draw_robot(pos[0], pos[1], theta)
        
        # Info text
        info = f"Time: {self.sim_time:.1f}s | Cycle: {self.cycle_count}\n"
        info += f"Waypoint: {self.current_waypoint_idx + 1}/{len(self.waypoints)}\n"
        if data:
            info += f"v={data['control'][0]:.3f} m/s | w={data['control'][1]:.3f} rad/s\n"
            info += f"Pos: ({data['true_state'][0]:.2f}, {data['true_state'][1]:.2f})\n"
            info += f"Heading: {np.degrees(data['true_state'][2]):.0f} deg\n"
            obstacle_status = data.get('obstacle_status')
            if obstacle_status:
                distances = obstacle_status['distances']
                info += (
                    f"Obs: {obstacle_status['state']} "
                    f"L={distances['left']:.2f} C={distances['center']:.2f} R={distances['right']:.2f}\n"
                )
            if self.current_waypoint_idx < len(self.waypoints):
                target = np.array(self.waypoints[self.current_waypoint_idx])
                dist = np.linalg.norm(data['true_state'][:2] - target)
                bearing = np.arctan2(target[1] - data['true_state'][1], target[0] - data['true_state'][0])
                bearing_error = np.degrees(self.controller._normalize_angle(bearing - data['true_state'][2]))
                info += f"Dist: {dist:.2f}m | Bearing err: {bearing_error:.0f} deg"
        
        self.ax.text(0.02, 0.98, info, transform=self.ax.transAxes,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='gray', alpha=0.9),
                    fontsize=9, zorder=10)
        
        self.ax.legend(loc='lower right', fontsize=9, framealpha=0.8)
        status = "MOVING" if self.controller.is_running else "ARRIVED"
        self.ax.set_title(f'MPC Control | {status} | WP {self.current_waypoint_idx + 1}/{len(self.waypoints)}', 
                         fontweight='bold', fontsize=13)
        
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def _draw_robot(self, x, y, theta, scale=0.6):
        length = 0.4 * scale
        width = 0.25 * scale
        
        robot = Rectangle((x - length/2, y - width/2), length, width,
                         angle=np.degrees(theta), rotation_point='center',
                         facecolor='royalblue', edgecolor='darkblue',
                         linewidth=2, alpha=0.9, zorder=10)
        self.ax.add_patch(robot)
        
        al = 0.3 * scale
        self.ax.arrow(x, y, al*np.cos(theta), al*np.sin(theta),
                     head_width=0.1, head_length=0.1, fc='red', ec='darkred',
                     linewidth=2, alpha=0.9, zorder=11)
        
        wo = width * 0.6
        wl = 0.12 * scale
        lwx, lwy = x - wo*np.sin(theta), y + wo*np.cos(theta)
        self.ax.plot([lwx - wl/2*np.cos(theta), lwx + wl/2*np.cos(theta)],
                    [lwy - wl/2*np.sin(theta), lwy + wl/2*np.sin(theta)],
                    'k-', linewidth=4, zorder=11)
        rwx, rwy = x + wo*np.sin(theta), y - wo*np.cos(theta)
        self.ax.plot([rwx - wl/2*np.cos(theta), rwx + wl/2*np.cos(theta)],
                    [rwy - wl/2*np.sin(theta), rwy + wl/2*np.sin(theta)],
                    'k-', linewidth=4, zorder=11)
        
        self.ax.scatter(x, y, c='yellow', marker='o', s=40, edgecolors='black', linewidths=1, zorder=12)

    def _draw_distance_sensors(self, x, y, theta, obstacle_status):
        if not obstacle_status:
            return

        distances = obstacle_status['distances']
        sensor_specs = [
            ('left', np.deg2rad(35), 'tab:purple'),
            ('center', 0.0, 'tab:red'),
            ('right', np.deg2rad(-35), 'tab:olive'),
        ]

        for name, offset, color in sensor_specs:
            distance = distances[name]
            angle = theta + offset
            end_x = x + distance * np.cos(angle)
            end_y = y + distance * np.sin(angle)

            self.ax.plot([x, end_x], [y, end_y], color=color, linewidth=2,
                         alpha=0.75, zorder=8, label=f'{name.capitalize()} sensor')
            self.ax.scatter(end_x, end_y, c=color, marker='.', s=60, zorder=9)


class RobotControllerSimple:
    """Controller using MPC with trajectory visualization"""
    
    def __init__(self, waypoints, initial_pose=(0, 0, 0), obstacles=None):
        self.mpc = MPCController(horizon=15)
        self.mpc.Q = np.diag([28.0, 28.0, 12.0])
        self.mpc.R = np.diag([0.35, 0.12])
        self.control_interface = ControlInterface()
        self.state_estimator = StateEstimator()
        self.obstacle_detection = ObstacleDetection()
        self.data_logger = DataLogger()
        self.interface = SimulationEnvironment(enable_noise=True, enable_delay=True)
        if obstacles:
            self.interface.set_obstacles(obstacles)
        self.model = DifferentialDriveModel()
        
        self.current_state = np.array(initial_pose, dtype=float)
        self.waypoints = list(waypoints)
        self.current_waypoint_index = 0
        self.is_running = False
        self.control_thread = None
        
        self.arrival_threshold = 0.15
        self.fallback_arrival_threshold = 0.30
        self.waypoint_stay_counter = 0
        self.required_stay_cycles = 4
        self.stuck_accept_cycles = 25
        self.best_waypoint_distance = float('inf')
        self.no_progress_cycles = 0
        self.reference_speed = 0.7
        self.approach_slow_radius = 1.4
        self.min_approach_speed = 0.18
        self.close_tracking_radius = 0.7
        
        self.viz_data = None
        self.viz_lock = threading.Lock()
        self.cycle_count = 0
        self.true_states = []
        
        self.interface.reset(self.current_state)
        self.state_estimator.reset(self.current_state)
        self.data_logger.log_metadata(
            mode="visualization",
            initial_pose=self.current_state.copy(),
            waypoints=self.waypoints,
            obstacles=obstacles or [],
            dt=RobotParams.MPC_DT,
            mpc_horizon=self.mpc.horizon
        )
        
        print(f"Controller initialized with {len(waypoints)} waypoints")
        print(f"Logging to {self.data_logger.path}")
    
    def start(self):
        self.is_running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        print("Controller started")
    
    def stop(self):
        self.is_running = False
        if self.control_thread:
            self.control_thread.join(timeout=1.0)
        self.data_logger.close()
        print("Controller stopped")
    
    def _generate_simple_reference(self, state, remaining_waypoints):
        """
        Generate a simple reference trajectory for visualization.
        Returns Nx3 array going from current state toward the waypoints.
        """
        horizon = 10
        dt = RobotParams.MPC_DT
        max_v = RobotParams.MPC_MAX_VELOCITY
        
        trajectory = []
        current_pos = state[:2].copy()
        current_heading = state[2]
        
        # Convert remaining waypoints to numpy array
        if remaining_waypoints:
            wp_array = np.array(remaining_waypoints)
        else:
            wp_array = np.array([current_pos])
        
        wp_idx = 0
        for i in range(horizon):
            if wp_idx < len(wp_array):
                target = wp_array[wp_idx]
                
                # Move toward target at max velocity
                direction = target - current_pos
                dist = np.linalg.norm(direction)
                
                if dist < 0.01:
                    wp_idx += 1
                    if wp_idx < len(wp_array):
                        target = wp_array[wp_idx]
                        direction = target - current_pos
                        dist = np.linalg.norm(direction)
                
                if dist > 0:
                    step_size = min(max_v * dt, dist)
                    current_pos = current_pos + (direction / dist) * step_size
                    desired_heading = np.arctan2(direction[1], direction[0])
                else:
                    desired_heading = current_heading
                
                trajectory.append([current_pos[0], current_pos[1], desired_heading])
            else:
                trajectory.append([current_pos[0], current_pos[1], current_heading])
        
        return np.array(trajectory)
    
    def _control_loop(self):
        """Main control loop using MPC controller"""
        dt = RobotParams.MPC_DT
        
        while self.is_running:
            try:
                # Step simulation
                feedback = self.interface.step(dt)
                self.state_estimator.update_with_feedback(feedback)
                self.current_state = self.state_estimator.get_estimated_state()
                true_state = self.interface.get_true_state()
                self.true_states.append(true_state.copy())
                
                # Check waypoint progress
                self._update_waypoint_progress()
                
                if self.current_waypoint_index >= len(self.waypoints):
                    print(f"\nAll waypoints reached in {self.cycle_count} cycles!")
                    self.is_running = False
                    break
                
                target = np.array(self.waypoints[self.current_waypoint_index])
                remaining = [list(wp) for wp in self.waypoints[self.current_waypoint_index:]]
                reference_traj = self._generate_tracking_reference(remaining, self.current_state)
                control_sequence = self.mpc.compute_control_sequence(self.current_state, reference_traj)
                predicted_traj = self.mpc.predict_trajectory(self.current_state, control_sequence)

                control = self._close_range_waypoint_control(control_sequence[0], target)
                safe_control, obstacle_status = self.obstacle_detection.filter_control(control, feedback)
                safe_control = self._limit_speed_near_waypoint(safe_control, target)
                control_sequence[0] = safe_control
                predicted_traj = self.mpc.predict_trajectory(self.current_state, control_sequence)
                v_cmd, omega_cmd = safe_control
                
                # Apply control
                rpm_left, rpm_right = self.control_interface.convert_to_wheel_commands(v_cmd, omega_cmd)
                self.interface.apply_control(rpm_left, rpm_right)
                self.data_logger.log_cycle(
                    cycle=self.cycle_count,
                    sim_time=self.interface.sim_time,
                    estimated_state=self.current_state.copy(),
                    true_state=true_state.copy(),
                    feedback=feedback.copy(),
                    control=np.array([v_cmd, omega_cmd]),
                    wheel_commands={"rpm_left": rpm_left, "rpm_right": rpm_right},
                    reference_traj=reference_traj.copy(),
                    predicted_traj=predicted_traj.copy(),
                    obstacle_status=obstacle_status.copy(),
                    current_waypoint=self.current_waypoint_index,
                    target_waypoint=target.copy(),
                    extra={
                        "arrival_threshold": self.arrival_threshold,
                        "fallback_arrival_threshold": self.fallback_arrival_threshold,
                        "best_waypoint_distance": self.best_waypoint_distance,
                        "no_progress_cycles": self.no_progress_cycles
                    }
                )
                
                # Update viz data
                with self.viz_lock:
                    self.viz_data = {
                        'true_state': true_state.copy(),
                        'estimated_state': self.current_state.copy(),
                        'predicted_traj': predicted_traj.copy(),
                        'reference_traj': reference_traj.copy(),
                        'control': np.array([v_cmd, omega_cmd]),
                        'obstacle_status': obstacle_status.copy(),
                        'current_waypoint': self.current_waypoint_index,
                        'cycle_count': self.cycle_count,
                        'sim_time': self.interface.sim_time
                    }
                
                self.cycle_count += 1
                
                if self.cycle_count % 20 == 0:
                    if self.current_waypoint_index < len(self.waypoints):
                        target = np.array(self.waypoints[self.current_waypoint_index])
                        dist = np.linalg.norm(true_state[:2] - target)
                        desired_heading = np.arctan2(
                            target[1] - self.current_state[1],
                            target[0] - self.current_state[0]
                        )
                        heading_error = self._normalize_angle(desired_heading - self.current_state[2])
                        print(f"Cycle {self.cycle_count}: pos=({true_state[0]:.2f}, {true_state[1]:.2f}), "
                              f"WP{self.current_waypoint_index+1} ({target[0]:.1f},{target[1]:.1f}), "
                              f"dist={dist:.2f}m, h_err={np.degrees(heading_error):.0f}deg, "
                              f"v={v_cmd:.2f}, w={omega_cmd:.2f}, "
                              f"obs={obstacle_status['state']}:{obstacle_status['nearest_distance']:.2f}m")
                
                time.sleep(dt)
                
            except Exception as e:
                print(f"Control error at cycle {self.cycle_count}: {e}")
                import traceback
                traceback.print_exc()
                break
    
    def _update_waypoint_progress(self):
        if self.current_waypoint_index >= len(self.waypoints):
            return
        
        target = np.array(self.waypoints[self.current_waypoint_index])
        distance = np.linalg.norm(self.current_state[:2] - target)
        reached_precisely = distance < self.arrival_threshold

        if distance < self.best_waypoint_distance - 0.01:
            self.best_waypoint_distance = distance
            self.no_progress_cycles = 0
        else:
            self.no_progress_cycles += 1

        stuck_near_waypoint = (
            distance < self.fallback_arrival_threshold and
            self.no_progress_cycles >= self.stuck_accept_cycles
        )
        
        if reached_precisely or stuck_near_waypoint:
            self.waypoint_stay_counter += 1
            if self.waypoint_stay_counter >= self.required_stay_cycles:
                reason = "precise" if reached_precisely else "near/stalled"
                print(f"  >> Reached WP{self.current_waypoint_index + 1}: ({target[0]:.1f}, {target[1]:.1f}) "
                      f"[{reason}, dist={distance:.2f}m]")
                self.current_waypoint_index += 1
                self.waypoint_stay_counter = 0
                self.best_waypoint_distance = float('inf')
                self.no_progress_cycles = 0
                if self.current_waypoint_index < len(self.waypoints):
                    next_wp = self.waypoints[self.current_waypoint_index]
                    print(f"  >> Heading to WP{self.current_waypoint_index + 1}: {next_wp}")
        else:
            self.waypoint_stay_counter = max(0, self.waypoint_stay_counter - 1)

    def _generate_tracking_reference(self, remaining_waypoints, state):
        trajectory = []
        current_pos = state[:2].copy()
        heading = state[2]
        waypoint_idx = 0

        for _ in range(self.mpc.horizon):
            if waypoint_idx >= len(remaining_waypoints):
                trajectory.append([current_pos[0], current_pos[1], heading])
                continue

            target = np.array(remaining_waypoints[waypoint_idx], dtype=float)
            direction = target - current_pos
            distance = np.linalg.norm(direction)

            while distance < self.arrival_threshold and waypoint_idx < len(remaining_waypoints) - 1:
                current_pos = target
                waypoint_idx += 1
                target = np.array(remaining_waypoints[waypoint_idx], dtype=float)
                direction = target - current_pos
                distance = np.linalg.norm(direction)

            if distance > 1e-6:
                step = min(self.reference_speed * RobotParams.MPC_DT, distance)
                current_pos = current_pos + (direction / distance) * step
                heading = np.arctan2(direction[1], direction[0])

            trajectory.append([current_pos[0], current_pos[1], heading])

        return np.array(trajectory)

    def _limit_speed_near_waypoint(self, control, target):
        distance = np.linalg.norm(self.current_state[:2] - target)
        speed_scale = np.clip(distance / self.approach_slow_radius, 0.0, 1.0)
        max_speed = max(self.min_approach_speed, RobotParams.MPC_MAX_VELOCITY * speed_scale)

        limited = np.asarray(control, dtype=float).copy()
        limited[0] = np.clip(limited[0], -max_speed, max_speed)
        return limited

    def _close_range_waypoint_control(self, mpc_control, target):
        target = np.asarray(target, dtype=float)
        delta = target - self.current_state[:2]
        distance = np.linalg.norm(delta)

        if distance > self.close_tracking_radius:
            return mpc_control

        desired_heading = np.arctan2(delta[1], delta[0])
        heading_error = self._normalize_angle(desired_heading - self.current_state[2])

        omega = np.clip(2.8 * heading_error, -RobotParams.MPC_MAX_ANGULAR_VEL, RobotParams.MPC_MAX_ANGULAR_VEL)

        if abs(heading_error) > 0.9:
            v = 0.0
        else:
            v = np.clip(0.55 * distance * np.cos(heading_error), 0.06, 0.25)

        return np.array([v, omega])
    
    def _normalize_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def get_viz_data(self):
        with self.viz_lock:
            return self.viz_data.copy() if self.viz_data else None


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    waypoints = [
       (4, 0),
       (8, 2),
       (12, 2),
       (16, 0)
    ]

    obstacles = [
       (1.3, 0.0),
       (4.0, 0.45),
       (7.0, 1.35),
       (10.0, 2.0),
       (13.5, 1.0)
    ]
    
    print("="*60)
    print("MPC REAL-TIME VISUALIZATION")
    print("="*60)
    print(f"Waypoints: {waypoints}")
    print(f"Obstacles: {obstacles}")
    print("Starting visualization...")
    print("="*60)
    
    viz = RealTimeVisualizer(waypoints, initial_pose=(0, 0, 0), obstacles=obstacles)
    
    try:
        viz.start()
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        viz.stop()
        print("Done!")
