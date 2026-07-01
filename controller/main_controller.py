"""
Main Controller - Integrates all layers with simulation/hardware switching
"""
import numpy as np
import time
import threading
from config.robot_params import RobotParams
from layers.layer1_plant_model import DifferentialDriveModel
from layers.layer3_mpc_core import MPCController
from layers.layer4_control_interface import ControlInterface
from layers.layer6_state_estimation import StateEstimator
from layers.layer7_simulation import SimulationEnvironment, HardwareInterface
from layers.layer8_visualization import RobotVisualization, PostRunAnalyzer
from layers.layer10_obstacle_detection import ObstacleDetection
from layers.layer11_data_logging import DataLogger

# ============================================
# MASTER SWITCH: Change this to switch between sim and hardware
# ============================================
USE_SIMULATION = True  # True for simulation, False for hardware
ENABLE_VISUALIZATION = False  # Set to True for real-time viz (may need main thread)

class RobotController:
    """
    Main robot controller integrating all layers
    
    This controller works IDENTICALLY in simulation and hardware
    by switching the interface layer.
    """
    
    def __init__(self, use_simulation=USE_SIMULATION, enable_viz=ENABLE_VISUALIZATION):
        self.use_simulation = use_simulation
        self.enable_viz = enable_viz
        
        # Initialize layers
        self.mpc = MPCController()
        self.control_interface = ControlInterface()
        self.state_estimator = StateEstimator()
        self.obstacle_detection = ObstacleDetection()
        self.data_logger = DataLogger()
        self._metadata_logged = False
        
        # Interface layer (simulation or hardware)
        if self.use_simulation:
            self.interface = SimulationEnvironment(enable_noise=True, enable_delay=True)
        else:
            self.interface = HardwareInterface()
        
        # Visualization (deferred to main thread)
        self.visualization = None
        if self.enable_viz:
            self.visualization = RobotVisualization(update_rate=10)
        
        # Control state
        self.current_state = np.zeros(3)
        self.waypoints = []
        self.current_waypoint_index = 0  # Track which waypoint we're heading to
        self.is_running = False
        self.control_thread = None
        
        # Monitoring
        self.control_history = []
        self.state_history = []
        self.tracking_errors = []
        self.true_states = []  # For simulation, store true states
        
        # Arrival detection
        self.arrival_threshold = 0.1  # meters - how close we need to be to waypoint
        self.waypoint_stay_counter = 0
        self.required_stay_cycles = 3  # Must be near waypoint for this many cycles
        self.waypoint_arrival_times = []
        
        # Visualization data buffer (thread-safe)
        self.viz_data = None
        self.viz_lock = threading.Lock()
        
        print(f"Robot Controller initialized ({'SIMULATION' if use_simulation else 'HARDWARE'} mode)")
        print(f"Logging to {self.data_logger.path}")
        if self.enable_viz:
            print("Visualization enabled (real-time)")
    
    def connect(self):
        """Connect to hardware or initialize simulation"""
        return self.interface.connect() if hasattr(self.interface, 'connect') else True
    
    def disconnect(self):
        """Disconnect from hardware or cleanup simulation"""
        self.is_running = False
        if self.control_thread:
            self.control_thread.join(timeout=2.0)
        
        if hasattr(self.interface, 'disconnect'):
            self.interface.disconnect()
        self.data_logger.close()
    
    def set_waypoints(self, waypoints):
        """
        Set waypoints for trajectory following
        
        Args:
            waypoints: list of (x, y) tuples
        """
        self.waypoints = waypoints.copy()
        self.current_waypoint_index = 0
        self.waypoint_stay_counter = 0
        self.waypoint_arrival_times = []
        print(f"Set {len(waypoints)} waypoints: {waypoints}")
    
    def set_initial_pose(self, x, y, theta):
        """Set initial robot pose"""
        self.current_state = np.array([x, y, theta])
        self.state_estimator.reset(self.current_state)
        
        if self.use_simulation:
            self.interface.reset(self.current_state)
        
        print(f"Initial pose set: x={x:.2f}, y={y:.2f}, theta={theta:.2f}")
    
    def start(self):
        """Start control loop"""
        if self.is_running:
            print("Controller already running")
            return
        
        self.is_running = True
        self.control_thread = threading.Thread(target=self._control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()
        print("Controller started")
    
    def stop(self):
        """Stop control loop"""
        self.is_running = False
        print("Controller stopping...")
    
    def _control_loop(self):
        """Main control loop running at MPC update rate"""
        dt = RobotParams.MPC_DT
        cycle_count = 0
        max_cycles = 2000  # Safety limit to prevent infinite loops
        
        print(f"\nStarting navigation through {len(self.waypoints)} waypoints...")
        if self.waypoints:
            print(f"Heading to waypoint 1: {self.waypoints[0]}")

        if not self._metadata_logged:
            self.data_logger.log_metadata(
                mode="simulation" if self.use_simulation else "hardware",
                initial_state=self.current_state.copy(),
                waypoints=self.waypoints,
                dt=RobotParams.MPC_DT,
                mpc_horizon=self.mpc.horizon,
                visualization_enabled=self.enable_viz
            )
            self._metadata_logged = True
        
        while self.is_running and cycle_count < max_cycles:
            loop_start = time.time()
            
            try:
                # 1. Step simulation and get feedback (simulation only)
                # For hardware, get feedback directly
                if self.use_simulation:
                    feedback = self.interface.step(dt)
                else:
                    feedback = self.interface.get_feedback()
                
                # 2. Update state estimation
                self.state_estimator.update_with_feedback(feedback)
                self.current_state = self.state_estimator.get_estimated_state()
                
                # Get true state (simulation only)
                if self.use_simulation:
                    true_state = self.interface.get_true_state()
                    self.true_states.append(true_state.copy())
                else:
                    true_state = self.current_state
                
                # 3. Check waypoint progress
                self._update_waypoint_progress()
                
                # 4. Check if we reached destination (all waypoints visited)
                if self._check_arrival():
                    print(f"\nAll waypoints reached in {cycle_count + 1} cycles!")
                    print(f"Final position: ({self.current_state[0]:.2f}, {self.current_state[1]:.2f})")
                    self.stop()
                    break
                
                # 5. Generate reference trajectory (using remaining waypoints)
                remaining_waypoints = self.waypoints[self.current_waypoint_index:]
                reference_traj = self.mpc.generate_reference_trajectory(
                    remaining_waypoints, 
                    self.current_state
                )
                
                # 6. Compute optimal control and apply obstacle safety filtering
                control_sequence = self.mpc.compute_control_sequence(self.current_state, reference_traj)
                control, obstacle_status = self.obstacle_detection.filter_control(control_sequence[0], feedback)
                control_sequence[0] = control
                predicted_traj = self.mpc.predict_trajectory(self.current_state, control_sequence)
                v_cmd, omega_cmd = control
                
                # 7. Convert to wheel commands
                rpm_left, rpm_right = self.control_interface.convert_to_wheel_commands(
                    v_cmd, omega_cmd
                )
                
                # 8. Send commands to robot/simulation
                if self.use_simulation:
                    self.interface.apply_control(rpm_left, rpm_right)
                else:
                    self.interface.send_control(rpm_left, rpm_right)
                
                # 9. Calculate tracking error to current target
                tracking_error = self._calculate_tracking_error(self.current_state)
                
                # 10. Log data
                self._log_data(cycle_count, self.current_state, control, feedback)
                self.data_logger.log_cycle(
                    cycle=cycle_count,
                    sim_time=getattr(self.interface, "sim_time", None),
                    estimated_state=self.current_state.copy(),
                    true_state=true_state.copy(),
                    feedback=feedback.copy(),
                    control=control.copy(),
                    wheel_commands={"rpm_left": rpm_left, "rpm_right": rpm_right},
                    reference_traj=reference_traj.copy(),
                    predicted_traj=predicted_traj.copy(),
                    obstacle_status=obstacle_status.copy(),
                    current_waypoint=self.current_waypoint_index,
                    target_waypoint=np.array(self.waypoints[self.current_waypoint_index]).copy()
                    if self.current_waypoint_index < len(self.waypoints) else None,
                    extra={"tracking_error": tracking_error.copy()}
                )
                
                # 11. Update visualization data buffer
                if self.enable_viz and self.visualization:
                    with self.viz_lock:
                        self.viz_data = {
                            'timestamp': time.time(),
                            'true_state': true_state.copy(),
                            'estimated_state': self.current_state.copy(),
                            'reference_traj': reference_traj.copy(),
                            'predicted_traj': predicted_traj.copy(),
                            'control': control.copy(),
                            'tracking_error': tracking_error.copy(),
                            'waypoints': remaining_waypoints,
                            'obstacle_status': obstacle_status.copy()
                        }
                
                cycle_count += 1
                
                # Print progress every 10 cycles
                if cycle_count % 10 == 0 or cycle_count == 1:
                    if self.current_waypoint_index < len(self.waypoints):
                        target_wp = self.waypoints[self.current_waypoint_index]
                        dist_to_target = np.linalg.norm(self.current_state[:2] - target_wp)
                        print(f"Cycle {cycle_count:4d}: pos=({self.current_state[0]:6.2f}, {self.current_state[1]:6.2f}), "
                              f"heading to WP{self.current_waypoint_index+1} {target_wp}, "
                              f"dist={dist_to_target:6.2f}m, "
                              f"v={v_cmd:5.2f}, w={omega_cmd:5.2f}, "
                              f"obs={obstacle_status['state']}:{obstacle_status['nearest_distance']:.2f}m")
                
                # Maintain loop rate
                elapsed = time.time() - loop_start
                sleep_time = max(0, dt - elapsed)
                time.sleep(sleep_time)
                
            except Exception as e:
                print(f"Control loop error at cycle {cycle_count}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        if cycle_count >= max_cycles:
            print(f"\nReached maximum cycles ({max_cycles}) - stopping")
            self.stop()
    
    def _update_waypoint_progress(self):
        """Check if we've reached the current target waypoint"""
        if self.current_waypoint_index >= len(self.waypoints):
            return  # All waypoints visited
        
        target_wp = np.array(self.waypoints[self.current_waypoint_index])
        distance = np.linalg.norm(self.current_state[:2] - target_wp)
        
        if distance < self.arrival_threshold:
            self.waypoint_stay_counter += 1
            
            if self.waypoint_stay_counter >= self.required_stay_cycles:
                print(f"  >> Reached waypoint {self.current_waypoint_index + 1}: {tuple(target_wp)} "
                      f"(distance={distance:.3f}m)")
                self.waypoint_arrival_times.append(time.time())
                
                # Move to next waypoint
                self.current_waypoint_index += 1
                self.waypoint_stay_counter = 0
                
                if self.current_waypoint_index < len(self.waypoints):
                    next_wp = self.waypoints[self.current_waypoint_index]
                    print(f"  >> Heading to waypoint {self.current_waypoint_index + 1}: {next_wp}")
        else:
            self.waypoint_stay_counter = max(0, self.waypoint_stay_counter - 1)
    
    def _calculate_tracking_error(self, state):
        """Calculate tracking error to current target waypoint"""
        if self.current_waypoint_index >= len(self.waypoints):
            return {'position_error': 0.0, 'heading_error': 0.0}
        
        target = np.array(self.waypoints[self.current_waypoint_index])
        position_error = np.linalg.norm(state[:2] - target)
        
        heading_to_target = np.arctan2(target[1] - state[1], target[0] - state[0])
        heading_error = self._normalize_angle(state[2] - heading_to_target)
        
        return {
            'position_error': position_error,
            'heading_error': heading_error
        }
    
    def _normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        return np.arctan2(np.sin(angle), np.cos(angle))
    
    def _check_arrival(self):
        """Check if ALL waypoints have been visited"""
        return self.current_waypoint_index >= len(self.waypoints)
    
    def _log_data(self, cycle, state, control, feedback):
        """Log control data for analysis"""
        log_entry = {
            'cycle': cycle,
            'timestamp': time.time(),
            'state': state.copy(),
            'control': control,
            'rpm_left': feedback.get('rpm_left', 0),
            'rpm_right': feedback.get('rpm_right', 0),
            'distance_left': feedback.get('distance_left', RobotParams.OBSTACLE_SENSOR_MAX_RANGE),
            'distance_center': feedback.get('distance_center', RobotParams.OBSTACLE_SENSOR_MAX_RANGE),
            'distance_right': feedback.get('distance_right', RobotParams.OBSTACLE_SENSOR_MAX_RANGE),
            'obstacle_status': self.obstacle_detection.get_status()
        }
        
        self.control_history.append(log_entry)
        self.state_history.append(state.copy())
        
        # Store tracking errors
        if self.current_waypoint_index < len(self.waypoints):
            error = self._calculate_tracking_error(state)
            self.tracking_errors.append(error)
    
    def get_status(self):
        """Get controller status"""
        return {
            'mode': 'SIMULATION' if self.use_simulation else 'HARDWARE',
            'running': self.is_running,
            'state': self.current_state.tolist(),
            'waypoints': self.waypoints,
            'current_waypoint': self.current_waypoint_index,
            'total_waypoints': len(self.waypoints),
            'cycles_completed': len(self.control_history),
            'obstacle_status': self.obstacle_detection.get_status()
        }
    
    def get_performance_metrics(self):
        """Get performance metrics"""
        if len(self.tracking_errors) == 0:
            return None
        
        position_errors = [e['position_error'] for e in self.tracking_errors]
        heading_errors = [e['heading_error'] for e in self.tracking_errors]
        
        return {
            'mean_position_error': np.mean(position_errors),
            'max_position_error': np.max(position_errors),
            'rms_position_error': np.sqrt(np.mean(np.square(position_errors))),
            'mean_heading_error': np.mean(np.abs(heading_errors)),
            'total_cycles': len(self.control_history),
            'total_distance': self._calculate_total_distance(),
            'waypoints_reached': self.current_waypoint_index,
            'total_waypoints': len(self.waypoints)
        }
    
    def _calculate_total_distance(self):
        """Calculate total distance traveled"""
        if len(self.state_history) < 2:
            return 0.0
        
        states = np.array(self.state_history)
        distances = np.sqrt(np.sum(np.diff(states[:, :2], axis=0)**2, axis=1))
        return np.sum(distances)
    
    def run_post_analysis(self, save_path=None):
        """Run post-run analysis and visualization"""
        analyzer = PostRunAnalyzer(self)
        analyzer.plot_comprehensive_analysis(save_path)


# Example usage
if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MPC Robot Controller')
    parser.add_argument('--sim', action='store_true', default=True, 
                       help='Run in simulation mode (default: True)')
    parser.add_argument('--viz', action='store_true', default=False,
                       help='Enable real-time visualization (default: False)')
    parser.add_argument('--waypoints', type=str, default='100,0;100,100;0,100;0,0',
                       help='Waypoints as semicolon-separated x,y pairs')
    parser.add_argument('--initial-pose', type=str, default='0,0,0',
                       help='Initial pose as x,y,theta')
    parser.add_argument('--max-velocity', type=float, default=1.0,
                       help='Maximum linear velocity (m/s)')
    
    args = parser.parse_args()
    
    # Parse waypoints
    waypoint_strings = args.waypoints.split(';')
    waypoints = []
    for wp_str in waypoint_strings:
        x, y = map(float, wp_str.split(','))
        waypoints.append((x, y))
    
    # Parse initial pose
    init_x, init_y, init_theta = map(float, args.initial_pose.split(','))
    
    # Override max velocity if specified
    if args.max_velocity != 1.0:
        RobotParams.MPC_MAX_VELOCITY = args.max_velocity
    
    print("="*60)
    print("MPC ROBOT CONTROLLER")
    print("="*60)
    print(f"Mode: {'SIMULATION' if args.sim else 'HARDWARE'}")
    print(f"Visualization: {'Enabled' if args.viz else 'Disabled'}")
    print(f"Max Velocity: {RobotParams.MPC_MAX_VELOCITY} m/s")
    print(f"Initial Pose: ({init_x}, {init_y}, {init_theta})")
    print(f"Waypoints: {waypoints}")
    print("="*60)
    print()
    
    # Create controller
    controller = RobotController(use_simulation=args.sim, enable_viz=args.viz)
    
    # Set initial pose
    controller.set_initial_pose(init_x, init_y, init_theta)
    
    # Set waypoints
    controller.set_waypoints(waypoints)
    
    # Connect and start
    controller.connect()
    
    # Start timing
    start_time = time.time()
    controller.start()
    
    # Wait for control to complete
    try:
        while controller.is_running:
            time.sleep(0.1)
            status = controller.get_status()
            
            # Print status update every 50 cycles
            if status['cycles_completed'] % 50 == 0 and status['cycles_completed'] > 0:
                elapsed = time.time() - start_time
                print(f"\n  Status Update at {elapsed:.1f}s:")
                print(f"    Waypoint: {status['current_waypoint']}/{status['total_waypoints']}")
                print(f"    Cycles: {status['cycles_completed']}")
                print(f"    Position: ({status['state'][0]:.2f}, {status['state'][1]:.2f})")
                print()
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        controller.stop()
    
    finally:
        elapsed_total = time.time() - start_time
        controller.disconnect()
        
        # Print performance metrics
        metrics = controller.get_performance_metrics()
        if metrics:
            print("\n" + "="*60)
            print("=== PERFORMANCE METRICS ===")
            print("="*60)
            print(f"Time elapsed:          {elapsed_total:.1f} seconds")
            print(f"Waypoints reached:     {metrics['waypoints_reached']}/{metrics['total_waypoints']}")
            print(f"Total cycles:          {metrics['total_cycles']}")
            print(f"Total distance:        {metrics['total_distance']:.2f} m")
            print(f"Mean position error:   {metrics['mean_position_error']:.3f} m")
            print(f"Max position error:    {metrics['max_position_error']:.3f} m")
            print(f"RMS position error:    {metrics['rms_position_error']:.3f} m")
            print(f"Mean heading error:    {np.degrees(metrics['mean_heading_error']):.1f} deg")
            
            # Performance grade
            if metrics['rms_position_error'] < 0.5:
                grade = "EXCELLENT"
            elif metrics['rms_position_error'] < 1.0:
                grade = "GOOD"
            elif metrics['rms_position_error'] < 2.0:
                grade = "FAIR"
            else:
                grade = "NEEDS IMPROVEMENT"
            print(f"Performance grade:     {grade}")
            
            # Waypoint timing
            if controller.waypoint_arrival_times:
                print(f"\nWaypoint arrival times:")
                for i, arrival_time in enumerate(controller.waypoint_arrival_times):
                    relative_time = arrival_time - start_time
                    print(f"  WP{i+1}: {relative_time:.1f}s")
            
            print("="*60)
        
        # Run post-run analysis
        if metrics and metrics['total_cycles'] > 5:
            print("\nGenerating post-run analysis...")
            try:
                controller.run_post_analysis(save_path="mpc_analysis.png")
                print("Analysis complete! Saved to mpc_analysis.png")
            except Exception as e:
                print(f"Could not generate analysis: {e}")
        else:
            print("\nNot enough data for post-run analysis")
