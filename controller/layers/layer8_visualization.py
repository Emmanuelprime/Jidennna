"""
Layer 8: Visualization Layer - Real-time and post-run visualization for MPC system
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import deque
import threading
import time

class RobotVisualization:
    """
    Real-time visualization of robot state, trajectory, and MPC predictions
    """
    
    def __init__(self, update_rate=10, history_length=100):
        self.update_rate = update_rate
        self.history_length = history_length
        
        # Data storage
        self.true_states = deque(maxlen=history_length)
        self.estimated_states = deque(maxlen=history_length)
        self.reference_trajectory = deque(maxlen=50)
        self.predicted_trajectory = deque(maxlen=50)
        self.control_commands = deque(maxlen=history_length)
        self.tracking_errors = deque(maxlen=history_length)
        
        # Waypoints
        self.waypoints = []
        
        # Visualization state
        self.fig = None
        self.axes = None
        
        # Robot dimensions for drawing
        self.robot_length = 0.5
        self.robot_width = 0.3
    
    def _update_plots(self):
        """Update all visualization plots"""
        if len(self.true_states) == 0:
            return
        
        # Clear all axes
        for ax in self.axes.flat:
            ax.clear()
        
        # Plot 1: Top-down trajectory view
        self._plot_trajectory_view(self.axes[0, 0])
        
        # Plot 2: State time history
        self._plot_state_history(self.axes[0, 1])
        
        # Plot 3: Control commands
        self._plot_control_history(self.axes[1, 0])
        
        # Plot 4: Tracking errors
        self._plot_tracking_errors(self.axes[1, 1])
        
        # Plot 5: Velocity profile
        self._plot_velocity_profile(self.axes[2, 0])
        
        # Plot 6: MPC cost and solver info
        self._plot_mpc_info(self.axes[2, 1])
        
        # Update figure
        self.fig.suptitle('MPC Robot Controller - Real-time Visualization', fontsize=14)
        self.fig.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def _plot_trajectory_view(self, ax):
        """Plot 1: Top-down view of robot trajectory"""
        ax.set_title('Robot Trajectory (Top View)')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Plot waypoints
        if self.waypoints:
            waypoints_array = np.array(self.waypoints)
            ax.plot(waypoints_array[:, 0], waypoints_array[:, 1], 
                   'k--', linewidth=1, alpha=0.5, label='Waypoints')
            ax.scatter(waypoints_array[:, 0], waypoints_array[:, 1], 
                      c='black', marker='x', s=50, alpha=0.7)
            
            # Connect waypoints with arrows
            for i in range(len(self.waypoints) - 1):
                ax.annotate('', xy=self.waypoints[i+1], xytext=self.waypoints[i],
                           arrowprops=dict(arrowstyle='->', color='gray', 
                                         alpha=0.5, lw=1))
        
        # Plot true trajectory
        true_states = np.array(self.true_states)
        ax.plot(true_states[:, 0], true_states[:, 1], 
               'b-', linewidth=2, label='True Trajectory', alpha=0.8)
        
        # Plot estimated trajectory
        if len(self.estimated_states) > 0:
            est_states = np.array(self.estimated_states)
            ax.plot(est_states[:, 0], est_states[:, 1], 
                   'r--', linewidth=1.5, label='Estimated Trajectory', alpha=0.6)
        
        # Plot reference trajectory
        if len(self.reference_trajectory) > 0:
            ref_traj = np.array(self.reference_trajectory)
            ax.plot(ref_traj[:, 0], ref_traj[:, 1], 
                   'g:', linewidth=1.5, label='Reference', alpha=0.6)
        
        # Plot MPC predicted trajectory
        if len(self.predicted_trajectory) > 0:
            pred_traj = np.array(self.predicted_trajectory)
            ax.plot(pred_traj[:, 0], pred_traj[:, 1], 
                   'orange', linewidth=1, label='MPC Prediction', alpha=0.5)
        
        # Draw current robot position
        if len(self.true_states) > 0:
            current_state = self.true_states[-1]
            self._draw_robot(ax, current_state[0], current_state[1], current_state[2])
        
        # Set limits with some padding
        if len(self.true_states) > 0:
            x_min, x_max = true_states[:, 0].min() - 0.5, true_states[:, 0].max() + 0.5
            y_min, y_max = true_states[:, 1].min() - 0.5, true_states[:, 1].max() + 0.5
            
            # Include waypoints in limits
            if self.waypoints:
                wp_array = np.array(self.waypoints)
                x_min = min(x_min, wp_array[:, 0].min() - 0.5)
                x_max = max(x_max, wp_array[:, 0].max() + 0.5)
                y_min = min(y_min, wp_array[:, 1].min() - 0.5)
                y_max = max(y_max, wp_array[:, 1].max() + 0.5)
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
        ax.legend(loc='upper right', fontsize=8)
    
    def _draw_robot(self, ax, x, y, theta, scale=0.3):
        """Draw robot as a oriented rectangle"""
        # Robot body
        robot = Rectangle(
            (x - self.robot_length/2 * scale, y - self.robot_width/2 * scale),
            self.robot_length * scale,
            self.robot_width * scale,
            angle=np.degrees(theta),
            rotation_point='center',
            facecolor='blue',
            edgecolor='darkblue',
            alpha=0.8,
            label='Robot'
        )
        ax.add_patch(robot)
        
        # Direction indicator
        arrow_length = 0.15 * scale
        dx = arrow_length * np.cos(theta)
        dy = arrow_length * np.sin(theta)
        ax.arrow(x, y, dx, dy, 
                head_width=0.05, head_length=0.05, 
                fc='red', ec='red', alpha=0.8)
        
        # Wheels
        wheel_offset = self.robot_width/2 * 1.2 * scale
        wheel_length = 0.1 * scale
        
        # Left wheel
        left_wheel_x = x - wheel_offset * np.sin(theta)
        left_wheel_y = y + wheel_offset * np.cos(theta)
        ax.plot([left_wheel_x - wheel_length/2 * np.cos(theta),
                 left_wheel_x + wheel_length/2 * np.cos(theta)],
                [left_wheel_y - wheel_length/2 * np.sin(theta),
                 left_wheel_y + wheel_length/2 * np.sin(theta)],
                'k-', linewidth=3)
        
        # Right wheel
        right_wheel_x = x + wheel_offset * np.sin(theta)
        right_wheel_y = y - wheel_offset * np.cos(theta)
        ax.plot([right_wheel_x - wheel_length/2 * np.cos(theta),
                 right_wheel_x + wheel_length/2 * np.cos(theta)],
                [right_wheel_y - wheel_length/2 * np.sin(theta),
                 right_wheel_y + wheel_length/2 * np.sin(theta)],
                'k-', linewidth=3)
    
    def _plot_state_history(self, ax):
        """Plot 2: State variables over time"""
        ax.set_title('State Variables')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('State Value')
        ax.grid(True, alpha=0.3)
        
        if len(self.true_states) == 0:
            return
        
        states = np.array(self.true_states)
        time_steps = np.arange(len(states))
        
        # Plot each state component
        ax.plot(time_steps, states[:, 0], 'b-', label='X Position (m)', linewidth=1.5)
        ax.plot(time_steps, states[:, 1], 'r-', label='Y Position (m)', linewidth=1.5)
        ax.plot(time_steps, states[:, 2], 'g-', label='Heading (rad)', linewidth=1.5)
        
        # Plot estimated states for comparison
        if len(self.estimated_states) > 0:
            est_states = np.array(self.estimated_states)
            ax.plot(time_steps[:len(est_states)], est_states[:, 0], 'b--', alpha=0.4)
            ax.plot(time_steps[:len(est_states)], est_states[:, 1], 'r--', alpha=0.4)
            ax.plot(time_steps[:len(est_states)], est_states[:, 2], 'g--', alpha=0.4)
        
        ax.legend(loc='best', fontsize=8)
    
    def _plot_control_history(self, ax):
        """Plot 3: Control commands over time"""
        ax.set_title('Control Commands')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Control Value')
        ax.grid(True, alpha=0.3)
        
        if len(self.control_commands) == 0:
            return
        
        controls = np.array(self.control_commands)
        time_steps = np.arange(len(controls))
        
        # Plot velocity commands
        ax.plot(time_steps, controls[:, 0], 'b-', label='Linear Velocity (m/s)', linewidth=1.5)
        ax.plot(time_steps, controls[:, 1], 'r-', label='Angular Velocity (rad/s)', linewidth=1.5)
        
        # Add max velocity limits
        ax.axhline(y=1.0, color='b', linestyle='--', alpha=0.3, label='Max v')
        ax.axhline(y=-1.0, color='b', linestyle='--', alpha=0.3)
        ax.axhline(y=2.0, color='r', linestyle='--', alpha=0.3, label='Max w')
        ax.axhline(y=-2.0, color='r', linestyle='--', alpha=0.3)
        
        ax.legend(loc='best', fontsize=8)
    
    def _plot_tracking_errors(self, ax):
        """Plot 4: Tracking errors over time"""
        ax.set_title('Tracking Errors')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Error')
        ax.grid(True, alpha=0.3)
        
        if len(self.tracking_errors) == 0:
            return
        
        errors = list(self.tracking_errors)
        time_steps = np.arange(len(errors))
        
        position_errors = [e['position_error'] for e in errors]
        heading_errors = [e['heading_error'] for e in errors]
        
        ax.plot(time_steps, position_errors, 'b-', label='Position Error (m)', linewidth=1.5)
        ax.plot(time_steps, heading_errors, 'r-', label='Heading Error (rad)', linewidth=1.5)
        
        # Add statistics
        if len(position_errors) > 0:
            rms_pos = np.sqrt(np.mean(np.square(position_errors)))
            ax.text(0.02, 0.98, f'RMS Pos: {rms_pos:.3f}m',
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=8)
        
        ax.legend(loc='upper right', fontsize=8)
    
    def _plot_velocity_profile(self, ax):
        """Plot 5: Robot velocity profile"""
        ax.set_title('Velocity Profile')
        ax.set_xlabel('Linear Velocity (m/s)')
        ax.set_ylabel('Angular Velocity (rad/s)')
        ax.grid(True, alpha=0.3)
        
        if len(self.control_commands) == 0:
            return
        
        controls = np.array(self.control_commands)
        
        # Scatter plot of velocity combinations
        sc = ax.scatter(controls[:, 0], controls[:, 1], 
                       c=np.arange(len(controls)), cmap='viridis', 
                       alpha=0.6, s=30)
        
        # Add colorbar to show time progression
        plt.colorbar(sc, ax=ax, label='Time Step')
        
        # Mark start and end
        if len(controls) > 1:
            ax.scatter(controls[0, 0], controls[0, 1], 
                      c='green', s=100, marker='o', label='Start', edgecolors='black')
            ax.scatter(controls[-1, 0], controls[-1, 1], 
                      c='red', s=100, marker='s', label='End', edgecolors='black')
        
        # Add velocity limits
        ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.3)
        ax.axvline(x=-1.0, color='gray', linestyle='--', alpha=0.3)
        ax.axhline(y=2.0, color='gray', linestyle='--', alpha=0.3)
        ax.axhline(y=-2.0, color='gray', linestyle='--', alpha=0.3)
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-2.2, 2.2)
        ax.legend(fontsize=8)
    
    def _plot_mpc_info(self, ax):
        """Plot 6: MPC performance information"""
        ax.set_title('MPC Performance')
        ax.axis('off')
        
        # Create text-based dashboard
        info_text = "=== MPC Controller Status ===\n\n"
        
        if len(self.tracking_errors) > 0:
            errors = list(self.tracking_errors)
            recent_errors = errors[-10:]  # Last 10 errors
            
            pos_errors = [e['position_error'] for e in recent_errors]
            head_errors = [e['heading_error'] for e in recent_errors]
            
            info_text += "Recent Performance (last {} steps):\n".format(len(recent_errors))
            info_text += "  Position Error: {:.3f} m (mean)\n".format(np.mean(pos_errors))
            info_text += "  Heading Error:  {:.3f} rad (mean abs)\n\n".format(np.mean(np.abs(head_errors)))
        
        if len(self.control_commands) > 0:
            recent_controls = np.array(self.control_commands)[-10:]
            info_text += "Control Commands:\n"
            info_text += "  Linear vel:  {:.3f} m/s (mean)\n".format(np.mean(recent_controls[:, 0]))
            info_text += "  Angular vel: {:.3f} rad/s (mean)\n\n".format(np.mean(recent_controls[:, 1]))
        
        if len(self.true_states) > 0:
            current_state = self.true_states[-1]
            info_text += "Current State:\n"
            info_text += "  X: {:.3f} m\n".format(current_state[0])
            info_text += "  Y: {:.3f} m\n".format(current_state[1])
            info_text += "  Theta: {:.1f} deg\n\n".format(np.degrees(current_state[2]))
        
        if self.waypoints:
            info_text += "Waypoints: {}\n".format(len(self.waypoints))
            if len(self.true_states) > 0 and len(self.waypoints) > 0:
                current_pos = self.true_states[-1][:2]
                distances = [np.linalg.norm(current_pos - wp) for wp in self.waypoints]
                min_dist = min(distances)
                info_text += "  Distance to nearest: {:.3f} m\n".format(min_dist)
        
        ax.text(0.1, 0.9, info_text, transform=ax.transAxes,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3),
               fontsize=9)


class PostRunAnalyzer:
    """
    Post-run analysis and visualization of MPC performance
    """
    
    def __init__(self, controller):
        self.controller = controller
        
    def plot_comprehensive_analysis(self, save_path=None):
        """Generate comprehensive post-run analysis plots"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        fig.suptitle('MPC Controller - Post-Run Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Trajectory tracking
        self._plot_trajectory_tracking(axes[0, 0])
        
        # Plot 2: Position error over time
        self._plot_position_error(axes[0, 1])
        
        # Plot 3: Heading error over time
        self._plot_heading_error(axes[0, 2])
        
        # Plot 4: Velocity commands
        self._plot_velocity_commands(axes[1, 0])
        
        # Plot 5: Wheel RPMs
        self._plot_wheel_rpms(axes[1, 1])
        
        # Plot 6: Error distribution
        self._plot_error_distribution(axes[1, 2])
        
        # Plot 7: Control effort analysis
        self._plot_control_effort(axes[2, 0])
        
        # Plot 8: Path curvature
        self._plot_path_curvature(axes[2, 1])
        
        # Plot 9: Performance summary
        self._plot_performance_summary(axes[2, 2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Analysis saved to {save_path}")
        
        plt.show()
    
    def _plot_trajectory_tracking(self, ax):
        """Plot actual vs reference trajectory"""
        ax.set_title('Trajectory Tracking', fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Plot reference waypoints
        if self.controller.waypoints:
            waypoints = np.array(self.controller.waypoints)
            ax.plot(waypoints[:, 0], waypoints[:, 1], 
                   'k--', linewidth=2, label='Reference Path')
            ax.scatter(waypoints[:, 0], waypoints[:, 1], 
                      c='black', marker='s', s=100, zorder=5, label='Waypoints')
        
        # Plot actual trajectory
        if self.controller.state_history:
            states = np.array(self.controller.state_history)
            ax.plot(states[:, 0], states[:, 1], 
                   'b-', linewidth=2, label='Actual Path', alpha=0.8)
            ax.scatter(states[0, 0], states[0, 1], 
                      c='green', marker='o', s=150, zorder=5, label='Start')
            ax.scatter(states[-1, 0], states[-1, 1], 
                      c='red', marker='*', s=150, zorder=5, label='End')
        
        ax.legend(fontsize=9)
    
    def _plot_position_error(self, ax):
        """Plot position error over time"""
        ax.set_title('Position Tracking Error', fontweight='bold')
        ax.set_xlabel('Control Cycle')
        ax.set_ylabel('Error (m)')
        ax.grid(True, alpha=0.3)
        
        if self.controller.tracking_errors:
            pos_errors = [e['position_error'] for e in self.controller.tracking_errors]
            cycles = np.arange(len(pos_errors))
            
            ax.plot(cycles, pos_errors, 'b-', linewidth=1.5)
            ax.fill_between(cycles, 0, pos_errors, alpha=0.2)
            
            # Add statistics
            mean_err = np.mean(pos_errors)
            rms_err = np.sqrt(np.mean(np.square(pos_errors)))
            
            ax.axhline(y=mean_err, color='r', linestyle='--', alpha=0.7, 
                      label='Mean: {:.3f}m'.format(mean_err))
            ax.axhline(y=rms_err, color='orange', linestyle='--', alpha=0.7, 
                      label='RMS: {:.3f}m'.format(rms_err))
            
            ax.legend(fontsize=9)
    
    def _plot_heading_error(self, ax):
        """Plot heading error over time"""
        ax.set_title('Heading Tracking Error', fontweight='bold')
        ax.set_xlabel('Control Cycle')
        ax.set_ylabel('Error (rad)')
        ax.grid(True, alpha=0.3)
        
        if self.controller.tracking_errors:
            head_errors = [e['heading_error'] for e in self.controller.tracking_errors]
            cycles = np.arange(len(head_errors))
            
            ax.plot(cycles, head_errors, 'r-', linewidth=1.5)
            ax.fill_between(cycles, 0, head_errors, alpha=0.2, color='red')
            
            mean_err = np.mean(np.abs(head_errors))
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.text(0.02, 0.98, 'Mean |error|: {:.3f} rad = {:.1f} deg'.format(
                   mean_err, np.degrees(mean_err)),
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)
    
    def _plot_velocity_commands(self, ax):
        """Plot velocity commands over time"""
        ax.set_title('Control Commands History', fontweight='bold')
        ax.set_xlabel('Control Cycle')
        ax.set_ylabel('Command Value')
        ax.grid(True, alpha=0.3)
        
        if self.controller.control_history:
            controls = np.array([c['control'] for c in self.controller.control_history])
            cycles = np.arange(len(controls))
            
            ax.plot(cycles, controls[:, 0], 'b-', label='v (m/s)', linewidth=1.5)
            ax.plot(cycles, controls[:, 1], 'r-', label='w (rad/s)', linewidth=1.5)
            
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax.legend(fontsize=9)
    
    def _plot_wheel_rpms(self, ax):
        """Plot wheel RPMs over time"""
        ax.set_title('Wheel RPM Commands', fontweight='bold')
        ax.set_xlabel('Control Cycle')
        ax.set_ylabel('RPM')
        ax.grid(True, alpha=0.3)
        
        if self.controller.control_history:
            rpm_left = np.array([c['rpm_left'] for c in self.controller.control_history])
            rpm_right = np.array([c['rpm_right'] for c in self.controller.control_history])
            cycles = np.arange(len(rpm_left))
            
            ax.plot(cycles, rpm_left, 'b-', label='Left Wheel', linewidth=1.5)
            ax.plot(cycles, rpm_right, 'r-', label='Right Wheel', linewidth=1.5)
            
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            ax.legend(fontsize=9)
    
    def _plot_error_distribution(self, ax):
        """Plot error distribution histogram"""
        ax.set_title('Position Error Distribution', fontweight='bold')
        ax.set_xlabel('Position Error (m)')
        ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)
        
        if self.controller.tracking_errors:
            pos_errors = [e['position_error'] for e in self.controller.tracking_errors]
            
            ax.hist(pos_errors, bins=20, alpha=0.7, color='blue', edgecolor='black')
            
            mean_err = np.mean(pos_errors)
            std_err = np.std(pos_errors)
            ax.axvline(x=mean_err, color='red', linestyle='--', linewidth=2,
                      label='Mean: {:.3f}m'.format(mean_err))
            
            ax.text(0.98, 0.98, 'Std: {:.3f}m\nMax: {:.3f}m\nMin: {:.3f}m'.format(
                   std_err, np.max(pos_errors), np.min(pos_errors)),
                   transform=ax.transAxes, verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)
            
            ax.legend(fontsize=9)
    
    def _plot_control_effort(self, ax):
        """Plot control effort analysis"""
        ax.set_title('Control Effort Analysis', fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Control Effort')
        ax.grid(True, alpha=0.3)
        
        if self.controller.control_history:
            controls = np.array([c['control'] for c in self.controller.control_history])
            
            effort = np.sum(controls**2, axis=1)
            cumulative_effort = np.cumsum(effort)
            
            ax.plot(np.arange(len(effort)), effort, 'b-', label='Instantaneous Effort', alpha=0.5)
            
            ax2 = ax.twinx()
            ax2.plot(np.arange(len(cumulative_effort)), cumulative_effort, 
                    'r-', label='Cumulative Effort', linewidth=2)
            ax2.set_ylabel('Cumulative Effort', color='red')
            
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    
    def _plot_path_curvature(self, ax):
        """Plot path curvature analysis"""
        ax.set_title('Path Curvature Analysis', fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Curvature (1/m)')
        ax.grid(True, alpha=0.3)
        
        if len(self.controller.state_history) > 2:
            states = np.array(self.controller.state_history)
            
            dx = np.diff(states[:, 0])
            dy = np.diff(states[:, 1])
            ddx = np.diff(dx)
            ddy = np.diff(dy)
            
            denominator = (dx[1:]**2 + dy[1:]**2)**(3/2)
            denominator[denominator < 1e-6] = 1e-6
            
            curvature = np.abs(dx[1:]*ddy - dy[1:]*ddx) / denominator
            
            ax.plot(np.arange(len(curvature)), curvature, 'b-', linewidth=1.5)
            ax.fill_between(np.arange(len(curvature)), 0, curvature, alpha=0.2)
            
            ax.text(0.02, 0.98, 'Mean curvature: {:.3f} 1/m\nMax curvature: {:.3f} 1/m'.format(
                   np.mean(curvature), np.max(curvature)),
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)
    
    def _plot_performance_summary(self, ax):
        """Plot performance summary metrics"""
        ax.set_title('Performance Summary', fontweight='bold')
        ax.axis('off')
        
        metrics = self.controller.get_performance_metrics()
        
        if metrics:
            summary_text = "=== MPC Performance Summary ===\n\n"
            summary_text += "Tracking Accuracy:\n"
            summary_text += "  Mean Position Error: {:.4f} m\n".format(metrics['mean_position_error'])
            summary_text += "  Max Position Error:  {:.4f} m\n".format(metrics['max_position_error'])
            summary_text += "  RMS Position Error:  {:.4f} m\n\n".format(metrics['rms_position_error'])
            
            summary_text += "Control Statistics:\n"
            summary_text += "  Total Control Cycles: {}\n".format(metrics['total_cycles'])
            
            if self.controller.control_history:
                controls = np.array([c['control'] for c in self.controller.control_history])
                summary_text += "  Avg Linear Velocity:  {:.3f} m/s\n".format(np.mean(np.abs(controls[:, 0])))
                summary_text += "  Avg Angular Velocity: {:.3f} rad/s\n\n".format(np.mean(np.abs(controls[:, 1])))
            
            summary_text += "Path Following:\n"
            if self.controller.waypoints:
                summary_text += "  Waypoints: {}\n".format(len(self.controller.waypoints))
            
            if 'total_distance' in metrics:
                summary_text += "  Total Distance: {:.3f} m\n".format(metrics['total_distance'])
            
            # Performance grade
            if metrics['rms_position_error'] < 0.05:
                grade = "Excellent"
            elif metrics['rms_position_error'] < 0.1:
                grade = "Good"
            elif metrics['rms_position_error'] < 0.2:
                grade = "Fair"
            else:
                grade = "Needs Improvement"
            
            summary_text += "\nPerformance Grade: {}".format(grade)
            
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3),
                   fontsize=9)