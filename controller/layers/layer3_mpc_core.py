"""
Layer 3: MPC Core - Model Predictive Control Implementation
"""
import numpy as np
from config.robot_params import RobotParams
from layers.layer1_plant_model import DifferentialDriveModel
from layers.layer2_linearization import Linearizer
from utils.qp_solver import QPSolver

class MPCController:
    """
    Model Predictive Controller for differential drive robot
    """
    
    def __init__(self, horizon=RobotParams.MPC_HORIZON, dt=RobotParams.MPC_DT):
        self.horizon = horizon
        self.dt = dt
        self.n_states = 3  # x, y, theta
        self.n_controls = 2  # v, omega
        
        # Initialize components
        self.model = DifferentialDriveModel(dt)
        self.linearizer = Linearizer(dt)
        self.qp_solver = QPSolver()
        
        # Cost weights
        self.Q = np.diag([RobotParams.Q_X, RobotParams.Q_Y, RobotParams.Q_THETA])
        self.R = np.diag([RobotParams.R_V, RobotParams.R_OMEGA])
        
        # Control limits
        self.u_min = np.array([-RobotParams.MPC_MAX_VELOCITY, -RobotParams.MPC_MAX_ANGULAR_VEL])
        self.u_max = np.array([RobotParams.MPC_MAX_VELOCITY, RobotParams.MPC_MAX_ANGULAR_VEL])
        
        # Store previous solution for warm start
        self.u_prev = np.zeros((self.horizon, self.n_controls))
        
    def compute_control(self, current_state, reference_trajectory):
        """
        Compute optimal control sequence using MPC
        
        Args:
            current_state: [x, y, theta]
            reference_trajectory: N x 3 array of reference states
            
        Returns:
            optimal_control: [v, omega] for current timestep
        """
        u_sequence = self.compute_control_sequence(current_state, reference_trajectory)
        return u_sequence[0]

    def compute_control_sequence(self, current_state, reference_trajectory):
        """
        Compute the full optimal control sequence.

        Args:
            current_state: [x, y, theta]
            reference_trajectory: N x 3 array of reference states

        Returns:
            u_sequence: horizon x 2 array of [v, omega] controls
        """
        current_state = np.asarray(current_state, dtype=float)
        reference_trajectory = self._prepare_reference(reference_trajectory, current_state)

        nominal_control = self._estimate_nominal_control(current_state, reference_trajectory[0])
        A, B = self.linearizer.linearize(current_state, nominal_control)
        
        # Build prediction matrices
        F, G = self._build_prediction_matrices(A, B)
        
        # Build cost function matrices
        P, q = self._build_cost_function(current_state, reference_trajectory, F, G)
        
        # Build constraint matrices
        lb, ub = self._build_constraints()
        
        # Solve QP problem
        u_opt = self.qp_solver.solve(P, q, lb, ub)
        u_opt = np.clip(u_opt, lb, ub)
        
        # Update previous solution for warm start
        self.u_prev = u_opt.reshape(self.horizon, self.n_controls)

        return self.u_prev.copy()

    def predict_trajectory(self, current_state, control_sequence):
        """
        Roll out the nonlinear model with a sequence of controls.
        """
        state = np.asarray(current_state, dtype=float).copy()
        trajectory = [state.copy()]

        for control in control_sequence:
            state = self.model.discrete_dynamics(state, np.asarray(control, dtype=float))
            state[2] = self._normalize_angle(state[2])
            trajectory.append(state.copy())

        return np.array(trajectory)
    
    def _build_prediction_matrices(self, A, B):
        """
        Build prediction matrices for state evolution
        X = F * x0 + G * U
        
        Where:
        X = [x1, x2, ..., xN]  (stacked states)
        U = [u0, u1, ..., u_{N-1}]  (stacked controls)
        """
        N = self.horizon
        
        # Initialize matrices
        F = np.zeros((N * self.n_states, self.n_states))
        G = np.zeros((N * self.n_states, N * self.n_controls))
        
        # Build F matrix
        for i in range(N):
            F[i*self.n_states:(i+1)*self.n_states, :] = np.linalg.matrix_power(A, i+1)
        
        # Build G matrix
        for i in range(N):
            for j in range(i+1):
                G[i*self.n_states:(i+1)*self.n_states, 
                  j*self.n_controls:(j+1)*self.n_controls] = np.linalg.matrix_power(A, i-j) @ B
        
        return F, G
    
    def _build_cost_function(self, current_state, reference_trajectory, F, G):
        """
        Build quadratic cost function
        J = (X - X_ref)^T Q (X - X_ref) + U^T R U
        = 1/2 * U^T P U + q^T U + constant
        
        Args:
            current_state: current robot state
            reference_trajectory: N x 3 reference states
            F, G: prediction matrices
            
        Returns:
            P: Hessian matrix
            q: gradient vector
        """
        N = self.horizon
        
        # Build block diagonal Q and R matrices
        Q_bar = np.kron(np.eye(N), self.Q)
        R_bar = np.kron(np.eye(N), self.R)
        
        reference_trajectory = self._prepare_reference(reference_trajectory, current_state)
        x_ref = reference_trajectory[:N].flatten()
        
        # Compute cost matrices
        # P = G^T Q_bar G + R_bar
        P = G.T @ Q_bar @ G + R_bar
        
        # q = G^T Q_bar (F x0 - x_ref)
        q = G.T @ Q_bar @ (F @ current_state - x_ref)
        
        # Ensure P is positive definite
        P = (P + P.T) / 2  # Symmetrize
        min_eig = np.min(np.linalg.eigvalsh(P))
        if min_eig < 1e-6:
            P += (1e-6 - min_eig) * np.eye(P.shape[0])
        
        return P, q
    
    def _build_constraints(self):
        """
        Build control constraints
        u_min <= u_k <= u_max for k = 0, ..., N-1
        """
        N = self.horizon
        
        # Stack constraints for all timesteps
        lb = np.tile(self.u_min, N)
        ub = np.tile(self.u_max, N)
        
        return lb, ub
    
    def generate_reference_trajectory(self, waypoints, current_state):
        """
        Generate reference trajectory from waypoints
        
        Args:
            waypoints: list of (x, y) waypoints
            current_state: current robot state
            
        Returns:
            reference_trajectory: horizon x 3 array of reference states
        """
        waypoints = [np.asarray(wp, dtype=float) for wp in waypoints]
        trajectory = []
        
        if len(waypoints) == 0:
            # Hold current position
            return np.tile(current_state, (self.horizon, 1))
        
        # Simple interpolation between waypoints
        current_waypoint_idx = 0
        current_state = np.asarray(current_state, dtype=float)
        current_pos = current_state[:2].copy()
        heading = current_state[2]
        
        for i in range(self.horizon):
            if current_waypoint_idx < len(waypoints):
                target = waypoints[current_waypoint_idx]
                direction = target - current_pos
                distance = np.linalg.norm(direction)
                
                while distance < 1e-6 and current_waypoint_idx < len(waypoints) - 1:
                    current_waypoint_idx += 1
                    target = waypoints[current_waypoint_idx]
                    direction = target - current_pos
                    distance = np.linalg.norm(direction)
                
                if distance > 1e-6:
                    step_size = min(RobotParams.MPC_MAX_VELOCITY * self.dt, distance)
                    interp_pos = current_pos + (direction / distance) * step_size
                    heading = np.arctan2(direction[1], direction[0])
                else:
                    interp_pos = current_pos.copy()
                
                current_pos = interp_pos
                
                trajectory.append([interp_pos[0], interp_pos[1], heading])
                
                # Move to next waypoint
                if np.linalg.norm(target - current_pos) < 1e-6:
                    current_pos = target
                    current_waypoint_idx += 1
            else:
                # Hold at last waypoint
                trajectory.append([current_pos[0], current_pos[1], heading])
        
        return np.array(trajectory)

    def _prepare_reference(self, reference_trajectory, current_state):
        reference = np.asarray(reference_trajectory, dtype=float)

        if reference.size == 0:
            return np.tile(current_state, (self.horizon, 1))

        reference = np.atleast_2d(reference)
        if reference.shape[1] != self.n_states:
            raise ValueError("reference_trajectory must have shape N x 3")

        if len(reference) < self.horizon:
            pad = np.tile(reference[-1], (self.horizon - len(reference), 1))
            reference = np.vstack([reference, pad])

        reference = reference[:self.horizon].copy()
        reference[:, 2] = np.vectorize(self._normalize_angle)(reference[:, 2])
        return reference

    def _estimate_nominal_control(self, current_state, reference_state):
        dx = reference_state[0] - current_state[0]
        dy = reference_state[1] - current_state[1]
        distance = np.hypot(dx, dy)

        desired_heading = np.arctan2(dy, dx) if distance > 1e-6 else reference_state[2]
        heading_error = self._normalize_angle(desired_heading - current_state[2])
        reference_heading_error = self._normalize_angle(reference_state[2] - current_state[2])

        v = min(RobotParams.MPC_MAX_VELOCITY, distance / max(self.dt, 1e-6))
        if abs(heading_error) > np.pi / 2:
            v = 0.0

        omega = (heading_error + 0.5 * reference_heading_error) / max(self.dt, 1e-6)
        return np.clip(np.array([v, omega]), self.u_min, self.u_max)

    def _normalize_angle(self, angle):
        return np.arctan2(np.sin(angle), np.cos(angle))
