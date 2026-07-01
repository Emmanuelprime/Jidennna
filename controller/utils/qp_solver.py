"""
Quadratic Programming Solver for MPC
Uses OSQP or simple active-set method
"""
import numpy as np
from scipy.optimize import minimize

try:
    from cvxopt import matrix, solvers
except ImportError:
    matrix = None
    solvers = None

class QPSolver:
    """
    Quadratic Program solver for MPC optimization
    min 0.5 * x^T P x + q^T x
    s.t. lb <= x <= ub
    """
    
    def __init__(self, solver_type='cvxopt'):
        self.solver_type = solver_type
        if self.solver_type == 'cvxopt' and solvers is None:
            self.solver_type = 'scipy'
        
    def solve(self, P, q, lb=None, ub=None):
        """
        Solve QP problem
        
        Args:
            P: Hessian matrix
            q: gradient vector
            lb: lower bounds
            ub: upper bounds
        Returns:
            x_opt: optimal solution
        """
        if self.solver_type == 'cvxopt':
            return self._solve_cvxopt(P, q, lb, ub)
        else:
            return self._solve_scipy(P, q, lb, ub)
    
    def _solve_cvxopt(self, P, q, lb, ub):
        """Solve using CVXOPT"""
        n = len(q)
        
        # Convert to CVXOPT format
        P = matrix(P)
        q = matrix(q)
        
        # Inequality constraints
        G = matrix(np.vstack([np.eye(n), -np.eye(n)]))
        h = matrix(np.hstack([ub, -lb]))
        
        solvers.options['show_progress'] = False
        solvers.options['maxiters'] = 100
        
        try:
            sol = solvers.qp(P, q, G, h)
            if sol.get('status') == 'optimal':
                return np.array(sol['x']).flatten()
        except (ArithmeticError, ValueError):
            pass

        return self._solve_bounded_least_risk(np.array(P), np.array(q).flatten(), lb, ub)
    
    def _solve_scipy(self, P, q, lb, ub):
        """Solve using SciPy optimizer"""
        def objective(x):
            return 0.5 * x.T @ P @ x + q.T @ x
        
        n = len(q)
        bounds = [(lb[i], ub[i]) for i in range(n)]
        x0 = np.zeros(n)
        
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        if result.success:
            return result.x

        return self._solve_bounded_least_risk(P, q, lb, ub)

    def _solve_bounded_least_risk(self, P, q, lb, ub):
        """Return a bounded fallback if the preferred QP solver fails."""
        try:
            x = -np.linalg.solve(P, q)
        except np.linalg.LinAlgError:
            x = np.zeros_like(q)

        if lb is not None and ub is not None:
            x = np.clip(x, lb, ub)

        return x
