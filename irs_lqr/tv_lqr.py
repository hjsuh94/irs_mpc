import numpy as np
from pydrake.all import (
    MathematicalProgram,
    OsqpSolver,
    SnoptSolver,
    ClpSolver,
    GurobiSolver,
    eq)


def get_solver(solver_name: str):
    if solver_name == "osqp":
        return OsqpSolver()

    if solver_name == "snopt":
        return SnoptSolver()

    if solver_name == "clp":
        return ClpSolver()

    if solver_name == "scs":
        return ScsSolver()

    if solver_name == "gurobi":
        return GurobiSolver()

    raise ValueError("Do not recognize solver.")


def solve_tvlqr(At, Bt, ct, Q, Qd, R, x0, xdt, xbound, ubound, solver,
                xinit=None, uinit=None):
    """
    Solve time-varying LQR problem as an instance of a quadratic program (QP).
    Uses Drake's OSQP solver by default. Can use other solvers that Drake
    supports, but OSQP will often result in fastest time (while being slightly inexact)
    args:
     - At   (np.array, dim: T x n x n) : time-varying dynamics matrix
     - Bt   (np.array, dim: T x n x m) : time-varying actuation matrix.
     - ct   (np.array, dim: T x n x 1) : bias term for affine dynamics.
     - Q    (np.array, dim: n x n): Quadratic cost on state error x(t) - xd(t)
     - Qd    (np.array, dim: n x n): Quadratic cost on final state error x(T) - xd(T)     
     - R    (np.array, dim: m x m): Quadratic cost on actuation.
     - x0   (np.array, dim: n): Initial state of the problem.
     - xdt  (np.array, dim: (T + 1) x n): Desired trajectory of the system.
     - xbound (np.array, dim: 2 x n): (lb, ub) Bound on state variables.
     - ubound (np.array, dim: 2 x u): (lb, ub) Bound on input variables.
     - solver (Drake's solver class): solver. Initialized outside the loop for 
             better performance.
     - xinit (np.array, dim: (T + 1) x n): initial guess for state.
     - uinit (np.array, dim: T x m): initial guess for input.
    NOTE(terry-suh): This implementation needs to be "blazing fast.". It is 
    performed O(iterations * timesteps^2).
    """

    prog = MathematicalProgram()

    timesteps = At.shape[0]
    state_dim = Q.shape[0]
    input_dim = R.shape[0]

    # 1. Declare new variables corresponding to optimal state and input.
    xt = prog.NewContinuousVariables(timesteps + 1, state_dim, "state")
    ut = prog.NewContinuousVariables(timesteps, input_dim, "input")

    if xinit is not None:
        prog.SetInitialGuess(xt, xinit)
    if uinit is not None:
        prog.SetInitialGuess(ut, uinit)
    # 2. Initial constraint.
    prog.AddConstraint(eq(xt[0, :], x0))

    # 3. Loop over to add dynamics constraints and costs.
    for t in range(timesteps):
        # Add affine dynamics constraint.
        prog.AddLinearEqualityConstraint(
            np.hstack((At[t], Bt[t], -np.eye(state_dim))), -ct[t],
            np.hstack((xt[t, :], ut[t, :], xt[t + 1, :]))
        )

        prog.AddBoundingBoxConstraint(xbound[0], xbound[1], xt[t, :])
        prog.AddBoundingBoxConstraint(ubound[0], ubound[1], ut[t, :])

        # Add cost.
        prog.AddQuadraticErrorCost(Q, xdt[t, :], xt[t, :])
        prog.AddQuadraticCost(R, np.zeros(input_dim), ut[t, :])

    # Add final constraint.
    prog.AddQuadraticErrorCost(Qd, xdt[timesteps, :], xt[timesteps, :])
    # prog.AddBoundingBoxConstraint(xbound[0], xbound[1], xt[timesteps,:])

    # 4. Solve the program.

    result = solver.Solve(prog)

    if not result.is_success():
        raise ValueError("TV_LQR failed. Optimization problem is not solved.")

    xt_star = result.GetSolution(xt)
    ut_star = result.GetSolution(ut)

    return xt_star, ut_star


def solve_tvlqr_quasistatic(
        At: np.ndarray, Bt: np.ndarray, ct: np.ndarray, Q: np.ndarray,
        Qd: np.ndarray, R: np.ndarray, x0: np.ndarray, x_trj_d: np.ndarray,
        x_bound: np.ndarray, u_bound: np.ndarray, indices_u_into_x: np.ndarray,
        solver,
        x_init=None, u_init=None, **kwargs):
    """
    Solve time-varying LQR problem as an instance of a quadratic program (QP).
    n == dim_x, m == dim_u.
    args:
     - At   (T x n x n) : time-varying dynamics matrix
     - Bt   (T x n x m) : time-varying actuation matrix.
     - ct   (T x n x 1) : bias term for affine dynamics.
     - Q    (n x n): Quadratic cost on state error x(t) - xd(t)
     - Qd   (n x n): Quadratic cost on final state error x(t) - xd(t)
     - R    (m x m): Quadratic cost on actuation.
     - x0   (n,): Initial state of the problem.
     - x_trj_d  (np.array, dim: (T + 1) x n): Desired trajectory of the system.
     - x_bound (2, T + 1, n): Bounds on state variables.
        x_bound[0, t]: lower bounds on xt[t]. x_bound[1, t]: upper bounds.
     - u_bound (2, T, m): Bounds on input variables.
        u_bound[0]: lower bounds on xt[t]. u_bound[1]: upper bounds.
     - indices_u_into_x: in a quasistatic system, x is the configuration of
     the whole system, whereas u is the commanded configuration of the
     actuated DOFs. x[indices_u_into_x] = u.
     - x_init ((T + 1) x n): initial guess for state.
     - u_init (T x m): initial guess for input.
    """
    prog = MathematicalProgram()

    T = At.shape[0]
    n_x = Q.shape[0]
    n_u = R.shape[0]

    # 1. Declare new variables corresponding to optimal state and input.
    xt = prog.NewContinuousVariables(T + 1, n_x, "state")
    ut = prog.NewContinuousVariables(T, n_u, "input")

    if x_init is not None:
        prog.SetInitialGuess(xt, x_init)
    if u_init is not None:
        prog.SetInitialGuess(ut, u_init)
    # 2. Initial constraint.
    prog.AddConstraint(eq(xt[0, :], x0))

    # 3. Loop over to add dynamics constraints and costs.
    for t in range(T):
        # Add affine dynamics constraint.
        prog.AddLinearEqualityConstraint(
            np.hstack((At[t], Bt[t], -np.eye(n_x))), -ct[t],
            np.hstack((xt[t, :], ut[t, :], xt[t + 1, :])))

        # Note that bounds are not added to xt[0], as it is already
        # equality-constrained.
        prog.AddBoundingBoxConstraint(x_bound[0, t], x_bound[1, t], xt[t + 1])
        prog.AddBoundingBoxConstraint(u_bound[0, t], u_bound[1, t], ut[t])

        # Add cost.
        prog.AddQuadraticErrorCost(Q, x_trj_d[t], xt[t])
        if t == 0:
            du = ut[t] - xt[t, indices_u_into_x]
        else:
            du = ut[t] - ut[t - 1]
        prog.AddQuadraticCost(du.dot(R).dot(du))

    # Add final constraint.
    prog.AddQuadraticErrorCost(Qd, x_trj_d[T, :], xt[T, :])

    # 4. Solve the program.
    result = solver.Solve(prog)

    if not result.is_success():
        raise ValueError("TV_LQR failed. Optimization problem is not solved.")

    xt_star = result.GetSolution(xt)
    ut_star = result.GetSolution(ut)

    return xt_star, ut_star
