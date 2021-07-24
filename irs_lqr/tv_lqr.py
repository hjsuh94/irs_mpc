import numpy as np
from pydrake.all import (
    MathematicalProgram, 
    OsqpSolver,
    SnoptSolver,
    ClpSolver,
    GurobiSolver,
    eq)

def get_solver(solver_name : str):
    if solver_name == "osqp":
        return OsqpSolver()

    if solver_name == "snopt":
        return SnoptSolver()

    if solver_name == "clp":
        return  ClpSolver()

    if solver_name == "scs":
        return ScsSolver()

    if solver_name == "gurobi":
        return GurobiSolver()

    raise ValueError("Do not recognize solver.")    

def TvLqr(At, Bt, ct, Q, Qd, R, x0, xdt, xbound, ubound, solver,
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
    prog.AddConstraint(eq(xt[0,:], x0))

    # 3. Loop over to add dynamics constraints and costs.
    for t in range(timesteps):
        # Add affine dynamics constraint.
        prog.AddLinearEqualityConstraint(
            np.hstack((At[t], Bt[t], -np.eye(state_dim))), -ct[t],
            np.hstack((xt[t,:], ut[t,:], xt[t+1,:]))
        )

        prog.AddBoundingBoxConstraint(xbound[0], xbound[1], xt[t,:])
        prog.AddBoundingBoxConstraint(ubound[0], ubound[1], ut[t,:])
        
        # Add cost.
        prog.AddQuadraticErrorCost(Q, xdt[t,:], xt[t,:])
        prog.AddQuadraticCost(R, np.zeros(input_dim), ut[t,:])

    # Add final constraint.
    prog.AddQuadraticErrorCost(Qd, xdt[timesteps,:], xt[timesteps,:])
    #prog.AddBoundingBoxConstraint(xbound[0], xbound[1], xt[timesteps,:])

    # 4. Solve the program.

    result = solver.Solve(prog)

    if not result.is_success():
        raise ValueError("TV_LQR failed. Optimization problem is not solved.")

    xt_star = result.GetSolution(xt)
    ut_star = result.GetSolution(ut)

    return xt_star, ut_star
