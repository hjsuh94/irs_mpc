import numpy as np
import pydrake.symbolic as ps
import torch
import time

#from quadrotor_plant import MakeQuadrotorPlant
from pydrake.examples.quadrotor import QuadrotorPlant
from pydrake.all import (
    Simulator, ResetIntegratorFromFlags,
)

from irs_lqr.dynamical_system import DynamicalSystem

class QuadrotorDynamics(DynamicalSystem):
    def __init__(self, h):
        super(QuadrotorDynamics, self).__init__()
        """
        x = [x pos, y pos, heading, speed, steering_angle]
        u = [acceleration, steering_velocity]
        """
        self.h = h
        self.dim_x = 12
        self.dim_u = 4

        self.quadrotor = QuadrotorPlant()
        #self.quadrotor_autodiff = MakeQuadrotorPlant(AutoDiffXd)

        self.time = 0.0

        self.simulator = Simulator(self.quadrotor)
        #self.simulator_autodiff = Simulator_[AutoDiffXd](self.quadrotor_autodiff)
        ResetIntegratorFromFlags(self.simulator, "semi_explicit_euler", self.h)

    def dynamics(self, x, u):

        # 1. Set x.
        context = self.simulator.get_mutable_context()
        state = context.get_mutable_continuous_state_vector()
        state.SetFromVector(x)

        # 2. Set u.
        self.quadrotor.get_input_port().FixValue(context, u)
        self.simulator.AdvanceTo(self.time + self.h)
        next_state = self.quadrotor.get_output_port().EvalBasicVector(context)

        self.time = self.time + self.h
        
        return next_state.get_value()

    def dynamics_batch(self, x, u):
        """
        Batch dynamics. Uses pytorch for 
        -args:
            x (np.array, dim: B x n): batched state
            u (np.array, dim: B x m): batched input
        -returns:
            xnext (np.array, dim: B x n): batched next state
        """
        x_new = np.zeros((x.shape[0], x.shape[1]))
        for b in range(x.shape[0]):
            x_new[b] = self.dynamics(x[b], u[b])
        return x_new
