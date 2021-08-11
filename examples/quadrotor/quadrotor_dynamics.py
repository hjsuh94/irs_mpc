import numpy as np
import pydrake.symbolic as ps
import torch
import time

#from quadrotor_plant import MakeQuadrotorPlant
from pydrake.examples.quadrotor import QuadrotorPlant
from pydrake.all import (
    Simulator, ResetIntegratorFromFlags,
)
from pydrake.forwarddiff import jacobian

from algorithm.dynamical_system import DynamicalSystem

class QuadrotorDynamics(DynamicalSystem):
    def __init__(self, h):
        super().__init__()
        """
        x = [x pos, y pos, heading, speed, steering_angle]
        u = [acceleration, steering_velocity]
        """
        self.h = h
        self.dim_x = 12
        self.dim_u = 4

        self.m = 0.775
        self.L = 0.15 
        self.g = 9.81

        self.I = np.array([
            [0.0015, 0, 0],
            [0, 0.0025, 0], 
            [0, 0, 0.0035]
        ])
        self.I_inv = np.linalg.inv(self.I)

        self.kF = 1.0
        self.kM = 0.0245

    def dynamics(self, x, u):
        xdot = np.empty(x.shape, dtype=np.float)

        uF = self.kF * u
        uM = self.kM * u
        Fg = np.array([0., 0., -self.m * self.g])
        F = np.array([0., 0., uF.sum()])
        M = np.array([self.L*(-uF[0] - uF[1] + uF[2] + uF[3]),
                    self.L*(-uF[0] - uF[3] + uF[1] + uF[2]),
                    - uM[0] + uM[1] - uM[2] + uM[3]])

        rpy = x[3:6]
        rpy_d = x[9:12]
        R_WB = self.CalcR_WB(rpy)

        # translational acceleration in world frame
        xyz_dd = 1./self.m*(R_WB.dot(F) + Fg)

        # pqr: angular velocity in body frame
        Phi_inv = self.CalcPhiInv(rpy)
        pqr = Phi_inv.dot(rpy_d)
        pqr_d = self.I_inv.dot(M - np.cross(pqr, self.I.dot(pqr)))

        '''
        rpy_d = Phi * pqr ==>
        rpy_dd = Phi_d * pqr + Phi * pqr_d
        Phi_d.size = (3,3,3): Phi_d[i,j] is the partial of Phi[i,j]
            w.r.t rpy.
        '''
        Phi_d = self.CalcPhiD(rpy)
        Phi = self.CalcPhi(rpy)
        rpy_dd = Phi.dot(pqr_d) + (Phi_d.dot(rpy_d)).dot(pqr)
    
        xdot[0:6] = x[6:12]
        xdot[6:9] = xyz_dd
        xdot[9:12] = rpy_dd

        return x + self.h * xdot

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

    def dynamics_autodiff(self, x, u):
        xdot = np.empty(x.shape, dtype=np.object)

        uF = self.kF * u
        uM = self.kM * u
        Fg = np.array([0., 0., -self.m * self.g])
        F = np.array([0., 0., uF.sum()])
        M = np.array([self.L*(-uF[0] - uF[1] + uF[2] + uF[3]),
                    self.L*(-uF[0] - uF[3] + uF[1] + uF[2]),
                    - uM[0] + uM[1] - uM[2] + uM[3]])

        rpy = x[3:6]
        rpy_d = x[9:12]
        R_WB = self.CalcR_WB(rpy)

        # translational acceleration in world frame
        xyz_dd = 1./self.m*(R_WB.dot(F) + Fg)

        # pqr: angular velocity in body frame
        Phi_inv = self.CalcPhiInv(rpy)
        pqr = Phi_inv.dot(rpy_d)
        pqr_d = self.I_inv.dot(M - np.cross(pqr, self.I.dot(pqr)))

        '''
        rpy_d = Phi * pqr ==>
        rpy_dd = Phi_d * pqr + Phi * pqr_d
        Phi_d.size = (3,3,3): Phi_d[i,j] is the partial of Phi[i,j]
            w.r.t rpy.
        '''
        Phi_d = self.CalcPhiD(rpy)
        Phi = self.CalcPhi(rpy)
        rpy_dd = Phi.dot(pqr_d) + (Phi_d.dot(rpy_d)).dot(pqr)
    
        xdot[0:6] = x[6:12]
        xdot[6:9] = xyz_dd
        xdot[9:12] = rpy_dd

        return x + self.h * xdot

    def dynamics_xu(self, xu):
        return self.dynamics_autodiff(
            xu[0:self.dim_x], xu[self.dim_x:self.dim_x + self.dim_u])

    def jacobian_xu(self, x, u):
        xu = np.hstack((x,u))
        return jacobian(self.dynamics_xu, xu)

    def jacobian_xu_batch(self, x, u):
        """
        Recoever linearized dynamics dfd(xu) as a function of x, u
        """ 
        dxdu_batch = np.zeros((
            x.shape[0], x.shape[1], x.shape[1] + u.shape[1]))
        for i in range(x.shape[0]):
            dxdu_batch[i] = self.jacobian_xu(x[i], u[i])
        return dxdu_batch                
        
    def CalcRx(self, phi):
        c = np.cos(phi)
        s = np.sin(phi)
        Rx = np.array([[1., 0., 0.],
                    [0, c, -s],
                    [0, s, c]])
        return Rx


    def CalcRy(self, theta):
        c = np.cos(theta)
        s = np.sin(theta)
        Ry = np.array([[c, 0., s],
                    [0, 1., 0],
                    [-s, 0., c]])
        return Ry


    def CalcRz(self, psi):
        c = np.cos(psi)
        s = np.sin(psi)
        Rz = np.array([[c, -s, 0],
                    [s, c, 0],
                    [0., 0., 1]])
        return Rz


    # Transformation matrix from Body frame to World frame.
    def CalcR_WB(self, rpy):
        phi = rpy[0] # roll angle
        theta = rpy[1] # pitch angle
        psi = rpy[2] # yaw angle

        return self.CalcRz(psi).dot(self.CalcRy(theta).dot(self.CalcRx(phi)))

    def CalcPhiInv(self, rpy):
        roll = rpy[0]
        pitch = rpy[1]
        sr = np.sin(roll)
        cr = np.cos(roll)
        sp = np.sin(pitch)
        cp = np.cos(pitch)

        Phi = np.array([[1, 0, -sp],
                        [0, cr, sr*cp],
                        [0, -sr, cr*cp]])
        return Phi

    def CalcPhi(self, rpy):
        roll = rpy[0]
        pitch = rpy[1]
        sr = np.sin(roll)
        cr = np.cos(roll)
        sp = np.sin(pitch)
        cp = np.cos(pitch)

        Phi = np.array([[1, sr*sp/cp, cr*sp/cp],
                        [0, cr, -sr],
                        [0, sr/cp, cr/cp]])
        return Phi


    def CalcPhiD(self, rpy):
        roll = rpy[0]
        pitch = rpy[1]
        sr = np.sin(roll)
        cr = np.cos(roll)
        sp = np.sin(pitch)
        cp = np.cos(pitch)
        cp2 = cp**2
        tp = sp/cp
        
        Phi_D = np.empty((3,3,3), dtype=object)
        Phi_D[:,0,:] = 0.0
        Phi_D[0, 1] = [cr * tp, sr / cp2, 0]
        Phi_D[0, 2] = [-sr * tp, cr / cp2, 0]
        Phi_D[1, 1] = [-sr, 0, 0]
        Phi_D[1, 2] = [-cr, 0, 0]
        Phi_D[2, 1] = [cr/cp, sr*sp/cp2, 0]
        Phi_D[2, 2] = [-sr/cp, cr*sp/cp2, 0]

        return Phi_D
