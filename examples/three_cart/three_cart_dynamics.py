import numpy as np
import pydrake.symbolic as ps
import torch
import time

class ThreeCartDynamics():
    def __init__(self, dt):
        """
        x = [q1, q2, q3, v1, v2, v3]
        u = [u1, u3]
        """
        self.dt = dt
        self.dim_x = 6
        self.dim_u = 2
        self.d = 0.2 # Cart width.

        """Non differentiable simulation, does not support Jacobian computations."""

    def dynamics_np(self, x, u):
        """
        Numeric expression for dynamics.
        x (np.array, dim: n): state
        u (np.array, dim: m): action
        """
        # 1. Unpack states.
        q1, q2, q3, v1, v2, v3 = x
        u1, u3 = u

        # 2. Update velocities.
        v1_semi = v1 + self.dt * u1
        v2_semi = v2
        v3_semi = v3 + self.dt * u3

        # 3. Update positions using semi-implicit integration.
        q1_semi = q1 + self.dt * v1_semi
        q2_semi = q2 + self.dt * v2_semi
        q3_semi = q3 + self.dt * v3_semi

        # 4. Resolve collisions.
        # Case 1. All three are in collision
        # Case 2. 1 and 2 are in collision.
        # Case 3. 2 and 3 are in collision.
        # Case 4. No collision.

        if (q2_semi - q1_semi < self.d) and (q3_semi - q2_semi < self.d):
            # If all three are colliding, we take their average and place them
            # in q2. This is consistent with Gauss's principle of least action
            # subject to the non-penetration constraint.
            q2_next = (1./3.) * (q1_semi + q2_semi + q3_semi)
            q1_next = q2_next - self.d
            q3_next = q2_next + self.d

            # Next, the velocities of all three carts are averaged to the same
            # value. This is consistent with momentum conservation under 
            # perfectly inelastic impacts. 
            v_avg = (1./3.) * (v1_semi + v2_semi + v3_semi)
            v1_next = v_avg
            v2_next = v_avg
            v3_next = v_avg

        elif (q2_semi - q1_semi < self.d):
            # If only the first two carts are colliding, we do the same and 
            # average their position values.
            penetration_depth = self.d - (q2_semi - q1_semi)
            q2_next = q2_semi + 0.5 * penetration_depth
            q1_next = q1_semi - 0.5 * penetration_depth

            # Velocities are averaged.
            v_avg = 0.5 * (v1_semi + v2_semi)
            v1_next = v_avg
            v2_next = v_avg

            # Third cart does its own thing.
            q3_next = q3_semi            
            v3_next = v3_semi

        elif (q3_semi - q2_semi < self.d):
            # If only the first two carts are colliding, we do the same and 
            # average their position values.
            penetration_depth = self.d - (q3_semi - q2_semi)
            q3_next = q3_semi + 0.5 * penetration_depth
            q2_next = q2_semi - 0.5 * penetration_depth

            # Velocities are averaged.
            v_avg = 0.5 * (v2_semi + v3_semi)
            v2_next = v_avg
            v3_next = v_avg  

            # First cart does its own thing.
            q1_next = q1_semi            
            v1_next = v1_semi                      

        else:
            # If there are no collisions, let them be.
            q1_next = q1_semi
            q2_next = q2_semi
            q3_next = q3_semi 

            v1_next = v1_semi
            v2_next = v2_semi
            v3_next = v3_semi

        x_new = np.array([q1_next, q2_next, q3_next, v1_next, v2_next, v3_next])
        return x_new

    def dynamics_batch_np(self, x, u):
        """
        Batch dynamics using numpy.
        -args:
            x (np.array, dim: B x n): batched state
            u (np.array, dim: B x m): batched input
        -returns:
            xnext (np.array, dim: B x n): batched next state
        """

        # 1. Unpack states.
        q1 = x[:,0]
        q2 = x[:,1]
        q3 = x[:,2]
        v1 = x[:,3]
        v2 = x[:,4]
        v3 = x[:,5]

        u1 = u[:,0]
        u3 = u[:,1]

        # 2. Update velocities.
        v1_semi = v1 + self.dt * u1
        v2_semi = v2
        v3_semi = v3 + self.dt * u3

        # 3. Update positions using semi-implicit integration.
        q1_semi = q1 + self.dt * v1_semi
        q2_semi = q2 + self.dt * v2_semi
        q3_semi = q3 + self.dt * v3_semi

        x_semi = np.vstack((q1_semi, q2_semi, q3_semi, v1_semi, v2_semi, v3_semi)).transpose()

        # 4. Resolve collisions.

        # Case 1. All three are penetrating.
        indices_1 = np.argwhere(
            (x_semi[:,1] - x_semi[:,0] < self.d) &
            (x_semi[:,2] - x_semi[:,1] < self.d)
        )
        indices_2 = np.argwhere(
            (x_semi[:,1] - x_semi[:,0] < self.d) &
            (x_semi[:,2] - x_semi[:,1] >= self.d)
        )
        indices_3 = np.argwhere(
            (x_semi[:,1] - x_semi[:,0] >= self.d) &
            (x_semi[:,2] - x_semi[:,1] < self.d)
        )                

        penetrating_samples = x_semi[indices_1,:]
        penetrating_samples = np.squeeze(penetrating_samples, axis=1)
        penetrating_samples[:,1] = np.mean(penetrating_samples[:,0:3], axis=1)
        penetrating_samples[:,0] = penetrating_samples[:,1] - self.d
        penetrating_samples[:,2] = penetrating_samples[:,1] + self.d

        v_average = np.mean(penetrating_samples[:,3:6], axis=1)
        penetrating_samples[:,3] = v_average
        penetrating_samples[:,4] = v_average
        penetrating_samples[:,5] = v_average
        x_semi[indices_1] = np.expand_dims(penetrating_samples, axis=1)

        # Case 2. 1 and 2 are penetrating.
        penetrating_samples = x_semi[indices_2,:]
        penetrating_samples = np.squeeze(penetrating_samples, axis=1)            
        penetration_depth = self.d - (penetrating_samples[:,1] - penetrating_samples[:,0])
        penetrating_samples[:,1] += penetration_depth
        penetrating_samples[:,0] -= penetration_depth
        v_average = np.mean(penetrating_samples[:,3:5], axis=1)
        penetrating_samples[:,3] = v_average
        penetrating_samples[:,4] = v_average
        x_semi[indices_2] = np.expand_dims(penetrating_samples, axis=1)

        # Case 3. 2 and 3 are penetrating.

        penetrating_samples = x_semi[indices_3,:]
        penetrating_samples = np.squeeze(penetrating_samples, axis=1)            
        penetration_depth = self.d - (penetrating_samples[:,2] - penetrating_samples[:,1])
        penetrating_samples[:,2] += penetration_depth
        penetrating_samples[:,1] -= penetration_depth
        v_average = np.mean(penetrating_samples[:,4:6], axis=1)
        penetrating_samples[:,4] = v_average
        penetrating_samples[:,5] = v_average
        x_semi[indices_3] = np.expand_dims(penetrating_samples, axis=1)

        return x_semi

    def projection(self, x, dx, u, du):
        """
        Project samples.
        - args:
            x (np.array, dim: n): nominal state for sampling.
            dx (np.array, dim: B x n): samples around the nominal state.
            u (np.array, dim: m): nominal input for sampling.
            du (np.array, dim: B x m): samples around the nominal input.
        - returns:
            x_proj: x + dx projected to the constraint.
            u_proj: u + du projected to the constraint.
        """

        # Search for all samples violating the no-penetration constraint.
        x_proj = x + dx 
        u_proj = u + du

        # Apply custom vectorized penetration handling.
        # TODO(terry-suh): Change this to general ray projection formulation.
        # Case 1. All three are penetrating.
        penetration_indices = np.argwhere(
            (x_proj[:,1] - x_proj[:,0] < self.d) &
            (x_proj[:,2] - x_proj[:,1] < self.d)
        )

        if len(penetration_indices) is not 0:
            penetrating_samples = x_proj[(penetration_indices),:]
            penetrating_samples = np.squeeze(penetrating_samples, axis=1)
            penetrating_samples[:,1] = np.mean(penetrating_samples[:,0:3], axis=1)
            penetrating_samples[:,0] = penetrating_samples[:,1] - self.d
            penetrating_samples[:,2] = penetrating_samples[:,1] + self.d

            x_proj[penetration_indices] = np.expand_dims(penetrating_samples, axis=1)

        # Case 2. 1 and 2 are penetrating.

        penetration_indices = np.argwhere(
            (x_proj[:,1] - x_proj[:,0] < self.d) &
            (x_proj[:,2] - x_proj[:,1] >= self.d)
        )

        if len(penetration_indices) is not 0:

            penetrating_samples = x_proj[penetration_indices,:]
            penetrating_samples = np.squeeze(penetrating_samples, axis=1)            
            penetration_depth = self.d - (penetrating_samples[:,1] - penetrating_samples[:,0])
            penetrating_samples[:,1] += 0.5 * penetration_depth
            penetrating_samples[:,0] -= 0.5 * penetration_depth

            x_proj[penetration_indices] = np.expand_dims(penetrating_samples, axis=1)

        # Case 3. 2 and 3 are penetrating.

        penetration_indices = np.argwhere(
            (x_proj[:,1] - x_proj[:,0] >= self.d) &
            (x_proj[:,2] - x_proj[:,1] < self.d)
        )

        if len(penetration_indices) is not 0:

            penetrating_samples = x_proj[penetration_indices,:]
            penetrating_samples = np.squeeze(penetrating_samples, axis=1)                        
            penetration_depth = self.d - (penetrating_samples[:,2] - penetrating_samples[:,1])
            penetrating_samples[:,2] += 0.5 * penetration_depth
            penetrating_samples[:,1] -= 0.5 * penetration_depth
            
            x_proj[penetration_indices] = np.expand_dims(penetrating_samples, axis=1)

        return x_proj, u_proj

