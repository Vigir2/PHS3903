import numpy as np
import init_functions as init_func
import parameters.h2O_model as h2O
import parameters.simulation_parameters as simP

class H2O:
    def __init__(self, dim, pos = None, vel = None, T = 100):
        self.dim = dim
        self.T = T
        if pos:
            self.pos = pos
            pass
        else:
            pos = init_func.init_water(dim=self.dim)
            self.O_pos, self.H1_pos, self.H2_pos, self.M_pos = (pos[i] for i in range(len(pos)))
            self.rand_orientation()
        if vel:
            self.vel = vel
            pass
        else:
            self.cm_vel = init_func.random_velocity(h2O.M, T, self.dim)
            self.rot_vel = init_func.random_rot_velocity(self.inertia_tensor(), T, self.dim)
    
    def cm_pos(self):
        r = (h2O.mH) * (self.H1_pos + self.H2_pos) + h2O.mO * self.O_pos
        return r/h2O.M
    
    def rpos(self, M=False):
        cm = self.cm_pos()
        if M:
            return self.O_pos - cm, self.H1_pos - cm, self.H2_pos - cm, self.M_pos - cm
        else:
             return self.O_pos - cm, self.H1_pos - cm, self.H2_pos - cm

    def rand_orientation(self):
        cm = self.cm_pos()
        rotation_matrix = init_func.random_euler_angles(self.dim, True)      
        r_pos = self.rpos(M = True)
        r_posr = np.array([rotation_matrix@r_pos[i] for i in range(len(r_pos))])
        self.O_pos, self.H1_pos, self.H2_pos, self.M_pos = cm + r_posr[0], cm + r_posr[1], cm + r_posr[2], cm + r_posr[3]

    def rand_position(self):
        r = init_func.random_pos(0, simP.a, self.dim)
        self.O_pos, self.H1_pos, self.H2_pos, self.M_pos = self.O_pos + r, self.H1_pos + r, self.H2_pos + r, self.M_pos + r

    def inertia_tensor(self):
        O_rpos, H1_rpos, H2_rpos = self.rpos(M=False)
        Ixx = h2O.mO * (O_rpos[1]**2 + O_rpos[2]**2) + h2O.mH * (H1_rpos[1]**2 + H1_rpos[2]**2) + h2O.mH * (H2_rpos[1]**2 + H2_rpos[2]**2)
        Iyy = h2O.mO * (O_rpos[0]**2 + O_rpos[2]**2) + h2O.mH * (H1_rpos[0]**2 + H1_rpos[2]**2) + h2O.mH * (H2_rpos[0]**2 + H2_rpos[2]**2)
        Izz = h2O.mO * (O_rpos[0]**2 + O_rpos[1]**2) + h2O.mH * (H1_rpos[0]**2 + H1_rpos[1]**2) + h2O.mH * (H2_rpos[0]**2 + H2_rpos[1]**2)
        Ixy = -(h2O.mO * O_rpos[0]*O_rpos[1] + h2O.mH * H1_rpos[0]*H1_rpos[1] + h2O.mH * H2_rpos[0]*H2_rpos[1])
        Ixz = -(h2O.mO * O_rpos[0]*O_rpos[2] + h2O.mH * H1_rpos[0]*H1_rpos[2] + h2O.mH * H2_rpos[0]*H2_rpos[2])
        Iyz = -(h2O.mO * O_rpos[1]*O_rpos[2] + h2O.mH * H1_rpos[1]*H1_rpos[2] + h2O.mH * H2_rpos[1]*H2_rpos[2])
        return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    
    def d_inertia_tensor(self):
        omega = self.rot_vel
        I = self.inertia_tensor()
        I_xx = 2*(omega[1]*I[0,2] - omega[2]*I[0,1])
        I_yy = 2*(omega[2]*I[0,1] - omega[0]*I[1,2])
        I_zz = 2*(omega[0]*I[1,2] - omega[1]*I[0,2])
        I_xy = omega[2] * (I[0,0] - I[1,1]) - omega[0] * I[0, 2] + omega[1] * I[1,2]
        I_xz = omega[1] * (I[2,2] - I[0,0]) - omega[2] * I[1, 2] + omega[0] * I[0,1]
        I_yz = omega[0] * (I[1,1] - I[2,2]) - omega[1] * I[0, 1] + omega[2] * I[0,2]
        return np.array([[I_xx, I_xy, I_xz], [I_xy, I_yy, I_yz], [I_xz, I_yz, I_zz]])
    
    def reset_molecule(self):
        pos = init_func.init_water(dim=self.dim)
        self.O_pos, self.H1_pos, self.H2_pos, self.M_pos = (pos[i] for i in range(len(pos)))
        self.rand_orientation()
        self.cm_vel = init_func.random_velocity(h2O.M, self.T, self.dim)
        self.rot_vel = init_func.random_rot_velocity(self.inertia_tensor(), self.T, self.dim)

"""
    def correct_pos(self):
        for i in np.where(self.O_pos > simP.a):
            self.O_pos[i] -= simP.a
        for i in np.where(self.H1_pos > simP.a):
            self.H1_pos[i] -= simP.a
        for i in np.where(self.H2_pos > simP.a):
            self.H2_pos[i] -= simP.a
        for i in np.where(self.M_pos > simP.a):
            self.M_pos[i] -= simP.a
"""




if __name__ == "__main__":
    m = H2O(3)
    print(m.cm_vel)
    print(m.rot_vel)
    m.d_inertia_tensor()