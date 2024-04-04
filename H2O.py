import numpy as np
import init_functions as init_func
import parameters.h2O_model as h2O
import parameters.simulation_parameters as simP
from scipy.stats import special_ortho_group

class H2O:
    """
    Molécule d'eau à 4 sites
    """
    def __init__(self, dim: int = None, T: int = None, state: np.ndarray = None):
        """
        Crée une molécule d'eau
        
        Paramètres:
        -----------
        - dim (int): Dimension de la molécule (2 ou 3)
        - T (float): Température (Utilisée lors de l'initialisation)
        - state (np.ndarray): État initial de la molécule (Optionnel)
        """

        if state is not None:
            # If an initial state is given from a previous simulation
            self.dim = state.shape[-1] 
            self.O_pos, self.H1_pos, self.H2_pos, self.M_pos = state[0], state[1], state[2], state[3]   # Initialization of atomic positions
            self.cm_vel, self.rot_vel = state[4], state[5]  # Initialization of molecular velocities
            
        else:
            # If no initial state is given 
            self.dim = dim
            self.T = T
            pos = init_func.init_water(dim=self.dim) # Initialization of atomic positions
            self.O_pos, self.H1_pos, self.H2_pos, self.M_pos = (pos[i] for i in range(len(pos)))
            self.__rand_orientation()   # Random molecular orientation
            self.cm_vel = init_func.random_velocity(h2O.M, T, self.dim)
            self.rot_vel = init_func.random_rot_velocity(self.inertia_tensor(), T, self.dim)
        
        # Atomic forces initialization
        self.O_force, self.H1_force, self.H2_force, self.M_force = (np.zeros(self.dim) for i in range(4))
    
    def cm_pos(self):
        """Retourne la position du centre de masse de la molécule en Å"""
        sum = h2O.mH * (self.H1_pos + self.H2_pos) + h2O.mO * self.O_pos
        return sum/h2O.M
    
    def rpos(self, M=False):
        """
        Calcule les positions relatives des atomes de la molécule par rapport au centre de masse.

        Input
        -----
        - M (bool): Si True, retourne également la position relative du pseudo-atome M

        Output
        ------
        - rpos (tuple): Positions relatives\n
            rpos[0] = O_rpos\n
            rpos[1] = H1_rpos\n
            rpos[2] = H2_rpos\n
            rpos[3] = M_rpos\n
        """
        cm = self.cm_pos()
        if M:
            return self.O_pos - cm, self.H1_pos - cm, self.H2_pos - cm, self.M_pos - cm
        else:
             return self.O_pos - cm, self.H1_pos - cm, self.H2_pos - cm

    def __rand_orientation(self):
        """Applique une rotation aléatoire sur la molécule"""
        cm = self.cm_pos()
        rotation_matrix = special_ortho_group.rvs(dim=self.dim)      
        r_pos = self.rpos(M = True)
        r_pos_rotated = np.array([rotation_matrix@r_pos[i] for i in range(4)])
        self.O_pos, self.H1_pos, self.H2_pos, self.M_pos = cm + r_pos_rotated[0], cm + r_pos_rotated[1], cm + r_pos_rotated[2], cm + r_pos_rotated[3]

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
        self.__rand_orientation()
        self.cm_vel = init_func.random_velocity(h2O.M, self.T, self.dim)
        self.rot_vel = init_func.random_rot_velocity(self.inertia_tensor(), self.T, self.dim)

    def reset_forces(self):
        self.O_force, self.H1_force, self.H2_force, self.M_force = (np.zeros(self.dim) for i in range(4))

    def kinetic_energy(self, *args):
        K = 0
        if "T" in args[0]:
            Kt = 1/2 * h2O.M * np.linalg.norm(self.cm_vel)**2
            K += Kt
        if "R" in args[0]:
            J = self.inertia_tensor()@self.rot_vel
            Kr = 1/2 * np.dot(self.rot_vel, J)
            K += Kr
        return K

    def correct_cm_pos(self):
        cm = self.cm_pos()
        self.O_pos -= np.floor(cm/simP.a) * simP.a
        self.H1_pos -= np.floor(cm/simP.a) * simP.a
        self.H2_pos -= np.floor(cm/simP.a) * simP.a
        self.M_pos -= np.floor(cm/simP.a) * simP.a


    def cm_force(self):
        force = self.O_force + self.H1_force + self.H2_force + self.M_force
        return force

    def torque(self):
        rpos = self.rpos(M=True)
        torque = np.cross(rpos[0], self.O_force) + np.cross(rpos[1], self.H1_force) + np.cross(rpos[2], self.H2_force) + np.cross(rpos[3], self.M_force)
        return torque

    def ang_momentum(self):
        return self.inertia_tensor()@self.rot_vel
    
    def omega_dot(self):
        inv = np.linalg.inv(self.inertia_tensor())
        return inv@(self.torque() - self.d_inertia_tensor()@self.rot_vel)
    
    def update_positions(self, R_n_1, omega_n_05, dt):
        rpos = self.rpos(M=True)
        rpos_new = np.zeros((4, self.dim))
        den = np.dot(omega_n_05, omega_n_05)
        for i in range(4):
            ai = np.dot(omega_n_05, rpos[i]) * omega_n_05 / den
            bi = rpos[i] - ai
            d_dot = np.cross(omega_n_05, rpos[i])
            phi = dt * np.linalg.norm(d_dot) / np.linalg.norm(bi)
            if np.degrees(phi) < 1:
                sinc = 1 - (phi**2)/6*(1-(phi**2)/20 * (1 - (phi**2)/42))
                bi_ = bi * np.cos(phi) + dt * d_dot * sinc
            else:
                bi_ = bi * np.cos(phi) + dt * d_dot * np.sin(phi)/phi
            d_n_1 = bi_ + ai
            rpos_new[i] = d_n_1
        self.O_pos = R_n_1 + rpos_new[0]
        self.H1_pos = R_n_1 + rpos_new[1]
        self.H2_pos = R_n_1 + rpos_new[2]
        self.M_pos = R_n_1 + rpos_new[3]

if __name__ == "__main__":
    print(np.floor(np.array([-0.1, 2.9, -5.6])))
