import numpy as np
import init_functions as init_func
import parameters.h2O_model as h2O
import parameters.simulation_parameters as simP

class H2O:
    def __init__(self, dim, pos = None):
        self.dim = dim
        self.pos = pos
        if self.pos:
            pass
        else:
            pos = init_func.init_water(dim=self.dim)
            self.O_pos, self.H1_pos, self.H2_pos, self.M_pos = (pos[i] for i in range(len(pos)))
    
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

if __name__ == "__main__":
    m = H2O(3)
    print(m.O_pos, m.H1_pos, m.H2_pos)
    m.rand_orientation()
    print(m.O_pos, m.H1_pos, m.H2_pos)
