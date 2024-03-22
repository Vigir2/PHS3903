import numpy as np
from H2O import H2O
import output_functions as out_func
import sys
import parameters.paths as paths
import parameters.physical_constants as pc
import parameters.h2O_model as h2O

class Universe:
    def __init__(self, N, T, P, cell, pos=None, vel=None, dim = 3, name=None):
        self.N = N
        self.T = T
        self.P = P
        self.cell = cell
        self.dim = dim
        self.trajectories = None

        if not name:
            self.name = str(np.random.uniform(0,1e6))
        else:
            self.name = name

        if pos:
            if vel:
                pass
        else:
            self.water_molecules = [H2O(self.dim, T=T) for i in range(N)]
            self.water_molecules[0].rand_position()
            cm = [self.water_molecules[0].cm_pos()]
            for m in self.water_molecules[1:]:
                key = True
                n = 0
                while key:
                    m.rand_position()
                    err = 0
                    for i in cm:
                        if np.linalg.norm((m.cm_pos() - i)) <= 3:
                            err +=1
                    if err == 0:
                        cm.append(m.cm_pos())
                        key = False
                    else:
                        m.reset_molecule()
                        n += 1
                    if n >= 50:
                        print("Error in placing water molecules in the cell volume!")
                        sys.exit()

    def temperature(self, *args):
        K = 0
        for m in self.water_molecules:
            K += m.kinetic_energy(args)
        return K / (3 * self.N * pc.kb * 1e-4 / pc.u)
    
    def cm_position(self):
        s = 0
        for m in self.water_molecules:
            s += m.cm_pos()
        return s / self.N
    
    def snapshot(self, vel = False):
        if not vel:
            snap = np.zeros((1, self.N, 4, self.dim))
            for i in range(self.N):
                m = self.water_molecules[i]
                snap[0][i] = np.array([m.O_pos, m.H1_pos, m.H2_pos, m.M_pos])
            if self.trajectories is None:
                self.trajectories = snap
            else:
                self.trajectories = np.append(self.trajectories, snap, axis=0)
    
    def write_trajectories(self):
        out_func.write_trajectory(self.trajectories, fname=paths.traj_fname(name=self.name))
   
    def write_xyz(self):
        out_func.write_xyz_file(self.water_molecules, fname=paths.config_fname(name=self.name))

if __name__ == "__main__":
    U = Universe(N=100, T = 300, P = 1, cell = 1, dim=3, name="Test")
    print(U.cm_position())
    """
    for i in range(20):
        for j in U.water_molecules:
            j.rand_orientation()
        U.snapshot()
    U.write_trajectories()
    """

