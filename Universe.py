import numpy as np
from H2O import H2O
import output_functions as out_func
import sys
import os
import parameters.paths as paths
from datetime import datetime

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
        date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        fname = os.path.join(paths.trajectories, self.name, self.name + "_Traj_" + date + ".xyz")
        if not os.path.exists(os.path.join(paths.trajectories, self.name)):
            os.makedirs(os.path.join(paths.trajectories, self.name))
        out_func.write_trajectory(self.trajectories, fname=fname)
   
    def write_xyz(self):
        date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        out_func.write_xyz_file(self.water_molecules, os.path.join(paths.coords, self.name, "Coords_" + date +".xyz"))
        return

if __name__ == "__main__":
    U = Universe(N=10, T = 100, P = 1, cell = 1, dim=3, name="Test")
    for i in range(20):
        for j in U.water_molecules:
            j.rand_orientation()
        U.snapshot()
    U.write_trajectories()