import numpy as np
from H2O import H2O
import output_functions as out_func
import sys
import parameters.paths as paths
import parameters.physical_constants as pc
import parameters.h2O_model as h2O
import log_strings as log
import parameters.simulation_parameters as simP
import integration as integ


class Universe:
    def __init__(self, name: str, N: int = None, cell: float = None, T: float = 100, P: float = 1, dim: int = 3, input_state: np.ndarray = None):
        
        if not name or type(name) != str:
            self.name = paths.gen_name() 
        else:
            self.name = name

        self.cell = cell
        self.dim = dim
        self.T = T
        self.P = P
        self.trajectories = None

        if input_state != None:
            if type(input_state) == str:
                input_state = np.load(input_state)
            self.N = input_state.shape[0]
            self.dim = input_state.shape[-1]
            self.water_molecules = []
            for i in input_state:
                self.water_molecules.append(H2O(state = i))

        else:
            self.N = N
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
                        print(log.init_error.format(name = self.name) + log.error_water_placement.format(N = self.N, a = simP.a, security_distance = simP.security_distance))
                        sys.exit()
        self.__remove_net_momentum()
        self.thermo, self.baro = 0, 0
        print(log.init_universe.format(name = self.name, N = self.N, T = self.temperature("T", "R")))

    def temperature(self, *args):
        K = 0
        if ("T" in args) ^ ("R" in args):
            for m in self.water_molecules:
                K += m.kinetic_energy(args)
            return 2* K / (3 * self.N * pc.kb_SI * 1e-4 / pc.u)
        if ("T" in args) and ("R" in args):
            for m in self.water_molecules:
                K += m.kinetic_energy(args)
            return K / (3 * self.N * pc.kb_SI * 1e-4 / pc.u)
    
    def pression(self):
        """
        À revoir
        """
        V = 0
        K = 0
        for m in self.water_molecules:
            V += np.dot(m.cm_pos(), m.cm_force())
            K += m.kinetic_energy("T")
        P = 1/(3*simP.a**2) * (2 * K + V)
        return pc.u * 1e10 * (1e12)**2 * P

    def cm_position(self):
        s = 0
        for m in self.water_molecules:
            s += m.cm_pos()
        return s / self.N
    
    def system_momentum(self):
        s = 0
        for m in self.water_molecules:
            s += h2O.M * m.cm_vel
        return s
    
    def __remove_net_momentum(self):
        sys_vel = self.system_momentum() / (self.N * h2O.M)
        for m in self.water_molecules:
            m.cm_vel -= sys_vel
    
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
    
    def write_trajectories(self, dt: float, delta: float):
        out_func.write_trajectory(self.trajectories, fname=paths.traj_fname(name=self.name), dt=dt, delta=delta)
   
    def write_xyz(self):
        out_func.write_xyz_file(self.water_molecules, fname=paths.config_fname(name=self.name))

    def save_state_log(self):
        out = np.zeros((self.N, 6, self.dim))
        for i in range(self.N):
            self.water_molecules[i].correct_cm_pos()
            out[i][0] = self.water_molecules[i].O_pos
            out[i][1] = self.water_molecules[i].H1_pos
            out[i][2] = self.water_molecules[i].H2_pos
            out[i][3] = self.water_molecules[i].M_pos
            out[i][4] = self.water_molecules[i].cm_vel
            out[i][5] = self.water_molecules[i].rot_vel
        out_func.write_state_log(out, paths.state_log_fname(self.name))

    def compute_forces(self):
        V = integ.compute_forces(self.water_molecules, rc=simP.rc, a=simP.a)
        return V
    
    def energy(self):
        K = 0
        for m in self.water_molecules:
            K += m.kinetic_energy("T", "R")
        V = self.compute_forces()
        return K+V, K, V, 2*K + V

    def correct_position(self):
        for m in self.water_molecules:
            m.correct_cm_pos()
    
    def nve_integration(self, dt: float, n: int, delta: int):
        self.compute_forces()
        for i in range(n):
            print(i)
            if i%delta == 0:
                U.snapshot()
            integ.nve_verlet_run(U=self, dt=dt)
            U.correct_position()
        U.write_trajectories(dt=dt, delta=delta)
        U.save_state_log()
        U.write_xyz()


if __name__ == "__main__":
    U = Universe(N = simP.N, T = simP.T, P = simP.P, cell = simP.a, dim=simP.dim, name=simP.name)
    #U = Universe(name="test_glace", input_state="Output\Test_integration_5000\state_log\Test_integration_5000.npy")
    U.nvt_integration(dt=0.002, n=300, delta=1)
    



    """
    for i in range(20):
        for j in U.water_molecules:
            j.rand_orientation()
        U.snapshot()
    U.write_trajectories()
    """

