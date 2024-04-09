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
    """Univers de simulation"""
    def __init__(self, name: str = None, N: int = None, a: float = None, T: float = None, dim: int = 3, input_state: np.ndarray = None):
        """
        Crée un univers de simulation

        Input
        -----
        - name (str): Nom du système
        - N (int): Nombre de molcules d'eau du système
        - a (float): Longueur du côté de la cellule de simulation cubique [Å]
        - T (float): Température d'initialisation du système [K]
        - dim (int): Dimension du système (2 ou 3)
        - input_state (np.ndarray ou path): État d'entré du système obtenu à partir d'une simulation précédente
        """
        # System name
        if not name or type(name) != str:
            self.name = paths.gen_name() 
        else:
            self.name = name
        self.a = a
        self.thermo, self.baro = 0, 0

        # If an input state is given
        if input_state != None:
            if type(input_state) == str:
                input_state = np.load(input_state)
            self.N = input_state.shape[0]
            self.dim = input_state.shape[-1]
            self.water_molecules = []
            for i in input_state:
                self.water_molecules.append(H2O(state = i))
        # If an input state is not given
        else:
            self.dim = dim
            self.N = N
            self.T = T
            self.water_molecules = [H2O(dim=self.dim, T=T) for i in range(N)]
            # Gives random positions to molecules in the simulation box ensuring a security distance between each molecule
            for m in self.water_molecules:
                n = 0
                m.rand_position(a = self.a)
                if hasattr(self, "cm"):
                    while (np.any([np.linalg.norm(integ.minimum_image(cm - m.cm_pos(), a = self.a)) < simP.security_distance for cm in self.cm])):
                        m.rand_position(a = self.a)
                        n += 1
                        if n >= 50:
                            print(log.init_error.format(name = self.name) + log.error_water_placement.format(N = self.N, a = self.a, security_distance = simP.security_distance))
                            sys.exit()
                    self.cm = np.append(self.cm, m.cm_pos().reshape(1,self.dim), axis=0)
                else:
                    self.cm = np.array([m.cm_pos()])
            delattr(self, "cm")
        self.__remove_net_momentum()
        self.compute_forces()
        print(log.init_universe.format(name = self.name, N = self.N, T = self.temperature("T", "R"), P = self.pression() / pc.bar_to_uÅfs))

    def __getitem__(self, index):
        return self.water_molecules[index]

    def temperature(self, *args):
        """
        Calcul la température du système à partir de son énergie cinétique moyenne
        
        Input
        -----
        - "T": Inclue la température translationnelle
        - "R": Inclue la température rotationnelle
        
        Output
        ------
        - temperature (float): Température du système [K]
        """
        K = 0
        arg = [m.lower() for m in args]
        if ("t" in arg) ^ ("r" in arg):
            for m in self.water_molecules:
                K += m.kinetic_energy(args)
            return 2 * K / (3 * self.N * pc.kb)
        if ("t" in arg) and ("r" in arg):
            for m in self.water_molecules:
                K += m.kinetic_energy(args)
            return K / (3 * self.N * pc.kb)
    
    def pression(self, K: float = None):
        """Calcule la pression du système en u / (Å * fs^2)"""
        if K != None:
            v = self.virial
            for m in self.water_molecules:
                v -= m.virial_correction()
            P = 1/(3*self.a**3) * (2 * K + v)
            return P
        else:
            K = 0
            v = self.virial
            for m in self.water_molecules:
                K += m.kinetic_energy("T")
                v -= m.virial_correction()
            P = 1/(3*self.a**3) * (2 * K + v)
            return P

    def cm_position(self):
        """Retourne la position du centre de masse du système en Å"""
        r = 0
        for m in self.water_molecules:
            r += m.cm_pos()
        return r / self.N
    
    def system_momentum(self):
        """Retourne lq quantité de mouvement globale du système en u*Å/fs"""
        p = 0
        for m in self.water_molecules:
            p += h2O.M * m.cm_vel
        return p
    
    def __remove_net_momentum(self):
        """Ajuste les vitesses des molécules du système pour avoir une quantité de mouvement totale nulle"""
        sys_cm_vel = self.system_momentum() / (self.N * h2O.M)
        for m in self.water_molecules:
            m.cm_vel -= sys_cm_vel
    
    def snapshot(self):
        """Enregistre la position actuelle des molécules du système afin d'obtenir une trajectoire"""
        snap = np.zeros((1, self.N, 3, self.dim))
        for i in range(self.N):
            m = self.water_molecules[i]
            snap[0][i] = np.array([m.O_pos, m.H1_pos, m.H2_pos])
        if hasattr(self, "trajectories"):
            self.trajectories = np.append(self.trajectories, snap, axis=0)
        else:
            self.trajectories = snap
    
    def write_trajectories(self, dt: float, delta: float):
        """
        Écrit la trajectoire des molécules du système dans un fichier .xyz
        """
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
        """Calcule les forces du système, l'énergie potentielle et le virriel"""
        integ.compute_forces(U=self, rc=simP.rc, a=self.a)
    
    def energy(self):
        """Retourne l'énergie totale du système en [uÅ^2/fs^2]"""
        K = 0
        for m in self.water_molecules:
            K += m.kinetic_energy("T", "R")
        V = self.potential_energy
        return K+V, K, V

    def correct_position(self):
        """Ajuste la position des molécules pour qu'elles soient dans la cellule de simulation"""
        for m in self.water_molecules:
            m.correct_cm_pos(a = self.a)
    
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
    
    def npt_integration(self, dt: float, n: int, delta: int, T0: float, P0: float):
        P0 *= pc.bar_to_uÅfs
        self.compute_forces()
        for i in range(int(n)):
            print(i)
            print(f"a = {self.a}")
            if i%delta == 0:
                U.snapshot()
            integ.npt_verlet_run(U=self, dt=dt, T0=T0, P0=P0)
            U.correct_position()
            print("T = ", U.temperature("T"), U.temperature("T", "r"))
            print("P = ", U.pression() * pc.uÅfs_to_bar)
        U.write_trajectories(dt=dt, delta=delta)
        U.save_state_log()
        U.write_xyz()

if __name__ == "__main__":
    U = Universe(name = simP.name, N = simP.N, T = simP.T, a = simP.a, dim=simP.dim)
    #U = Universe(name="test_glace", input_state="Output\Test_integration_5000\state_log\Test_integration_5000.npy")
    #U.nve_integration(dt=1, n=150, delta=1)
    #U = Universe()
    #U = Universe("testvtf", a=simP.a, N=simP.N, T=simP.T, dim=3)
    #print(U.pression())
    U.npt_integration(dt = 2, n = 2000, delta = 1, T0 = 100, P0 = 100)
    


