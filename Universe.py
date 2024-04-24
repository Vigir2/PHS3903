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
import time


class Universe:
    """Univers de simulation"""
    def __init__(self, name: str = None, N: int = None, a: float = None, T: float = None, dim: int = 3, input_state: np.ndarray = None) -> None:
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
            self.water_molecules = [H2O(dim=self.dim, T=T) for _ in range(N)]
            # Gives random positions to molecules in the simulation box ensuring a security distance between each molecule
            for m in self.water_molecules:
                n = 0
                m.rand_position(a = self.a)
                if hasattr(self, "cm"):
                    while (np.any([np.linalg.norm(integ.minimum_image(cm - m.cm_pos(), a = self.a)) < simP.security_distance for cm in self.cm])):
                        m.rand_position(a = self.a)
                        n += 1
                        if n >= 75:
                            print(log.init_error.format(name = self.name) + log.error_water_placement.format(N = self.N, a = self.a, security_distance = simP.security_distance))
                            sys.exit()
                    self.cm = np.append(self.cm, m.cm_pos().reshape(1,self.dim), axis=0)
                else:
                    self.cm = np.array([m.cm_pos()])
            delattr(self, "cm")
        self.__remove_net_momentum()
        self.compute_forces()
        self.temp = self.temperature("T", "R")
        self.pressure = self.pression()
        print(log.init_universe.format(name = self.name, N = self.N, T = self.temp, P = self.pressure * pc.uÅfs_to_bar), end="\n\n")

    def __getitem__(self, index) -> H2O:
        return self.water_molecules[index]

    def temperature(self, *args: str) -> float:
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
        if len(args) > 2:
            print(log.error_temperature.format(a = args))
            sys.exit()
        f = {3: {'t': 3, 'r': 3}, 2: {'t': 2, 'r': 1}}
        arg = [m.lower() for m in args]
        Nf = 0
        N_correction = 0
        for key in arg:
            Nf += f[self.dim][key]
            if key == 't':
                N_correction = f[self.dim]['t']
        K = 0
        for m in self.water_molecules:
            K += m.kinetic_energy(args)
        return 2 * K / ((Nf * self.N - N_correction) * pc.kb)

    
    def pression(self, K: float = None) -> float:
        """
        Calcule la pression du système en u / (Å * fs^2)
        
        Input
        -----
        - K (float): Énergie cinétique du système [Optionnel]

        Output
        ------
        - P (float): Pression en u / (Å * fs^2)
        """
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

    def cm_position(self) -> np.ndarray:
        """Retourne la position du centre de masse du système en Å"""
        r = 0
        for m in self.water_molecules:
            r += m.cm_pos()
        return r / self.N
    
    def system_momentum(self) -> np.ndarray:
        """Retourne lq quantité de mouvement globale du système en u*Å/fs"""
        p = 0
        for m in self.water_molecules:
            p += h2O.M * m.cm_vel
        return p
    
    def __remove_net_momentum(self) -> None:
        """Ajuste les vitesses des molécules du système pour avoir une quantité de mouvement totale nulle"""
        sys_cm_vel = self.system_momentum() / (self.N * h2O.M)
        for m in self.water_molecules:
            m.cm_vel -= sys_cm_vel
    
    def snapshot(self) -> None:
        """Enregistre la position actuelle des molécules du système afin d'obtenir une trajectoire"""
        snap = np.zeros((1, self.N, 3, self.dim))
        for i in range(self.N):
            m = self.water_molecules[i]
            snap[0][i] = np.array([m.O_pos, m.H1_pos, m.H2_pos])
        if hasattr(self, "trajectories"):
            self.trajectories = np.append(self.trajectories, snap, axis=0)
        else:
            self.trajectories = snap
    
    def __write_trajectories(self, dt: float, delta: float, a: np.ndarray, format: str = "vtf") -> None:
        """
        Écrit la trajectoire des molécules du système dans un fichier .xyz ou .vtf"""
        out_func.write_trajectory(self.trajectories, fname=paths.traj_fname(name=self.name, format=format), dt=dt, a=a, delta=delta, format=format)
   
    def __write_xyz(self) -> None:
        """Enregistre la configuration actuelle du système dans un fichier .xyz"""
        out_func.write_xyz_file(self.water_molecules, fname=paths.config_fname(name=self.name))

    def __save_state_log(self) -> None:
        """Enregistre l'état actuel du système dans un format pouvant être lu pour initialiser une nouvelle simulation"""
        out_func.write_state_log(self.water_molecules, paths.state_log_fname(self.name))

    def __save_state_variables(self, data: dict) -> None:
        """Enregistre les variables d'états du système en fonction du temps"""
        out_func.write_state_variables(data=data, name=self.name)

    def compute_forces(self, Ewald: bool = False) -> None:
        """Calcule les forces du système, l'énergie potentielle et le virriel"""
        if simP.rc <= self.a/2:
            rc = self.a/2
        else:
            rc = simP.rc
        if Ewald == False:
            integ.compute_forces(U=self, rc=rc, a=self.a)
        else:
            pass
    
    def energy(self) -> float:
        """Retourne l'énergie totale du système en [uÅ^2/fs^2]"""
        K = 0
        for m in self.water_molecules:
            K += m.kinetic_energy("T", "R")
        V = self.potential_energy
        self.total_energy = K+V, K, V
        return K+V, K, V

    def correct_position(self) -> None:
        """Ajuste la position des molécules pour qu'elles soient dans la cellule de simulation"""
        for m in self.water_molecules:
            m.correct_cm_pos(a = self.a)
    
    def nve_integration(self, dt: float, n: int, delta: int = 1, *args: str) -> None:
        """
        Effectue une intégration des équation du mouvement pour un ensemble NVE avec un algorithme de Verlet vitesse

        Input
        -----
        - dt (float): Pas de temps utilisé pour l'intégration [fs]
        - n (int): Nombre de pas
        - delta (int): Intervalles de pas de temps auxquels les cordonnées sont enregistrées
        - E, T, P (str): Quantités à mesurées pendant l'intégration
        """
        print(log.nve_initiation.format(time=(dt*n/1000), n=n), end="\n\n")
        data = dict()
        E, T, P = False, False, False
        if "E" in args:
            E = True
            data["E"] = []
        if "T" in args:
            T = True
            data["T"] = []
        if "P" in args:
            P = True
            data["P"] = []
        for i in range(int(n)):
            print(f"n = {i}, t = {i * dt} fs")
            print(np.array(self.energy()) / self.N * pc.uÅfs_to_KJ_mol, end="\n\n")
            if i%delta == 0:
                self.snapshot()
                if E:
                    data["E"].append(self.total_energy)
                if T:
                    data["T"].append(self.temperature("T", "R"))
                if P:
                    data["P"].append(self.pression())
            integ.nve_verlet_run(U=self, dt=dt)
            self.correct_position()
            self.__remove_net_momentum()
        data['t'] = np.arange(0, n*dt, delta)
        self.__write_trajectories(dt=dt, delta=delta, format="vtf", a=self.a)
        self.__save_state_variables(data=data)
        self.__save_state_log()
    
    def npt_integration(self, dt: float, n: int, delta: int, T0: float, P0: float, *args: str) -> None:
        """
        Effectue une intégration des équations du mouvement pour un ensemble NPT avec un algorithme de Verlet vitesse
        
        Input
        -----
        - dt (float): Pas de temps utilisé pour l'intégration [fs]
        - n (int): Nombre de pas
        - delta (int): Intervalles de pas de temps auxquels les cordonnées sont enregistrées
        - T0 (float): Température cible [K]
        - P0 (float): Pression cible [bar]
        - E, H, T, P, V (str): Quantités à mesurées pendant l'intégration
        """
        print(log.npt_initiation.format(time=(dt*n/1000), n=n, T = T0, P = P0))
        P0 *= pc.bar_to_uÅfs
        data = dict()
        a = []
        E, T, P, H, V = False, False, False, False, False
        if "E" in args:
            E = True
            data["E"] = []
        if "T" in args:
            T = True
            data["T"] = []
        if "P" in args:
            P = True
            data["P"] = []
        if "H" in args:
            H = True
            data["H"] = []
        if "V" in args:
            V = True
            data["V"] = []

        Nf = lambda dim: 6 if dim == 3 else 3
        for i in range(int(n)):
            print(f"n = {i}")
            print(f"a = {self.a}")
            print("T = ", self.temp, self.temperature("t"), self.temperature("r"))
            print("P = ", self.pressure * pc.uÅfs_to_bar, end="\n\n")
            if i%delta == 0:
                self.snapshot()
                a.append(self.a)
                if E:
                    data["E"].append(self.energy())
                if T:
                    data["T"].append(self.temp)
                if P:
                    data["P"].append(self.pressure)
                if H:
                    if E:
                        data["H"].append(data["E"][-1] + self.pressure * self.a**3)
                    else:
                        data["H"].append(self.energy() + self.pressure * self.a**3)
                if V:
                    data["V"].append(self.a**3)
            integ.npt_verlet_run(U=self, dt=dt, T0=T0, P0=P0, Nf=Nf(self.dim))
            self.correct_position()
            self.__remove_net_momentum()
        data['t'] = np.arange(0, n*dt, delta)
        self.__write_trajectories(dt=dt, delta=delta, a=a, format="vtf")
        self.__save_state_log()
        self.__save_state_variables(data=data)
    
"""     def ewald_nve_integration(self, n, delta, dt):
        E = pc.kb * self.T
        #print("E = ", E)
        delta1, delta2 = simP.delta1, simP.delta2
        delta1 *= E
        delta2 *= E
        alpha = 1/simP.rc * np.sqrt(-np.log((4 * np.pi * pc.epsilon0 * simP.rc * delta1)/(2*h2O.q)**2)) #[Å^-1]
        Smax = (self.N * 4 * h2O.q)**2 #[e^2]
        V = self.a**3
        kmax2 = -4 * alpha**2 * np.log((2*pc.epsilon0 * V * delta2)/Smax) #[Å^-2]
        u = 2 * np.pi * simP.b1 / self.a
        v = 2 * np.pi * simP.b2 / self.a
        w = 2 * np.pi * simP.b3 / self.a
        rbasis = np.array([u, v, w])
        umax = int(np.ceil(np.sqrt(kmax2/np.dot(u,u))))
        vmax = int(np.ceil(np.sqrt(kmax2/np.dot(v,v))))
        lambmax = int(np.ceil(np.sqrt(kmax2/np.dot(w,w))))
        ewald_correction = alpha / (4 * np.pi**(3/2) * pc.epsilon0) * self.N * 6 * h2O.q**2
        integ.compute_forces_ewald(self, simP.rc, self.a, alpha, umax, vmax, lambmax, rbasis, ewald_correction)
        for i in range(int(n)):
            print(f"n = {i}")
            print(self.energy(), end="\n\n")
            if i%delta == 0:
                self.snapshot()
                integ.nve_verlet_run(self, dt, True, simP.rc, alpha, umax, vmax, lambmax, rbasis, ewald_correction)
            U.correct_position()
        self.__write_trajectories(dt=dt, delta=delta, format="vtf", a=self.a) """



if __name__ == "__main__":
    U = Universe(name = simP.name, N = simP.N, T = simP.T, a = simP.a, dim=simP.dim)
    #U.npt_integration(1, 4000, 1, 150, 1, "V", "P", "T")
    #U._Universe__write_xyz()
    #U = Universe(name="test_glace", input_state="Output\Test_integration_5000\state_log\Test_integration_5000.npy")
    U.nve_integration(dt=1, n=12, delta=1)
    #U = Universe()
    #U = Universe("testvtf", a=simP.a, N=simP.N, T=simP.T, dim=3)
    #print(U.pression())
    #U.npt_integration(dt = 1, n = 500, delta = 2, T0 = 200, P0 = 1)
    #U.nve_integration(1, 15, 1)
    #U = Universe("test", 100, 25, 300, 3)
    #U.ewald_nve_integration(100, 1, 2)
    #U.ewald_npt_integration(100, 0.001, 0.001)
    #U.nve_integration(1, 10, 1)
    #U.npt_integration(dt = 1, n = 10, delta = 1, T0 = 200, P0 = 1)
    #U.ewald_npt_integration(100, 0.000001, 0.000001)
    #integ.compute_forces(U, rc=simP.rc, a=U.a)


