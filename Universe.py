import numpy as np
from H2O import H2O
import output_functions as out_func
import init_functions as init_func
import sys
import parameters.paths as paths
import parameters.physical_constants as pc
import parameters.h2O_model as h2O
import log_strings as log
import parameters.simulation_parameters as simP
import integration as integ
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
        self.compute_forces(Ewald=False)
        self.temp = self.temperature("T", "R")
        self.pressure = self.pression()
        print(log.init_universe.format(name = self.name, N = self.N, T = self.temp, P = self.pressure * pc.uÅfs_to_bar), end="\n\n")

    def __getitem__(self, index) -> H2O:
        return self.water_molecules[index]
    
    def get_state(self) -> tuple:
        """
        Retourne l'état actuel du système
        
        Output
        -----
        - state (np.ndarray): État du système
        """
        out = np.zeros((U.N, 6, U.dim))
        for i in range(self.N):
            out[i][0] = self.water_molecules[i].O_pos
            out[i][1] = self.water_molecules[i].H1_pos
            out[i][2] = self.water_molecules[i].H2_pos
            out[i][3] = self.water_molecules[i].M_pos
            out[i][4] = self.water_molecules[i].cm_vel
            out[i][5] = self.water_molecules[i].rot_vel
        return out, self.a

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

    def compute_forces(self, Ewald: bool = False, parallel: bool = False) -> None:
        """Calcule les forces du système, l'énergie potentielle et le virriel"""
        if simP.rc <= self.a/2:
            rc = self.a/2
        else:
            rc = simP.rc
        if Ewald == False:
            integ.compute_forces(U=self, rc=rc, a=self.a)
        else:
            alpha, c_max, rbasis = init_func.ewald_parameters(rc=rc, a=simP.a, N=self.N)
            integ.compute_forces_ewald(U=self, rc=rc, a=self.a, alpha=alpha, c_max=c_max, rbasis=rbasis, parallel=parallel)
    
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
    
    def nve_integration(self, dt: float, n: int, delta: int = 1, Ewald: bool = False, parallel: bool = False, graphs: bool = True, **kwargs: bool) -> None:
        """
        Effectue une intégration des équation du mouvement pour un ensemble NVE avec un algorithme de Verlet vitesse

        Input
        -----
        - dt (float): Pas de temps utilisé pour l'intégration [fs]
        - n (int): Nombre de pas
        - delta (int): Intervalles de pas de temps auxquels les cordonnées sont enregistrées
        - Ewald (bool): Utilise la méthode de la sommation d'Ewald pour le calcul des interractions électrostatiques
        - graphs (bool): Trace en temps réel les variables thermodynamiques sélectionnées
        - E, T, P (bool): Quantités à mesurées pendant l'intégration
        """
        print(log.nve_initiation.format(time=(dt*n/1000), n=n), end="\n\n")
        if Ewald:
            self.ewald_correction = 1/(4*np.pi**(3/2)*pc.epsilon0) * self.N * (2*h2O.q**2 + (2*h2O.q)**2)
        data = dict()
        
        if "E" in kwargs:
            if kwargs["E"]:
                data["E"] = []
        if "T" in kwargs:
            if kwargs["T"]:
                data["T"] = []
        if "P" in kwargs:
            if kwargs["P"]:
                data["P"] = []

        if graphs:
            data_length = len(data)
            units = {"E": r"KJ mol$^{{-1}}$", "T": "K", "P": "bar"}
            plt.ion()
            fig, axes = plt.subplots(data_length, 1, sharex=True)
            for index, value in enumerate(data):
                axes[index].set_title(value)
                axes[index].plot([],[])
                axes[index].set_ylabel(f"{value}  ({units[value]})")
                axes[index].axhline(y = 0, linestyle='--', color = 'k', linewidth=0.5)
            axes[-1].set_xlabel("t (fs)")
            plt.subplots_adjust(hspace=0.5)

        tic = time.time()
        for i in range(int(n)):
            print(f"n = {i}, t = {i * dt} fs")
            print(np.array(self.energy()) / self.N * pc.uÅfs_to_KJ_mol, end="\n\n")

            if i%delta == 0:
                self.snapshot()
                if "E" in data:
                    data["E"].append(self.total_energy[0]/self.N*pc.uÅfs_to_KJ_mol)
                if "T" in data:
                    data["T"].append(self.temperature("T", "R"))
                if "P" in data:
                    data["P"].append(self.pression()*pc.uÅfs_to_bar)
                if graphs:
                    for index, key in enumerate(data):
                        axes[index].lines[0].set_ydata(data[key])
                        axes[index].lines[0].set_xdata(np.arange(0, i + 1, delta))
                        axes[index].lines[1].remove()
                        axes[index].axhline(y = np.mean(data[key]), linestyle='--', color='k')
                        axes[index].set_title(f"{np.mean(data[key]):.3f} {units[key]}")
                        axes[index].relim()
                        axes[index].autoscale_view()
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    time.sleep(0.05)

            integ.nve_verlet_run(U=self, dt=dt, Ewald=Ewald, parallel=parallel)
            self.correct_position()
            self.__remove_net_momentum()
        toc = time.time()
        print(log.simulation_completed.format(time=(toc-tic)/3600))
        data['t'] = np.arange(0, n*dt, delta)
        self.__write_trajectories(dt=dt, delta=delta, format="vtf", a=self.a)
        self.__save_state_variables(data=data)
        self.__save_state_log()
    
    def npt_integration(self, dt: float, n: int, delta: int, T0: float, P0: float, Ewald: bool = False, graphs: bool = True, **kwargs: str) -> None:
        """
        Effectue une intégration des équations du mouvement pour un ensemble NPT avec un algorithme de Verlet vitesse
        
        Input
        -----
        - dt (float): Pas de temps utilisé pour l'intégration [fs]
        - n (int): Nombre de pas
        - delta (int): Intervalles de pas de temps auxquels les cordonnées sont enregistrées
        - T0 (float): Température cible [K]
        - P0 (float): Pression cible [bar]
        - Ewald (bool): Utilise la méthode de sommation d'Ewald pour le calcul des interractions électrostatiques
        - graphs (bool): Trace en temps réel les variables thermodynamiques sélectionnées
        - E, H, T, P, V (bool): Quantités à mesurées pendant l'intégration
        """
        print(log.npt_initiation.format(time=(dt*n/1000), n=n, T = T0, P = P0))
        if Ewald:
            self.ewald_correction = 1/(4*np.pi**(3/2)*pc.epsilon0) * self.N * (2*h2O.q**2 + (2*h2O.q)**2)
        P0 *= pc.bar_to_uÅfs
        data = dict()
        a = []
        if "E" in kwargs:
            if kwargs["E"]:
                data["E"] = []
        if "T" in kwargs:
            if kwargs["T"]:
                data["T"] = []
        if "P" in kwargs:
            if kwargs["P"]:
                data["P"] = []
        if "H" in kwargs:
            if kwargs["H"]:
                data["H"] = []
        if "V" in kwargs:
            if kwargs["V"]:
                data["V"] = []
        
        if graphs:
            data_length = len(data)
            units = {"E": r"KJ mol$^{{-1}}$", "T": "K", "P": "bar", "V": r"Å$^3$", "H": r"KJ mol$^{{-1}}$"}
            plt.ion()
            fig, axes = plt.subplots(data_length, 1, sharex=True)
            for index, value in enumerate(data):
                axes[index].set_title(value)
                axes[index].plot([],[])
                axes[index].set_ylabel(f"{value}  ({units[value]})")
                axes[index].axhline(y = 0, linestyle='--', color = 'k', linewidth=0.5)
            axes[-1].set_xlabel("t (fs)")
            plt.subplots_adjust(hspace=0.5)

        Nf = lambda dim: 6 * U.N - 3 if dim == 3 else 3 * U.N - 2
        tic = time.time()
        for i in range(int(n)):
            print(f"n = {i}, t = {i * dt} fs")
            print(f"a = {self.a}")
            print(f"T = {self.temp} K")
            print(f"P = {self.pressure * pc.uÅfs_to_bar} bar", end="\n\n")
            if i%delta == 0:
                self.snapshot()
                a.append(self.a)
                if "E" in data:
                    data["E"].append(self.energy()[0] / self.N * pc.uÅfs_to_KJ_mol)
                if "T" in data:
                    data["T"].append(self.temp)
                if "P" in data:
                    data["P"].append(self.pressure * pc.uÅfs_to_bar)
                if "H" in data:
                    if "E" in data:
                        data["H"].append(data["E"][-1] + (self.pressure * self.a**3) / self.N * pc.uÅfs_to_KJ_mol)
                    else:
                        data["H"].append((self.energy()[0] + self.pressure * self.a**3) / self.N * pc.uÅfs_to_KJ_mol)
                if "V" in data:
                    data["V"].append(self.a**3)
                if graphs:
                    for index, key in enumerate(data):
                        axes[index].lines[0].set_ydata(data[key])
                        axes[index].lines[0].set_xdata(np.arange(0, i + 1, delta))
                        axes[index].lines[1].remove()
                        axes[index].axhline(y = np.mean(data[key]), linestyle='--', color='k')
                        axes[index].set_title(f"{np.mean(data[key]):.3f} {units[key]}")
                        axes[index].relim()
                        axes[index].autoscale_view()
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    time.sleep(0.05)
            integ.npt_verlet_run(U=self, dt=dt, T0=T0, P0=P0, Nf=Nf(self.dim), Ewald=Ewald)
            self.correct_position()
            self.__remove_net_momentum()
        toc = time.time()
        print(log.simulation_completed.format(time=(toc-tic)/3600))
        data['t'] = np.arange(0, n*dt, delta)
        self.__write_trajectories(dt=dt, delta=delta, a=a, format="vtf")
        self.__save_state_log()
        self.__save_state_variables(data=data)
    
    def nvt_integration(self, dt: float, n: int, delta: int, T0: float, Ewald: bool = False, graphs: bool = True, **kwargs: str) -> None:
        """
        Effectue une intégration des équations du mouvement pour un ensemble NPT avec un algorithme de Verlet vitesse
        
        Input
        -----
        - dt (float): Pas de temps utilisé pour l'intégration [fs]
        - n (int): Nombre de pas
        - delta (int): Intervalles de pas de temps auxquels les cordonnées sont enregistrées
        - T0 (float): Température cible [K]
        - P0 (float): Pression cible [bar]
        - Ewald (bool): Utilise la méthode de sommation d'Ewald pour le calcul des interractions électrostatiques
        - graphs (bool): Trace en temps réel les variables thermodynamiques sélectionnées
        - E, H, T, P, V (bool): Quantités à mesurées pendant l'intégration
        """
        print(log.nvt_initiation.format(time=(dt*n/1000), n=n, T = T0))
        if Ewald:
            self.ewald_correction = 1/(4*np.pi**(3/2)*pc.epsilon0) * self.N * (2*h2O.q**2 + (2*h2O.q)**2)
        data = dict()
        a = []
        if "E" in kwargs:
            if kwargs["E"]:
                data["E"] = []
        if "T" in kwargs:
            if kwargs["T"]:
                data["T"] = []
        if "P" in kwargs:
            if kwargs["P"]:
                data["P"] = []
        if "H" in kwargs:
            if kwargs["H"]:
                data["H"] = []
        if "V" in kwargs:
            if kwargs["V"]:
                data["V"] = []
        
        if graphs:
            data_length = len(data)
            units = {"E": r"KJ mol$^{{-1}}$", "T": "K", "P": "bar", "V": r"Å$^3$", "H": r"KJ mol$^{{-1}}$"}
            plt.ion()
            fig, axes = plt.subplots(data_length, 1, sharex=True)
            for index, value in enumerate(data):
                axes[index].set_title(value)
                axes[index].plot([],[])
                axes[index].set_ylabel(f"{value}  ({units[value]})")
                axes[index].axhline(y = 0, linestyle='--', color = 'k', linewidth=0.5)
            axes[-1].set_xlabel("t (fs)")
            plt.subplots_adjust(hspace=0.5)

        Nf = lambda dim: 6 * U.N - 3 if dim == 3 else 3 * U.N - 2
        tic = time.time()
        for i in range(int(n)):
            print(f"n = {i}, t = {i * dt} fs")
            print(f"T = {self.temp} K")
            if i%delta == 0:
                self.snapshot()
                a.append(self.a)
                if "E" in data:
                    data["E"].append(self.energy()[0] / self.N * pc.uÅfs_to_KJ_mol)
                if "T" in data:
                    data["T"].append(self.temp)
                if "P" in data:
                    data["P"].append(self.pression() * pc.uÅfs_to_bar)
                if "H" in data:
                    if "E" in data:
                        data["H"].append(data["E"][-1] + (self.pression() * self.a**3) / self.N * pc.uÅfs_to_KJ_mol)
                    else:
                        data["H"].append((self.energy()[0] + self.pression() * self.a**3) / self.N * pc.uÅfs_to_KJ_mol)
                if "V" in data:
                    data["V"].append(self.a**3)
                if graphs:
                    for index, key in enumerate(data):
                        axes[index].lines[0].set_ydata(data[key])
                        axes[index].lines[0].set_xdata(np.arange(0, i + 1, delta))
                        axes[index].lines[1].remove()
                        axes[index].axhline(y = np.mean(data[key]), linestyle='--', color='k')
                        axes[index].set_title(f"{np.mean(data[key]):.3f} {units[key]}")
                        axes[index].relim()
                        axes[index].autoscale_view()
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    time.sleep(0.05)
            integ.nvt_verlet_run(U=self, dt=dt, T0=T0, Nf=Nf(self.dim), Ewald=Ewald)
            self.correct_position()
            self.__remove_net_momentum()
        toc = time.time()
        print(log.simulation_completed.format(time=(toc-tic)/3600))
        data['t'] = np.arange(0, n*dt, delta)
        self.__write_trajectories(dt=dt, delta=delta, a=a, format="vtf")
        self.__save_state_log()
        self.__save_state_variables(data=data)
    
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



if __name__ == "__main__":
    U = Universe(name = simP.name, N = simP.N, T = simP.T, a = simP.a, dim=simP.dim)

    #U.nve_integration(dt=1, n=50, delta=1, Ewald=True, parallel=True, graphs=True, E=True, T = True, P = True)
    #x = input()
    #U.nve_integration(dt=1, n=50, delta=1, Ewald=True, parallel=False, graphs=True, E=True, T = True, P = True)
    #U = Universe(name="glace_formation", a=14.87832291048, input_state="Output\\test_ewald\\state_log\\test_ewald.npy")
    U.npt_integration(dt = 1, n = 5000, delta = 1, T0 = 300, P0 = 5, graphs=False, T=True, P=True, V=True, E=True, Ewald=True)
    #U.nve_integration(dt=1, n=2, delta=1, graphs=False, E=False, P=False, T=False)
    #U.npt_integration(1, 4000, 1, 150, 1, "V", "P", "T")
    #U._Universe__write_xyz()
    #U = Universe(name="test_glace", input_state="Output\Test_integration_5000\state_log\Test_integration_5000.npy")
    #U = Universe()
    #U = Universe("testvtf", a=simP.a, N=simP.N, T=simP.T, dim=3)
    #print(U.pression())
    #U.nve_integration(1, 15, 1)
    #U = Universe("test", 100, 25, 300, 3)
    #U.ewald_nve_integration(100, 1, 2)
    #U.ewald_npt_integration(100, 0.001, 0.001)
    #U.nve_integration(1, 10, 1)
    #U.npt_integration(dt = 1, n = 10, delta = 1, T0 = 200, P0 = 1)
    #U.ewald_npt_integration(100, 0.000001, 0.000001)
    #integ.compute_forces(U, rc=simP.rc, a=U.a)
    # U.ewald_correction = 1/(4*np.pi**(3/2)*pc.epsilon0) * U.N * (2*h2O.q**2 + (2*h2O.q)**2)
    # U.compute_forces(Ewald=True, parallel=True)
    # print(U.energy())
    # U.compute_forces(Ewald=True, parallel=False)
    # print(U.energy())
    # U.compute_forces(Ewald=True, parallel=True)
    # print(U.energy())
    # U.compute_forces(Ewald=True, parallel=False)
    # print(U.energy())
    # U.compute_forces(Ewald=True)
    # print(U.energy())


