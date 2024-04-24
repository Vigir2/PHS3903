import numpy as np
import log_strings as log 
import parameters.paths as paths

def write_xyz_file(water_molecules: list, fname: str) -> None:
    """
    Écrit la configuration du sytème dans un format xyz pouvant être visualisé avec des logiciels comme Vesta ou VMD

    Input
    -----
    water_molecules (list): Liste des molécules du système
    fname (str): Nom du fichier de sortie
    """
    with open(fname, "w") as f:
        f.write(f"{3*len(water_molecules)}\n\n")
        for m in water_molecules:
            f.write(f"O     {'      '.join([str(m[0][i]) for i in range(m.dim)])}\n")
            f.write(f"H     {'      '.join([str(m[1][i]) for i in range(m.dim)])}\n")
            f.write(f"H     {'      '.join([str(m[2][i]) for i in range(m.dim)])}\n")
    print('\n' + log.write_file.format(fname=fname))

def write_trajectory(traj: np.ndarray, fname: str, dt: float, delta: float, a: float = None, format: str = "vtf", dim: int = 3) -> None:
    """
    Écrit les trajectoires du système dans un fichier vtf ou xyz pouvant être visualisé sur VMD

    Input
    -----
    - traj (np.ndarray): Vecteur des trajectoires des molécules du système
    - fname (str): Nom du fichier de sortie
    - dt (float): Pas de temps utilisé lors de la simulation [fs]
    - delta (int): Intervalle de pas de temps auquel les positions sont enregistrées
    - a (float | np.ndarray): taille de la cellule de simulation [Å]
    - format (str): Format du fichier de sortie
    - dim (int): Dimension du système
    """
    if format == "xyz":
        with open(fname, "w") as f:
            n = 0
            for timestep in traj:
                f.write(f"{3*len(timestep)}\n")
                f.write(f"t = {n * dt * delta / 1000} ps\n")
                for m in timestep:
                    f.write(f"O     {'      '.join([str(m[0][i]) for i in range(dim)])}\n")
                    f.write(f"H     {'      '.join([str(m[1][i]) for i in range(dim)])}\n")
                    f.write(f"H     {'      '.join([str(m[2][i]) for i in range(dim)])}\n")
                n += 1
        print('\n' + log.write_file.format(fname=fname))
    if format == "vtf":
        with open(fname, "w") as f:
            f.write(f"atom 0:{traj.shape[1]-1}   name O\n")
            f.write(f"atom {traj.shape[1]}:{3*traj.shape[1] - 1}   name H\n")
            n = 0
            for timestep in traj:
                f.write("\ntimestep\n")
                if ((type(a) == float) or (type(a) == int)) and (n == 0):
                    f.write(f"pbc {a} {a} {a}\n")
                elif (type(a) == list) or (type(a) == np.ndarray): 
                    f.write(f"pbc {a[n]} {a[n]} {a[n]}\n")
                f.write(f"# t = {n * dt * delta / 1000} ps\n")
                for O in timestep[:,0]:
                    f.write("{x:.10f} {y:.10f} {z:.10f}\n".format(x=O[0], y=O[1], z=O[2]))
                for H1 in timestep[:,1]:
                    f.write("{x:.10f} {y:.10f} {z:.10f}\n".format(x=H1[0], y=H1[1], z=H1[2]))
                for H2 in timestep[:,2]:
                    f.write("{x:.10f} {y:.10f} {z:.10f}\n".format(x=H2[0], y=H2[1], z=H2[2]))
                n += 1
        print('\n' + log.write_file.format(fname=fname))



def write_state_log(water_molecules: list, fname: str):
    """
    Écrit l'état d'un système dans un fichier npy pouvant être donné à un objet Univers pour initialiser une nouvelle simulation.

    Input
    -----
    - water_molecules (list): Liste des molécules du système
    - fname (str): Nom du fichier de sortie
    """
    out = np.zeros((len(water_molecules), 6, water_molecules[0].dim))
    for i in range(len(water_molecules)):
        out[i][0] = water_molecules[i].O_pos
        out[i][1] = water_molecules[i].H1_pos
        out[i][2] = water_molecules[i].H2_pos
        out[i][3] = water_molecules[i].M_pos
        out[i][4] = water_molecules[i].cm_vel
        out[i][5] = water_molecules[i].rot_vel
    np.save(fname, out)
    print('\n' + log.write_file.format(fname=fname))

def write_state_variables(data: dict, name: str) -> None:
    """
    Enregistre les variables d'états du système en fichiers npy.
    
    Input
    -----
    - data (dict): Dictionnaire des vecteurs de variables d'état
    - name (str): Nom du système
    """
    for key, value in data.items():
        fname = paths.state_variables_fname(name=name, var=key)
        np.save(fname, value)
    print('\n' + log.write_state_variables.format(var = ", ".join(list(data.keys())), loc = "\\".join(fname.split("\\")[:-1])))

if __name__ == "__main__":
    from H2O import H2O
    m = []
    for i in range(30):
        m.append(H2O(3))
        m[-1].rand_orientation()
        m[-1].rand_position()

    write_xyz_file(m)
