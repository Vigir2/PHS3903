import numpy as np
import log_strings as log 
import os
import parameters.paths as paths

def write_xyz_file(l, fname):
    with open(fname, "w") as f:
        f.write(f"{3*len(l)}\n\n")
        for m in l:
            f.write(f"O     {'      '.join([str(m.O_pos[i]) for i in range(m.dim)])}\n")
            f.write(f"H     {'      '.join([str(m.H1_pos[i]) for i in range(m.dim)])}\n")
            f.write(f"H     {'      '.join([str(m.H2_pos[i]) for i in range(m.dim)])}\n")
    print(log.write_file.format(fname=fname))

def write_trajectory(traj: np.ndarray, fname: str, dt: float, delta: float, a: float = None, format: str = "vtf", dim=3):
    if format == "xyz":
        with open(fname, "w") as f:
            n = 0
            for timestep in traj:
                f.write(f"{3*len(timestep)}\n")
                f.write(f"Timestep {n * dt * delta} ps\n")
                for m in timestep:
                    f.write(f"O     {'      '.join([str(m[0][i]) for i in range(dim)])}\n")
                    f.write(f"H     {'      '.join([str(m[1][i]) for i in range(dim)])}\n")
                    f.write(f"H     {'      '.join([str(m[2][i]) for i in range(dim)])}\n")
                n += 1
        print(log.write_file.format(fname=fname))
    if format == "vtf":
        with open(fname, "w") as f:
            f.write(f"atom 0:{traj.shape[1]-1}   name O\n")
            f.write(f"atom {traj.shape[1]}:{3*traj.shape[1] - 1}   name H\n")
            n = 0
            for timestep in traj:
                f.write("\ntimestep\n")
                if ((type(a) == float) or (type(a) == int)) and (n == 0):
                    f.write(f"pbc {a} {a} {a}\n")
                elif (type(a) != None) and (type(a) != float) and (type(a) != int): 
                    f.write(f"pbc {a[n]} {a[n]} {a[n]}\n")
                f.write(f"# {n * dt * delta} ps\n")
                for O in timestep[:,0]:
                    f.write("{x:.10f} {y:.10f} {z:.10f}\n".format(x=O[0], y=O[1], z=O[2]))
                for H1 in timestep[:,1]:
                    f.write("{x:.10f} {y:.10f} {z:.10f}\n".format(x=H1[0], y=H1[1], z=H1[2]))
                for H2 in timestep[:,2]:
                    f.write("{x:.10f} {y:.10f} {z:.10f}\n".format(x=H2[0], y=H2[1], z=H2[2]))
                n += 1
            



def write_state_log(state_log, fname):
    """Enregistre l'état du système dans un fichier .npy"""
    np.save(fname, state_log)
    print(log.write_file.format(fname=fname))

def write_state_variables(data: dict, name: str):
    """Enregistre les variables d'états du système sous forme d'array numpy"""
    for key, value in data.items():
        fname = paths.state_variables_fname(name=name, var=key)
        np.save(fname, value)
    print("\\".join(fname.split("\\")[:-1]))
    print(log.write_state_variables.format(var = ", ".join(list(data.keys())), loc = "\\".join(fname.split("\\")[:-1])))

if __name__ == "__main__":
    from H2O import H2O
    m = []
    for i in range(30):
        m.append(H2O(3))
        m[-1].rand_orientation()
        m[-1].rand_position()

    write_xyz_file(m)
