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

def write_trajectory(traj: np.ndarray, fname: str, dt: float, delta: float, dim=3):
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
