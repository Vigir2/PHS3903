import numpy as np
from datetime import datetime

def write_xyz_file(l, fname):
    with open(fname, "w") as f:
        f.write(f"{3*len(l)}\n\n")
        for m in l:
            f.write(f"O     {'      '.join([str(m.O_pos[i]) for i in range(m.dim)])}\n")
            f.write(f"H     {'      '.join([str(m.H1_pos[i]) for i in range(m.dim)])}\n")
            f.write(f"H     {'      '.join([str(m.H2_pos[i]) for i in range(m.dim)])}\n")
    print(f"File {fname} has been written succesfully")

def write_trajectory(traj, fname, dim=3):
    with open(fname, "w") as f:
        n = 1
        for timestep in traj:
            f.write(f"{3*len(timestep)}\n")
            f.write(f"Timestep {n}\n")
            for m in timestep:
                f.write(f"O     {'      '.join([str(m[0][i]) for i in range(dim)])}\n")
                f.write(f"H     {'      '.join([str(m[1][i]) for i in range(dim)])}\n")
                f.write(f"H     {'      '.join([str(m[2][i]) for i in range(dim)])}\n")
            n += 1
    print(f"File {fname} has been written succesfully")

if __name__ == "__main__":
    from H2O import H2O
    m = []
    for i in range(30):
        m.append(H2O(3))
        m[-1].rand_orientation()
        m[-1].rand_position()

    write_xyz_file(m)
