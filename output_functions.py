import numpy as np
from datetime import datetime

def write_xyz_file(l, fname=f"fuck_you.xyz"):
    with open(fname, "w") as f:
        f.write(f"{3*len(l)}\n\n")
        for m in l:
            print(m)
            f.write(f"O     {'      '.join([str(m.O_pos[i]) for i in range(m.dim)])}\n")
            f.write(f"H     {'      '.join([str(m.H1_pos[i]) for i in range(m.dim)])}\n")
            f.write(f"H     {'      '.join([str(m.H2_pos[i]) for i in range(m.dim)])}\n")


if __name__ == "__main__":
    from H2O import H2O
    m = []
    for i in range(10):
        m.append(H2O(3))
        m[-1].rand_orientation()
        m[-1].rand_position()

    write_xyz_file(m)
