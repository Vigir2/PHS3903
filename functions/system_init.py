import numpy as np
import h20_model as h20
import simulation_parameters as simP

def random_pos(min=float, max=float, dim=int):
    out = np.random.uniform(min, max, dim)
    return out

def random_euler_angles(dim=int, matrix=True):
    phi, theta, psi = np.random.uniform(-np.pi, np.pi, dim)
    if matrix:
        out = np.arary([[np.cos(psi)*np.cos(phi)-np.sin(psi)*np.cos(theta)*np.sin(theta), -(np.sin(psi)*np.cos(phi) + np.cos(psi)*np.cos(theta)*np.sin(phi)), np.sin(theta)*np.sin(phi)],
                        [np.cos(psi)*np.sin(phi)+np.sin(psi)*np.cos(theta)*np.cos(phi), -(np.sin(psi)*np.sin(theta)-np.cos(psi)*np.cos(theta)*np.cos(phi)), -np.sin(theta)*np.cos(phi)],
                        [np.sin(theta)*np.sin(psi), np.sin(theta)*np.cos(psi), np.cos(theta)]])
        return out
    else:
        return phi, theta, psi
    
def init_water(dim=int):
    if dim == 3:
        pos = np.zeros((4,3))
        pos[1] = h20.l * simP.b1
        pos[2] = h20.l*(np.cos(h20.theta)*simP.b1 + np.sin(h20.theta)*simP.b2)
        pos[3] = h20.z*(np.cos(h20.theta/2)*simP.b1 + np.sin(h20.theta/2)*simP.b2)
        return pos
    
if __name__ == "__main__":
    print("Hello World")
    atoms = ['O', 'H', 'H', 'H']
    with open("h2O.xyz", "w") as file:
        file.write("4\n")
        pos = init_water(3)
        for i in range(len(pos)):
            file.write(f"{atoms[i]}     {'  '.join([str(j) for j in pos[i]])}\n")