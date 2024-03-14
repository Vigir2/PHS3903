import numpy as np
import parameters.h2O_model as h2O
import parameters.simulation_parameters as simP

def random_pos(min=float, max=float, dim=int):
    out = np.random.uniform(min, max, dim)
    return out

def random_euler_angles(dim=int, matrix=True):
    phi, theta, psi = np.random.uniform(-np.pi, np.pi, dim)
    if matrix:
        out = np.array([[np.cos(psi)*np.cos(phi)-np.sin(psi)*np.cos(theta)*np.sin(phi), -(np.sin(psi)*np.cos(phi) + np.cos(psi)*np.cos(theta)*np.sin(phi)), np.sin(theta)*np.sin(phi)],
                        [np.cos(psi)*np.sin(phi)+np.sin(psi)*np.cos(theta)*np.cos(phi), -(np.sin(psi)*np.sin(phi)-np.cos(psi)*np.cos(theta)*np.cos(phi)), -np.sin(theta)*np.cos(phi)],
                        [np.sin(theta)*np.sin(psi), np.sin(theta)*np.cos(psi), np.cos(theta)]])
        return out
    else:
        return phi, theta, psi
    
def init_water(dim=int):
    if dim == 3:
        pos = np.zeros((4,3))
        pos[1] = h2O.l * simP.b1
        pos[2] = h2O.l*(np.cos(h2O.theta)*simP.b1 + np.sin(h2O.theta)*simP.b2)
        pos[3] = h2O.z*(np.cos(h2O.theta/2)*simP.b1 + np.sin(h2O.theta/2)*simP.b2)
        return pos
    
    
if __name__ == "__main__":
    random_euler_angles(3, True)