import numpy as np
import parameters.h2O_model as h2O
import parameters.simulation_parameters as simP
import parameters.physical_constants as pc


def random_pos(min=float, max=float, dim=int):
    out = np.random.uniform(min, max, dim)
    return out
    
def init_water(dim: int = 3):
    """
    Initialise une molécule d'eau avec l'atome d'oxygène centré à (0,0,0)

    Input
    -----
    - dim (int): Dimension de la molécule (2 ou 3)

    Return
    ------
    - pos (np.ndarray): Matrice de position des sites de la molécule d'eau\n
        pos[0] = O_pos\n
        pos[1] = H1_pos\n
        pos[2] = H2_pos\n
        pos[3] = H3_pos\n
    """
    if dim == 3:
        pos = np.zeros((4,3))
        pos[1] = h2O.l * simP.b1
        pos[2] = h2O.l*(np.cos(h2O.theta)*simP.b1 + np.sin(h2O.theta)*simP.b2)
        pos[3] = h2O.z*(np.cos(h2O.theta/2)*simP.b1 + np.sin(h2O.theta/2)*simP.b2)
        return pos
    elif dim == 2:
        b1 = np.array([1, 0])
        b2 = np.array([0, 1])
        pos = np.zeros((4,2))
        pos[1] = h2O.l * b1
        pos[2] = h2O.l*(np.cos(h2O.theta)*b1 + np.sin(h2O.theta)*b2)
        pos[3] = h2O.z*(np.cos(h2O.theta/2)*b1 + np.sin(h2O.theta/2)*b2)
        return pos

def random_velocity(T:float, M: float = h2O.M, dim: int = 3):
    """
    Retourne une vitesse aléatoire obtenue d'après une distribution de Maxwell-Boltzmann

    Input
    -----
    - T (float): Température cible utilisée pour générer la distribution de Maxwell-Boltzmann [K]
    - M (float): Masse de la particule [u]
    - dim (int): Dimension (2 ou 3)

    Output
    ------
    - v (np.array): Vitesse [Å/fs]
    """
    sigma = np.sqrt(pc.kb_SI*T/(M*pc.u))
    rand_v = np.random.normal(0, sigma, dim)
    return rand_v/100

def random_rot_velocity(I, T, dim = 3):
    Ip, R = np.linalg.eig(I)
    out = np.zeros(dim)
    for i in range(dim):
        sigma = np.sqrt(pc.kb_SI*T/(Ip[i] * pc.u * 1e-20))
        out[i] = np.random.normal(0, sigma, 1) / 1e12
    return R@out


    
if __name__ == "__main__":
    pass
