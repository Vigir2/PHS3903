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
    sigma = np.sqrt(pc.kb*T/M)
    rand_v = np.random.normal(0, sigma, dim)
    return rand_v

def random_rot_velocity(T: float, I: np.ndarray, dim: int = 3):
    """
    Retourne une vitesse angulaire aléatoire obtenue d'après une distribution de Maxwell-Boltzmann

    Input
    -----
    - T (float): Température cible utilisée pour générer la distribution de Maxwell-Boltzmann [K]
    - I (np.ndarray): Tenseur d'inertie de la molécule [u*Å^2]
    - dim (int): Dimension (2 ou 3)

    Output
    ------
    - omega (np.array): Vitesse [1/fs]
    """
    Ip, R = np.linalg.eig(I)
    out = np.zeros(dim)
    for i in range(dim):
        sigma = np.sqrt(pc.kb*T/Ip[i])
        out[i] = np.random.normal(0, sigma, 1)
    return R@out


    
if __name__ == "__main__":
    print(np.linalg.norm(random_velocity(300)))
    import numpy as np

    # Constants
    k = 1.38e-23  # Boltzmann constant in J/K
    T = 300       # Temperature in Kelvin
    m = 18.01528e-3 / 6.022e23  # Mass of a water molecule in kg

    # Calculate root-mean-square speed
    v_rms = np.sqrt(3 * k * T / m)

    print("Root-mean-square speed of water molecules at 300K:", v_rms * 1e-5, "m/s")

    
