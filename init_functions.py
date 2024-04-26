import numpy as np
import parameters.h2O_model as h2O
import parameters.simulation_parameters as simP
import parameters.physical_constants as pc


def random_pos(min: float, max: float, dim: int):
    """Génère un vecteur déplacement aléatoire entre les positions min et max"""
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

def ewald_parameters(rc: float, a: float, N: int) -> tuple:
    """
    Calcule les paramètres à utiliser pour le calcul des forces électrostatiques par sommation d'Ewald

    Input
    -----
    - rc (float): Rayon de coupure utilisé pour le calcul des forces de Lennard-Jones
    - a (float): Longueur d'un côté de la cellule de simulation
    - N (int): Nombre de molécules dans la simulation

    Output
    ------
    - alpha (float): Paramètre de convergence de la sommation d'Ewald
    - c_max (int): Coefficient maximal des vecteurs pour la sommation dans le réseau réciproque
    - rbasis (tuple): Vecteurs du réseau réciproque
    """

    delta1 = simP.delta1 * pc.KJ_mol_to_uÅfs * N
    delta2 = simP.delta2 * pc.KJ_mol_to_uÅfs * N

    u = 2 * np.pi / a * np.array([1, 0, 0])
    v = 2 * np.pi / a * np.array([0, 1, 0])
    w = 2 * np.pi / a * np.array([0, 0, 1])

    alpha = 1/rc * np.sqrt(-np.log((4*np.pi*pc.epsilon0*rc*delta1)/(2*h2O.q)**2))
    S_max = (N * 4 * h2O.q)**2
    kmax_squared = -4 * alpha**2 * np.log((2 * pc.epsilon0 * a**3 * delta2) / (S_max))
    coeff_max_squared = kmax_squared * (a / (2 * np.pi))**2
    return alpha, int(np.ceil(np.sqrt(coeff_max_squared))), [u, v, w]
    
if __name__ == "__main__":
    print(np.linalg.norm(random_velocity(300)))
    import numpy as np

    # Constants
    k = 1.38e-23  # Boltzmann constant in J/K
    T = 300       # Temperature in Kelvin
    m = 18.01528e-3 / 6.022e23  # Mass of a water molecule in kg
    I = 1e-47 / pc.u * 10**20
    #print(I)
    # Calculate root-mean-square speed
    v_rms = np.sqrt(3 * k * T / m)
    print(v_rms)
    #omega_rms = np.sqrt()
    omega_rms = np.sqrt(3*pc.kb*300/I)
    print(omega_rms)
    print(np.sqrt(2)/omega_rms)

    print("Root-mean-square speed of water molecules at 300K:", v_rms * 1e-5, "m/s")

    
