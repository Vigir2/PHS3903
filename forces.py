import numpy as np
import parameters.h2O_model as h2O
import matplotlib.pyplot as plt
import parameters.physical_constants as pc

def lennard_jones(r: float, rc: float, shifted: bool = True):
    if shifted:
        out = 4 * h2O.epsilon * ((h2O.sigma / r)**12 - (h2O.sigma / r)**6 - (h2O.sigma / rc)**12 + (h2O.sigma / rc)**6)
        return out
    else:
        out = 4 * h2O.epsilon * ((h2O.sigma / r)**12 - (h2O.sigma / r)**6)
        return out
    
def lennard_jones_force(r: float):
    out = (24 * h2O.epsilon / r) * (2 * (h2O.sigma / r)**12 - (h2O.sigma / r)**6)
    return out

def coulomb(r: float, rc: float, q1: float, q2: float, shifted: bool = False):
    if shifted:
        out = pc.k * q1 * q2 * (1/r - 1/rc)
        return out
    else:
        out = pc.k * q1 * q2 * (1/r)
        return out
    
def coulomb_force(r: float, q1: float, q2: float):
    """
    Changer epsilon0 par k pour éviter le calcul répété
    """
    out = pc.k * q1 * q2 / r**2
    return out
    
if __name__ == "__main__":
    r = np.linspace(3,20,200)
    plt.plot(r, coulomb(r, rc=10, q1=h2O.q, q2=h2O.q))
    plt.plot(r, coulomb(r, rc=10, q1=h2O.q, q2=h2O.q, shifted=True))
    plt.plot(r, lennard_jones_force(r))
    #plt.ylim(-100, 100)
    plt.axhline(0)
    plt.show()