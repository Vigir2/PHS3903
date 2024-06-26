import numpy as np
import parameters.h2O_model as h2O
import parameters.physical_constants as pc
import math

def lennard_jones(r: float, rc: float, shifted: bool = True) -> float:
    """Retourne l'énergie associée au potentiel de Lennard-Jones en [u * Å^2 / fs^2]"""
    if shifted:
        out = 4 * h2O.epsilon * ((h2O.sigma / r)**12 - (h2O.sigma / r)**6 - (h2O.sigma / rc)**12 + (h2O.sigma / rc)**6)
        return out
    else:
        out = 4 * h2O.epsilon * ((h2O.sigma / r)**12 - (h2O.sigma / r)**6)
        return out
    
def lennard_jones_force(r: float) -> float:
    """Retourne la force associée au potentiel de L-J en [u * Å / fs^2]"""
    out = (24 * h2O.epsilon / r) * (2 * (h2O.sigma / r)**12 - (h2O.sigma / r)**6)
    return out

def coulomb(r: float, rc: float, q1: float, q2: float, shifted: bool = False) -> float:
    """Retourne l'énergie de l'interraction électrostatique calculée avec le potentiel de Coulomb en [u * Å^2 / fs^2]"""
    if shifted:
        out = pc.k * q1 * q2 * (1/r - 1/rc)
        return out
    else:
        out = pc.k * q1 * q2 * (1/r)
        return out
    
def coulomb_force(r: float, q1: float, q2: float) -> float:
    """Retourne la force de Coulomb entre deux particules chargées en [u * Å / fs^2]"""
    out = pc.k * q1 * q2 / r**2
    return out

def ewald_electrostatic(r, alpha, q1, q2):
    """Retourne le potentiel électrostatique pour la sommation d'Ewald dans l'espace réel"""
    out = 1 / (4 * np.pi * pc.epsilon0) * (q1 * q2 / r) * math.erfc(alpha * r)
    return out

def ewald_electrostatic_correction(r: float, alpha: float, q1: float, q2: float):
    """Retourne la correction de rigidité de l'interraction par sommation d'Ewald"""
    out = 1 / (4 * np.pi * pc.epsilon0) * (q1 * q2 / r) * math.erf(alpha * r)
    return out

def ewald_electrostatic_force(r, alpha, q1, q2):
    """Retourne la force électrostatique pour la sommation d'Ewald dans l'espace réel"""
    out = (q1 * q2) / (4 * np.pi * pc.epsilon0 * r**2) * (math.erfc(alpha*r)/r + 2*alpha/np.sqrt(np.pi) * np.exp(-alpha**2 * r**2))
    return out

def ewald_electrostatic_force_correction(r: float, alpha: float, q1: float, q2: float):
    """Retourne la force associé au potentiel de correction pour la rigidité dans la sommation d'Ewald dans l'espace réel"""
    out = (q1 * q2) / (4 * np.pi * pc.epsilon0 * r**2) * (math.erf(alpha * r)/r - 2*alpha/np.sqrt(np.pi) * np.exp(-alpha**2 * r**2))
    return out

    
if __name__ == "__main__":
    pass
    #r = np.linspace(3,20,200)
    #plt.plot(r, coulomb(r, rc=10, q1=h2O.q, q2=h2O.q))
    #plt.plot(r, coulomb(r, rc=10, q1=h2O.q, q2=h2O.q, shifted=True))
    #plt.plot(r, lennard_jones_force(r))
    ##plt.ylim(-100, 100)
    #plt.axhline(0)
    #plt.show()
    print(pc.k)