import numpy as np
from Universe import Universe
import parameters.simulation_parameters as simP
from itertools import repeat
from multiprocessing import Pool
import shutil
import os
import time

def generate_equilibrium_state(name: str = None, N: int = None, T: float = None, P: float = None, a: float = None, n: int = 3) -> tuple:
    name_eq = name + f"_equilibrium_T{T}_P{P}"
    U = Universe(name=name_eq, N=N, a=a, T=T)
    U.npt_integration(dt=1.2, n=n, delta=2, T0=T, P0=P, Ewald=True, graphs=False, T=True, P=True, V=True)
    shutil.move(os.path.join("Output", name_eq), os.path.join("Output", name, "EQ"))
    out_state, a = U.get_state()
    return out_state, a

def enthalpie_measurement(T: float, name: str, input_state: np.ndarray, a: float, P: float, n_npt: int = 6, n_nvt: int = 3) -> float:
    name_npt = name + f"_npt_T{T}_P{P}"
    U = Universe(name=name_npt, a=a, input_state=input_state)
    data = U.npt_integration(dt=1, n=n_npt, delta=2, T0=T, P0=P, Ewald=True, graphs=False, H=True, P=True, T=True, V=True, E=True)
    shutil.move(os.path.join("Output", name_npt), os.path.join("Output", name, "NPT"))
    a_mean = np.mean(data["V"][len(data["V"])//2:])**(1/3)
    input_state = U.get_state()[0]
    name_nvt = name + f"_nvt_T{T}_P{P}"
    U = Universe(name=name_nvt, a=a_mean, input_state=input_state)
    data = U.nvt_integration(dt=1, n=n_nvt, delta=1, T0=T, P0=P, Ewald=True, graphs=False, H=True, P=True, T=True, V=True, E=True)
    shutil.move(os.path.join("Output", name_nvt), os.path.join("Output", name, "NVT"))
    H = np.mean(data["H"])
    return H

def H_vs_T(name: str, N: int, T: np.ndarray, P: float, a0: float):
    input_state, a = generate_equilibrium_state(name=name, N=N, T=T[0], P=P, a=a0)
    with Pool() as pool:
        H = pool.starmap(enthalpie_measurement, zip(T, repeat(name), repeat(input_state), repeat(a), repeat(P)))
    np.save(os.path.join("Output", name, "H_vs_T"), np.array(H))
    np.save(os.path.join("Output", name, "T"), np.array(T))
    return H

if __name__ == "__main__":
    name = "Eau_liquide_H_vs_T"
    N = 100
    P = 10
    a0 = 18
    T = np.arange(300, 430, 10)
    print(T)
    tic = time.time()
    H = H_vs_T(name=name, N=N, T=T, P=P, a0=a0)
    toc = time.time()
    print(f"Time = {(toc-tic)/3600} h")
    print(H)

    

