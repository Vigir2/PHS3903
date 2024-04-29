import numpy as np
from Universe import Universe
import parameters.simulation_parameters as simP
from itertools import repeat
from multiprocessing import Pool
import shutil
import os
import time

def generate_equilibrium_state(name: str = None, N: int = None, T: float = None, P: float = None, a: float = None, n: int = 7000) -> tuple:
    name_eq = name + f"_equilibrium_T{T}_P{P}"
    U = Universe(name=name_eq, N=N, a=a, T=T)
    U.npt_integration(dt=1.2, n=n, delta=2, T0=T, P0=P, Ewald=True, graphs=False, T=True, P=True, V=True)
    shutil.move(os.path.join("Output", name_eq), os.path.join("Output", name, "EQ"))
    out_state, a = U.get_state()
    return out_state, a

def enthalpie_measurement(T: float, name: str, input_state: np.ndarray, a: float, P: float, n_npt: int = 4000, n_nvt: int = 3000) -> float:
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
    T = np.mean(data["T"])
    return H, T

def H_vs_T(name: str, N: int, T: np.ndarray, P: float, a0: float):
    input_state, a = generate_equilibrium_state(name=name, N=N, T=T[0], P=P, a=a0)
    with Pool() as pool:
        Results = pool.starmap(enthalpie_measurement, zip(T, repeat(name), repeat(input_state), repeat(a), repeat(P)))
    Results = np.array(Results)
    np.save(os.path.join("Output", name, "H"), Results[:,0])
    np.save(os.path.join("Output", name, "T"), Results[:,1])
    return Results

def only_nvt(T, name, P):
    for file in os.listdir(os.path.join("Output", name, "NPT", name + f"_npt_T{T}_P{P}", "State_variables")):
        if "Volume" in file:
            V = np.load(os.path.join("Output", name, "NPT", name + f"_npt_T{T}_P{P}", "State_variables", file))
            a = np.mean(V[len(V)//2:])**(1/3)
    input_state = os.path.join("Output", name, "NPT", name + f"_npt_T{T}_P{P}", "state_log", name + f"_npt_T{T}_P{P}.npy")
    name_nvt = name + f"_nvt_T{T}_P{P}"
    U = Universe(name=name_nvt, a=a, input_state=input_state)
    data = U.nvt_integration(dt=1, n=3000, delta=1, T0=T, P0=P, Ewald=True, graphs=False, H=True, P=True, T=True, V=True, E=True)
    shutil.move(os.path.join("Output", name_nvt), os.path.join("Output", name, "NVT"))
    H = np.mean(data["H"])
    return H

def time_vs_N(N, name, T0, a, Ewald):
    name_t = name + f"_N{N}"
    U = Universe(name=name_t, N=N, a=a, T=T0)
    tic = time.time()
    U.nve_integration(dt=1, n=300, delta=1, Ewald=Ewald, graphs=False, E=True)
    toc = time.time()
    shutil.move(os.path.join("Output", name_t), os.path.join("Output", name, name_t))
    return (toc-tic)

def nvt_measurement(T, name, N, a, Ewald):
    name_nvt = name + f"_T{T}"
    U = Universe(name=name_nvt, N=N, a=a, T=10)
    data = U.nvt_integration(dt=1.5, n=5000, delta=1, T0=T, Ewald=Ewald, graphs=False, T=True, E=True, P=True)
    shutil.move(os.path.join("Output", name_nvt), os.path.join("Output", name, name_nvt))
    E = np.mean(data["E"][len(data["E"])//2:])
    T = np.mean(data["T"][len(data["T"])//2:])
    return E, T

""" if __name__ == "__main__":
    name = "Eau_liquide_H_vs_T_new"
    N = 100
    P = 10
    a0 = 18
    T = np.arange(300, 430, 10)
    print(T)
    tic = time.time()
    Results = H_vs_T(name=name, N=N, T=T, P=P, a0=a0)
    toc = time.time()
    print(f"Time = {(toc-tic)/3600} h")
    print(Results[:,0])
    print(Results[:,1]) """

""" if __name__ == "__main__":
    name = "Temps_Ewald_NVE_a30_rc15_no_parallel"
    N = np.array([20, 40, 60, 80, 100, 125, 150, 175, 200, 250, 300])
    print(N)
    a0 = 30
    T = 300
    with Pool() as pool:
        t = pool.starmap(time_vs_N, zip(N, repeat(name), repeat(T), repeat(a0), repeat(False)))
    t = list(map(time_vs_N, N, repeat(name), repeat(T), repeat(a0), repeat(True)))
    print(f"t = {t}")
    np.save(os.path.join("Output", name, "t_vs_N.npy"), np.array(t))
    np.save(os.path.join("Output", name, "N.npy"), N)
 """
        
if __name__ == "__main__":
    name = "test"
    a = 17
    N = 96
    
    # T = np.linspace(10, 100, 12)
    # T = np.linspace(110, 200, 12)
    # T = np.linspace(210, 300, 12)
    # T = np.linspace(310, 400, 12)
    # T = np.linspace(410, 500, 12)

    print(f"{T = }")
    Ewald = True
    with Pool(6) as pool:
        Results = pool.starmap(nvt_measurement, zip(T, repeat(name), repeat(N), repeat(a), repeat(Ewald)))
    Results = np.array(Results)
    E = Results[:,0]
    T = Results[:,1]
    print(f"{E = }")
    print(f"{T = }")
    np.save(os.path.join("Output", name, "E.npy"), E)
    np.save(os.path.join("Output", name, "T.npy"), T)


    

