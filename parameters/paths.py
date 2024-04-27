import os
from datetime import datetime
import numpy as np

output = "Output"
state_var = {"E": "Energy", "P": "Pressure", "T": "Temperature", "H": "Enthalpy", "t": 'Time', "V": 'Volume'}

def traj_fname(name: str, format: str) -> str:
    """Nom de fichier utilisé pour enregistrer les trajectoires"""
    date = datetime.now().strftime('%Y-%m-%d_%H-%M')
    fname = os.path.join(output, name, "Trajectories", name + "_" + date + f".{format}")
    if not os.path.exists(os.path.join(output, name, "Trajectories")):
        os.makedirs(os.path.join(output, name, "Trajectories"))
    return fname

def config_fname(name: str) -> str:
    """Nom de fichier utilisé pour enregistrer les fichiers de configuration xyz"""
    date = datetime.now().strftime('%Y-%m-%d_%H-%M')
    fname = os.path.join(output, name, "Configuration", name + "_" + date + ".xyz")
    if not os.path.exists(os.path.join(output, name, "Configuration")):
        os.makedirs(os.path.join(output, name, "Configuration"))
    return fname

def state_log_fname(name: str) -> str:
    """Nom de fichier utilisé pour les logs"""
    fname = os.path.join(output, name, "state_log", name + ".npy")
    if not os.path.exists(os.path.join(output, name, "state_log")):
        os.makedirs(os.path.join(output, name, "state_log"))
    return fname

def state_variables_fname(name: str, var: str) -> str:
    """Nom de fichier utilisé pour les variables d'états"""
    date = datetime.now().strftime('%Y-%m-%d_%H-%M')
    fname = os.path.join(output, name, "State_variables", name + "_" + state_var[var] + "_" + date + ".npy")
    if not os.path.exists(os.path.join(output, name, "State_variables")):
        os.makedirs(os.path.join(output, name, "State_variables"))
    return fname

def gen_name() -> None:
    """Génère un nom aléatoire sous forme d'identifiant numérique unique pour le système"""
    exist = True
    while exist:
        n = "ID" + str(np.random.randint(0, 1e3))
        if os.path.exists(os.path.join(output, n)):
            continue
        else:
            exist = False
    return n

if __name__ == "__main__":
    print(datetime.now().isoformat())