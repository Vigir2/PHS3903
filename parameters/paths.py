import os
from datetime import datetime
import numpy as np

output = "Output"
state_var = {"E": "Energy", "P": "Pressure", "T": "Temperature", "H": "Enthalpie", "t": 'Time'}

def traj_fname(name):
    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    fname = os.path.join(output, name, "Trajectories", name + "_" + date + ".xyz")
    if not os.path.exists(os.path.join(output, name, "Trajectories")):
        os.makedirs(os.path.join(output, name, "Trajectories"))
    return fname

def config_fname(name):
    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    fname = os.path.join(output, name, "Configuration", name + "_" + date + ".xyz")
    if not os.path.exists(os.path.join(output, name, "Configuration")):
        os.makedirs(os.path.join(output, name, "Configuration"))
    return fname

def state_log_fname(name):
    """Nom de fichier utilisé pour les log"""
    fname = os.path.join(output, name, "state_log", name + ".npy")
    if not os.path.exists(os.path.join(output, name, "state_log")):
        os.makedirs(os.path.join(output, name, "state_log"))
    return fname

def state_variables_fname(name: str, var: str):
    """Nom de fichier utilisé pour les variables d'états"""
    date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    fname = os.path.join(output, name, "State_variables", name + "_" + state_var[var] + "_" + date + ".npy")
    if not os.path.exists(os.path.join(output, name, "State_variables")):
        os.makedirs(os.path.join(output, name, "State_variables"))
    return fname

def gen_name():
    exist = True
    while exist:
        n = "ID" + str(round(np.random.uniform(0,1e3)))
        if os.path.exists(os.path.join(output, n)):
            continue
        else:
            exist = False
    return n

if __name__ == "__main__":
    pass