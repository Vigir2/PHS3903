import os
from datetime import datetime
import numpy as np

output = "Output"

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
    fname = os.path.join(output, name, "state_log", name + ".npy")
    if not os.path.exists(os.path.join(output, name, "state_log")):
        os.makedirs(os.path.join(output, name, "state_log"))
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