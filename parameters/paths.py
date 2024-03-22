import os
from datetime import datetime

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