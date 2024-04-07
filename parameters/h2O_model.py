import numpy as np
import parameters.physical_constants as pc

q = 0.6791 #[e]

l = 0.8724 #[Å]

z = 0.1594 #[Å]

theta = np.radians(103.6) #[deg]

sigma = 3.16655 #[Å]

epsilon_SI = 0.89036 #[Kj/mol]
epsilon = epsilon_SI * 1000 / (pc.Na) * pc.joules_to_uÅfs #[u * ang^2/fs^2]

mO = 15.999 #[u]

mH = 1.00784 #[u]

M = 2*mH + mO #[u]
