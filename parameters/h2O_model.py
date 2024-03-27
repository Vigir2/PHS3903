import numpy as np
import parameters.physical_constants as pc

q = 0.6791 #[e]

l = 0.8724 #[Ang]

z = 0.1594 #[Ang]

theta = np.radians(103.6) #[deg]

sigma = 3.16655 #[Ang]

epsilon_SI = 0.89036 #[Kj/mol]
epsilon = epsilon_SI / (pc.Na * pc.u * 10) #[u * ang^2/ps^2]

mO = 15.999 #[u]

mH = 1.00784 #[u]

M = 2*mH + mO #[u]
