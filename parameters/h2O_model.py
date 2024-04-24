import numpy as np
import parameters.physical_constants as pc
from pint import UnitRegistry
ureg = UnitRegistry()

q = 0.6791 #[e]

l = 0.8724 #[Å]

z = 0.1594 #[Å]

theta = np.radians(103.6) #[deg]

sigma = 3.16655 #[Å]

epsilon_SI = 0.89036 #[Kj/mol]
epsilon = epsilon_SI * pc.KJ_mol_to_uÅfs #[u * Å^2/fs^2]

mO = 15.999 #[u]

mH = 1.00784 #[u]

M = 2*mH + mO #[u]


