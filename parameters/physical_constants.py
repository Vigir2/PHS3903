import numpy as np
kb = 1.380649e-23 #[J/K]

u = 1.660539e-27 #[kg]

Na = 6.02214076e23 #[1/mol]

e = 1.602176634e-19 #[C]

epsilon0_SI = 8.8541878128e-12 #[F/m]

epsilon0 = epsilon0_SI * u /(1e6 * e**2) #[e^2 * ps^2 / (u * ang^3)]

k = 1/(4 * np.pi * epsilon0) #[u * ang^3 / (ps^2 * e^2)]
print(k)