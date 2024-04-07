import numpy as np

kb_SI = 1.380649e-23 #[J/K]

u = 1.660539e-27 #[kg]

epsilon0_SI = 8.8541878128e-12 #[F/m]

Na = 6.02214076e23 #[1/mol]

e = 1.602176634e-19 #[C]

joules_to_uÅfs = 1e-10 / u # Converts J to u * Å^2 / fs^2

newtons_to_uÅfs = 1e-20 / u # Converts N to u * Å / fs^2

uÅfs_to_Pa = u * 1e40

bar_to_uÅfs = 1 / (u * 1e37)

epsilon0 = epsilon0_SI /(e**2 * newtons_to_uÅfs * 1e20) #[e^2 * fs^2 / (u * ang^3)]

k = 1/(4 * np.pi * epsilon0) #[u * ang^3 / (fs^2 * e^2)]

Kt = 0.6 /1e22 #[J / (ps * ang * K)]

kb = kb_SI * joules_to_uÅfs #[u * Å^2 / fs^2 / K]

print(bar_to_uÅfs)