import numpy as np
from pint import UnitRegistry
ureg = UnitRegistry()

kb_SI = 1.380649e-23 #[J/K]

u = 1.660539e-27 #[kg]

epsilon0_SI = 8.8541878128e-12 #[F/m]

Na = 6.02214076e23 #[1/mol]

e = 1.602176634e-19 #[C]

joules_to_uÅfs = 1e-10 / u # Converts J to u * Å^2 / fs^2

newtons_to_uÅfs = 1e-20 / u # Converts N to u * Å / fs^2

uÅfs_to_Pa = u * 1e40

bar_to_uÅfs = joules_to_uÅfs / 1e25

uÅfs_to_bar = 1 / bar_to_uÅfs

epsilon0 = epsilon0_SI * u / e**2 #[e^2 * fs^2 / (u * Å^3)]

k = 1/(4 * np.pi * epsilon0) #[u * Å^3 / (fs^2 * e^2)]

Kt = 0.6 /1e22 #[J / (ps * ang * K)]

kb = kb_SI * ureg("J/K").to("amu*angstrom^2/fs^2/K").magnitude #[u * Å^2 / fs^2 / K]

uÅfs_to_eV = 1/(joules_to_uÅfs * e)

print(bar_to_uÅfs)
print(ureg("bar").to("amu*/angstrom/fs^2").magnitude)