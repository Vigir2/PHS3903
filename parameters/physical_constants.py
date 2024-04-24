import numpy as np
from pint import UnitRegistry
ureg = UnitRegistry()

# SI constants #

kb_SI = 1.380649e-23 #[J/K]

u = 1.660539e-27 #[kg]

epsilon0_SI = 8.8541878128e-12 #[F/m]

Na = 6.02214076e23 #[1/mol]

e = 1.602176634e-19 #[C]


# Unit converters #

uÅfs_to_Pa = ureg("amu/angstrom/fs^2").to("Pa").magnitude

bar_to_uÅfs = ureg("bar").to("amu/angstrom/fs^2").magnitude

uÅfs_to_bar = ureg("amu/angstrom/fs^2").to("bar").magnitude

uÅfs_to_eV = ureg("amu*angstrom^2/fs^2").to("eV").magnitude

uÅfs_to_KJ_mol = ureg("amu*angstrom^2/fs^2/molecule").to("kJ/mol").magnitude

KJ_mol_to_uÅfs = ureg("kJ/mol").to("amu*angstrom^2/fs^2/molecule").magnitude


# Physical constants in uÅfs unit system #

epsilon0 = epsilon0_SI * ureg("F/m").to("e^2 * fs^2 / u / angstrom^3").magnitude #[e^2 * fs^2 / (u * Å^3)]

k = 1/(4 * np.pi * epsilon0) #[u * Å^3 / (fs^2 * e^2)]

kb = kb_SI * ureg("J/K").to("amu*angstrom^2/fs^2/K").magnitude #[u * Å^2 / fs^2 / K]

