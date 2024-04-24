import numpy as np
import parameters.physical_constants as pc

name = "test_pression"
N = 125
dim = 3


a = 28 #[Ang]
rc = 12

# Basis vectors
b1 = np.array([1, 0, 0])
b2 = np.array([0, 1, 0])
b3 = np.array([0, 0, 1])

A = a*b1
B = a*b2
C = a*b3

# Initial conditions
##########################################

security_distance = 2.5 #[Ang]


# Thermodynamical properties
##########################################

T = 300 #[K]
P = 10   #[bar]

tau_p = 10000
tau_t = 500


# Ewald parameters
########################################

delta1 = 0.001
delta2 = 0.01

