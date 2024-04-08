import numpy as np

name = "test_npt"
N = 125
dim = 3


a = 35 #[Ang]
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

security_distance = 4.5 #[Ang]


# Thermodynamical properties
##########################################

T = 100 #[K]
P = 10   #[bar]

tau_p = 700
tau_t = 700