import numpy as np

name = "Server"
N = 150
dim = 3


a = 35 #[Ang]
rc = 15

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

T = 150 #[K]
P = 1   #[bar]

tau_p = 500
tau_t = 250