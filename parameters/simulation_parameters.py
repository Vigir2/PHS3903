import numpy as np

name = "Test_pression"
N = 100
dim = 3


a = 24 #[Ang]
rc = 10

# Basis vectors
b1 = np.array([1, 0, 0])
b2 = np.array([0, 1, 0])
b3 = np.array([0, 0, 1])

A = a*b1
B = a*b2
C = a*b3

# Initial conditions
##########################################

security_distance = 4.8 #[Ang]


# Thermodynamical properties
##########################################

T = 200 #[K]
P = 1   #[bar]