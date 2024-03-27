import numpy as np

name = "Test_multi"
N = 1000
dim = 3

a = 40 #[Ang]

b1 = np.array([1, 0, 0])
b2 = np.array([0, 1, 0])
b3 = np.array([0, 0, 1])

A = a*b1
B = a*b2
C = a*b3

# Initial conditions
##########################################

security_distance = 3 #[Ang]


# Thermodynamical properties
##########################################

T = 300 #[K]
P = 1   #[bar]