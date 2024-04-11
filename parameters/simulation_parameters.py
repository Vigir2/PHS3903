import numpy as np
import parameters.physical_constants as pc
import parameters.h2O_model as h2O

name = "test_2D"
N = 20
dim = 3


a = 30 #[Ang]
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

security_distance = 4.0 #[Ang]


# Thermodynamical properties
##########################################

T = 200 #[K]
P = 10   #[bar]

tau_p = 5000
tau_t = 250


# Ewald parameters
########################################

delta1 = 0.01
delta2 = 0.01
