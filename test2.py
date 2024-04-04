import numpy as np
from scipy.stats import special_ortho_group

# Generate a random rotation matrix
random_rotation_matrix = special_ortho_group.rvs(3)

print("Random Rotation Matrix:")
print(random_rotation_matrix)
