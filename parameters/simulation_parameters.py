import numpy as np

# Paramètres de simulation #
##########################################
name = "test_nvt"
N = 100 # Nombre de molécules d'eau
dim = 3 # Dimensionalité de la simulation
a = 18 #[Å] Taille de la cellule de simulation
rc = 15 # [Å] Rayon de coupure pour la sommation dans l'espace réel


# Conditions initiales
##########################################
security_distance = 2.8 #[Å]  Distance minimale de sécurité entre les moléccules d'eau
T = 300 #[K] Température initiale


# Paramètres de cconvergence
##########################################
tau_p = 2500 # [fs] Temps de relaxation du barostat
tau_t = 1000 # [fs] Temps de relaxation du thermostat
delta1 = 0.1 # [kJ/mol] Précision de  la sommation d'Ewald pour le calcul des forces  électrostatiques dans l'espace réel
delta2 = 0.1 # [kJ/mol] Précision de  la sommation d'Ewald pour le calcul des forces  électrostatiques dans l'espace réciproque


# Système de coordonné (Ne pas toucher)
##########################################
b1 = np.array([1, 0, 0])
b2 = np.array([0, 1, 0])
b3 = np.array([0, 0, 1])
A = a*b1
B = a*b2
C = a*b3