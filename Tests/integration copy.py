from H2O import H2O
from Universe import Universe
import math
import parameters.simulation_parameters as simP
import forces as f
import parameters.h2O_model as h2O
import parameters.physical_constants as pc
import init_functions as init

import numpy as np
def minimum_image(r, a: float):
    """Retourne la distance associée à l'image la plus proche entre deux atomes d'après les conditions frontières périodiques"""
    return r - a * np.rint(r/a)

def tkinetic_energy(V: np.ndarray):
    K = 0
    for v in V:
        K += 1/2 * h2O.M * np.dot(v, v)
    return K

q = [None, h2O.q, h2O.q, -2*h2O.q]
name = [None, "H1_pos", "H2_pos", "M_pos"]
force = [None, "H1_force", "H2_force", "M_force"]

def compute_forces(U: Universe, rc: float, a: float):
    """
    Calcule les forces du systèmes U, l'énergie potentielle et le Viriel

    Input
    -----
    - U (Universe): Univers de simulation
    - rc (float): Rayon de coupure utilisé pour le calcul des forces
    - a (float): Paramètre de maille de la cellule de simulation
    """
    for m in U:
        m._H2O__reset_forces()
    virial = 0
    psi = 0
    for i in range(4 * U.N - 4):
        for j in range(4*(math.floor(i/4) + 1), 4 * U.N):
            if (i%4 == 0) and (j%4 == 0):
                # L-J forces between O atoms
                R = minimum_image(U[math.floor(j/4)][0] - U[math.floor(i/4)][0], a=a)
                r = np.linalg.norm(R)
                if r < rc:
                    psi += f.lennard_jones(r=r, rc=rc, shifted=True)
                    fij = f.lennard_jones_force(r=r) * R/r # Force on j due to i
                    U[math.floor(j/4)].O_force += fij
                    U[math.floor(i/4)].O_force += -fij
                    virial += np.dot(fij, R)
                pass
            elif (i%4 == 0) ^ (j%4 == 0):
                # Non-interracting atoms
                pass
            else:
                # Electrostatic forces for H and M
                R = minimum_image(U[math.floor(j/4)][j%4] - U[math.floor(i/4)][i%4], a=a)
                r = np.linalg.norm(R)
                if r < rc:
                    psi += f.coulomb(r=r, rc=rc, q1=q[i%4], q2=q[j%4], shifted=True)
                    fij = f.coulomb_force(r=r, q1=q[i%4], q2=q[j%4]) * R/r # Force on j due to i
                    setattr(U[math.floor(j/4)], force[j%4], getattr(U[math.floor(j/4)], force[j%4]) + fij) 
                    setattr(U[math.floor(i/4)], force[i%4], getattr(U[math.floor(i/4)], force[i%4]) - fij)
                    virial += np.dot(fij, R)
    setattr(U, "virial", virial)
    setattr(U, "potential_energy", psi)
    return


def nve_verlet_run(U: Universe, dt: float):
    """Effectue une itération de verlet vitesse en ensemble NVE"""
    Fn = np.zeros((U.N, U.dim))
    Tn = np.zeros((U.N, U.dim))
    Vn = np.zeros((U.N, U.dim))
    Jn = np.zeros((U.N, U.dim))
    Rn = np.zeros((U.N, U.dim))
    omegan = np.zeros((U.N, U.dim))
    omega_dot = np.zeros((U.N, U.dim))

    # n
    for i in range(U.N):
        Fn[i] = U[i].cm_force()
        Tn[i] = U[i].torque()
        Vn[i] = U[i].cm_vel
        Jn[i] = U[i].ang_momentum()
        Rn[i] = U[i].cm_pos()
        omegan[i] = U[i].rot_vel
        omega_dot[i] = U[i].omega_dot()

    # n+1/2
    V_n_05 = Vn + Fn * dt/(2*h2O.M) #O(Δt3)
    J_n_05 = Jn + Tn * dt/2 #O(Δt3)
    R_n_1 = Rn + V_n_05 * dt #O(Δt4)
    omega_n_05 = omegan + omega_dot * dt/2 #O(Δt3)
    
    # n + 1
    for i in range(U.N):
        U.water_molecules[i]._H2O__update_positions(R_n_1 = R_n_1[i], omega_n_05 = omega_n_05[i], dt = dt)
    U.compute_forces()
    F_n_1 = np.zeros((U.N, U.dim))
    T_n_1 = np.zeros((U.N, U.dim))
    for i in range(U.N):
        F_n_1[i] = U[i].cm_force()
        T_n_1[i] = U[i].torque()
    V_n_1 = V_n_05 + F_n_1 * dt / (2*h2O.M) #O(Δt2)
    J_n_1 = J_n_05 + T_n_1 * dt/2 #O(Δt2)
    for i in range(U.N):
        U[i].cm_vel = V_n_1[i]
        U[i].rot_vel = np.linalg.inv(U[i].inertia_tensor())@J_n_1[i]

def npt_verlet_run(U: Universe, T0: float, P0: float, dt: int, Nf: int):
    Q = Nf * U.N * pc.kb * simP.tau_t**2
    W = Nf * U.N * pc.kb * simP.tau_p**2
    Fn = np.zeros((U.N, U.dim))
    Tn = np.zeros((U.N, U.dim))
    Vn = np.zeros((U.N, U.dim))
    Jn = np.zeros((U.N, U.dim))
    Rn = np.zeros((U.N, U.dim))
    omegan = np.zeros((U.N, U.dim))
    omega_dot = np.zeros((U.N, U.dim))

    # n
    thermo0 = U.thermo
    baro0 = U.baro
    for i in range(U.N):
        Fn[i] = U[i].cm_force()
        Tn[i] = U[i].torque()
        Vn[i] = U[i].cm_vel
        Jn[i] = U[i].ang_momentum()
        Rn[i] = U[i].cm_pos()
        omegan[i] = U[i].rot_vel
        omega_dot[i] = U[i].omega_dot()

    # n+1/4
    K = tkinetic_energy(Vn)
    T = 2 * K / (U.dim * U.N * pc.kb)
    P = U.pression(K)
    thermo_025 = thermo0 + dt/(4*Q) * (U.dim * U.N * (T - T0) * pc.kb + W * baro0**2 - pc.kb * T0)
    baro_025 = baro0 + dt/4 * (3 * U.a**3 / W * (P - P0) - thermo_025 * baro0)

    # n+1/2
    Vt_05 = Vn - dt/2 * (baro_025 + thermo_025) * Vn
    K = tkinetic_energy(V = Vt_05)
    P_n_05 = U.pression(K=K)
    T_n_05 = 2 /(U.dim * U.N * pc.kb)* K
    baro_05 = baro_025 + dt/4 * (3*U.a**3 / W * (P_n_05 - P0) - baro_025 * thermo_025)
    thermo_05 = thermo_025 + dt/(4*Q) * (U.dim * U.N * (T_n_05 - T0) * pc.kb + W * baro_05**2 - pc.kb * T0)
    V_n_05 = Vt_05 + dt/(2*h2O.M) * Fn
    J_n_05 = Jn + Tn * dt/2
    omega_n_05 = omegan + omega_dot * dt/2
    
    # n + 1
    R_n_1 = Rn + dt * (V_n_05 + baro_05*(Rn - U.cm_position()))
    for i in range(U.N):
        U.water_molecules[i]._H2O__update_positions(R_n_1 = R_n_1[i], omega_n_05 = omega_n_05[i], dt = dt)
    a_1 = U.a * np.exp(dt * baro_05)
    setattr(U, "a", a_1)
    U.compute_forces()
    Fn_1 = np.zeros((U.N, U.dim))
    Tn_1 = np.zeros((U.N, U.dim))
    for i in range(U.N):
        Fn_1[i] = U.water_molecules[i].cm_force()
        Tn_1[i] = U.water_molecules[i].torque()
    Vt_1 = V_n_05 + dt / (2 * h2O.M) * Fn_1
    K = tkinetic_energy(V = Vt_1)
    P_n_1 = U.pression(K=K)
    T_n_1 = 2 /(U.dim * U.N * pc.kb)* K
    thermo_075 = thermo_05 + dt/(4*Q) * (U.dim * U.N * (T_n_1 - T0) * pc.kb + W * baro_05**2 - pc.kb * T0)
    baro_075 = baro_05 + dt/4 * (3*U.a**3 / W * (P_n_1 - P0) - baro_05 * thermo_075)
    V_n_1 = Vt_1 - dt/2 * (thermo_075 + baro_075) * Vt_1
    J_n_1 = J_n_05 + Tn_1 * dt/2
    for i in range(U.N):
        U[i].cm_vel = V_n_1[i]
        U[i].rot_vel = np.linalg.inv(U[i].inertia_tensor())@J_n_1[i]
    K = tkinetic_energy(V = V_n_1)
    P_n_1 = U.pression(K=K)
    T_n_1 = 2 /(U.dim * U.N * pc.kb)* K
    baro_1 = baro_075 + dt/4 * (3*U.a**3 / W * (P_n_1 - P0) - baro_075 * thermo_075)
    thermo_1 = thermo_075 + dt/(4*Q) * (U.dim * U.N * (T_n_1 - T0) * pc.kb + W * baro_1**2 - pc.kb * T0)
    U.baro = baro_1
    U.thermo = thermo_1
    U.pressure = P_n_1
    U.tkinetic_energy = K

def andersen(water_molecules, T, a, dt):
    p = pc.Kt * a * dt / (pc.kb_SI * len(water_molecules))
    for m in water_molecules:
        if np.random.uniform(0, 1) < p:
            m.rot_vel = init.random_rot_velocity(m.inertia_tensor(), T, dim = 3)
    

if __name__ == "__main__":
    pass