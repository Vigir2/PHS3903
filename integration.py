from H2O import H2O
import Universe
import math
import parameters.simulation_parameters as simP
import forces as f
import parameters.h2O_model as h2O
import parameters.physical_constants as pc
import init_functions as init
import numpy as np
from functools import partial
from multiprocessing import Pool

q = [None, h2O.q, h2O.q, -2*h2O.q]
name = [None, "H1_pos", "H2_pos", "M_pos"]
force = [None, "H1_force", "H2_force", "M_force"]

def minimum_image(r, a: float) -> np.ndarray:
    """Retourne la distance associée à l'image la plus proche entre deux atomes d'après les conditions frontières périodiques"""
    return r - a * np.rint(r/a)

def tkinetic_energy(V: np.ndarray) -> float:
    """Retourne l'énergie cinétique totale translationnelle pour un ensemble de vitesses"""
    K = 0
    for v in V:
        K += 1/2 * h2O.M * np.dot(v, v)
    return K

def rkinetic_energy(J: np.ndarray, omega: np.ndarray) -> float:
    """Retourne l'énergie cinétique rotationnelle pour un ensemble de vitesses rotationnnelles"""
    K = 0
    for i in range(J.shape[0]):
        K += 1/2 * np.dot(omega[i], J[i])
    return K

def compute_forces(U: Universe, rc: float, a: float) -> None:
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
            elif (i%4 == 0) ^ (j%4 == 0):
                # Non-interracting atoms
                pass
            else:
                # Electrostatic forces for H and M
                R = minimum_image(U[math.floor(j/4)][j%4] - U[math.floor(i/4)][i%4], a=a)
                r = np.linalg.norm(R)
                if r < rc:
                    energy = f.coulomb(r=r, rc=rc, q1=q[i%4], q2=q[j%4], shifted=True)
                    virial += energy
                    psi += energy
                    fij = f.coulomb_force(r=r, q1=q[i%4], q2=q[j%4]) * R/r # Force on j due to i
                    setattr(U[math.floor(j/4)], force[j%4], getattr(U[math.floor(j/4)], force[j%4]) + fij) 
                    setattr(U[math.floor(i/4)], force[i%4], getattr(U[math.floor(i/4)], force[i%4]) - fij)
    setattr(U, "virial", virial)
    setattr(U, "potential_energy", psi)

def compute_forces_ewald(U: Universe, rc: float, a: float, alpha: float, c_max: int, rbasis: tuple, parallel: bool = False) -> None:
    """
    Calcule les forces du systèmes en utilisant la méthode de sommation d'Ewald pour le calcul des interractions électrostatiques
    
    Input
    -----
    - U (Universe): Univers de simulation
    - rc (float): Rayon de coupure
    - a (float): Largeur de la cellule de simulation
    - alpha (float): Paramètre de convergence de la sommation d'Ewald
    - c_max (int): Coefficient maximale de la sommation dans l'espace réciproque
    - rbasis (tuple): Base de l'espace réciproque
    """
    psi = 0
    virial = 0
    psi -= alpha * U.ewald_correction
    virial -= alpha * U.ewald_correction
    for m in U:
        m._H2O__reset_forces()

    # Real space summation
    for i in range(4*U.N-1):
        for j in range(i + 1, 4 * U.N):
            if math.floor(i/4) == math.floor(j/4):
                # Same molecule
                if (i%4 > 0) and (j%4 > 0):
                    # Electrostatic correction
                    R = U[math.floor(j/4)][j%4] - U[math.floor(i/4)][i%4]
                    r = np.linalg.norm(R)
                    E = f.ewald_electrostatic_correction(r, alpha, q[i%4], q[j%4])
                    psi -= E
                    virial -= E
                    fij = f.ewald_electrostatic_force_correction(r, alpha, q[i%4], q[j%4]) * R
                    fj = getattr(U[math.floor(j/4)], force[j%4]) + fij
                    fi = getattr(U[math.floor(i/4)], force[i%4]) - fij
                    setattr(U[math.floor(j/4)], force[j%4], fj)
                    setattr(U[math.floor(i/4)], force[i%4], fi)
            elif (i%4 == 0) and (j%4 == 0):
                # LJ interraction between O
                R = minimum_image(U[math.floor(j/4)][0] - U[math.floor(i/4)][0], a=a)
                r = np.linalg.norm(R)
                if r < rc:
                    psi += f.lennard_jones(r, rc, shifted=True)
                    fij = f.lennard_jones_force(r=r) * R/r
                    U[math.floor(j/4)].O_force += fij
                    U[math.floor(i/4)].O_force -= fij
                    virial += np.dot(fij, R)
            elif (i%4 > 0) and (j%4 > 0):
                # Electrostatic interractions
                R = minimum_image(U[math.floor(j/4)][j%4] - U[math.floor(i/4)][i%4], a=a)
                r = np.linalg.norm(R)
                if r < rc:
                    E = f.ewald_electrostatic(r, alpha, q[i%4], q[j%4])
                    psi += E
                    virial += E
                    fij = f.ewald_electrostatic_force(r, alpha, q[i%4], q[j%4]) * R
                    fj = getattr(U[math.floor(j/4)], force[j%4]) + fij
                    fi = getattr(U[math.floor(i/4)], force[i%4]) - fij
                    setattr(U[math.floor(j/4)], force[j%4], fj)
                    setattr(U[math.floor(i/4)], force[i%4], fi)

    # Reciprocal space summation
    if not parallel:
        for lamb in range(c_max + 1):
            if lamb == 0:
                for nu in range(c_max + 1):
                    if nu == 0:
                        for mu in range(1, c_max + 1):
                            k = mu * rbasis[0] + nu * rbasis[1] + lamb * rbasis[2]
                            pre_factor = np.exp(-np.dot(k,k)/(4*alpha**2)) / np.dot(k,k)
                            S = 0
                            for i in range(U.N):
                                S += (q[1] * np.exp(-1.j * np.dot(k, U[i][1])) + q[2] * np.exp(-1.j * np.dot(k, U[i][2])) + q[3] * np.exp(-1.j * np.dot(k, U[i][3])))
                            S_squared = abs(S)**2
                            E = 1/(a**3 * pc.epsilon0) * pre_factor * S_squared
                            psi += E
                            virial += E
                            for m in U:
                                m.H1_force += 2*q[1]/(pc.epsilon0 * a**3) * pre_factor * (np.exp(1.j * np.dot(k,m[1])) * S).imag * k
                                m.H2_force += 2*q[2]/(pc.epsilon0 * a**3) * pre_factor * (np.exp(1.j * np.dot(k,m[2])) * S).imag * k
                                m.M_force += 2*q[3]/(pc.epsilon0 * a**3) * pre_factor * (np.exp(1.j * np.dot(k,m[3])) * S).imag * k
                    else:
                        for mu in range(-c_max, c_max + 1):
                            k = mu * rbasis[0] + nu * rbasis[1] + lamb * rbasis[2]
                            pre_factor = np.exp(-np.dot(k,k)/(4*alpha**2)) / np.dot(k,k)
                            S = 0
                            for i in range(U.N):
                                S += (q[1] * np.exp(-1.j * np.dot(k, U[i][1])) + q[2] * np.exp(-1.j * np.dot(k, U[i][2])) + q[3] * np.exp(-1.j * np.dot(k, U[i][3])))
                            S_squared = abs(S)**2
                            E = 1/(a**3 * pc.epsilon0) * pre_factor * S_squared
                            psi += E
                            virial += E
                            for m in U:
                                m.H1_force += 2*q[1]/(pc.epsilon0 * a**3) * pre_factor * (np.exp(1.j * np.dot(k,m[1])) * S).imag * k
                                m.H2_force += 2*q[2]/(pc.epsilon0 * a**3) * pre_factor * (np.exp(1.j * np.dot(k,m[2])) * S).imag * k
                                m.M_force += 2*q[3]/(pc.epsilon0 * a**3) * pre_factor * (np.exp(1.j * np.dot(k,m[3])) * S).imag * k
            else:
                for mu in range(-c_max, c_max + 1):
                    for nu in range(-c_max, c_max + 1):
                        k = mu * rbasis[0] + nu * rbasis[1] + lamb * rbasis[2]
                        pre_factor = np.exp(-np.dot(k,k)/(4*alpha**2)) / np.dot(k,k)
                        S = 0
                        for i in range(U.N):
                            S += (q[1] * np.exp(-1.j * np.dot(k, U[i][1])) + q[2] * np.exp(-1.j * np.dot(k, U[i][2])) + q[3] * np.exp(-1.j * np.dot(k, U[i][3])))
                        S_squared = abs(S)**2
                        E = 1/(a**3 * pc.epsilon0) * pre_factor * S_squared
                        psi += E
                        virial += E
                        for m in U:
                            m.H1_force += 2*q[1]/(pc.epsilon0 * a**3) * pre_factor * (np.exp(1.j * np.dot(k,m[1])) * S).imag * k
                            m.H2_force += 2*q[2]/(pc.epsilon0 * a**3) * pre_factor * (np.exp(1.j * np.dot(k,m[2])) * S).imag * k
                            m.M_force += 2*q[3]/(pc.epsilon0 * a**3) * pre_factor * (np.exp(1.j * np.dot(k,m[3])) * S).imag * k
    else:
        k_config = []
        for lamb in range(c_max + 1):
            if lamb == 0:
                for nu in range(c_max + 1):
                    if nu == 0:
                        for mu in range(1, c_max + 1):
                            k_config.append([mu, nu, lamb])
                    else:
                        for mu in range(-c_max, c_max + 1):
                            k_config.append([mu, nu, lamb])
            else:
                for mu in range(-c_max, c_max + 1):
                    for nu in range(-c_max, c_max + 1):
                        k_config.append([mu, nu, lamb])
        pos_config = []
        for m in U:
            pos_config.append([m.O_pos, m.H1_pos, m.H2_pos, m.M_pos])
        with Pool() as pool:
            Output = pool.map(partial(parallel_ewald, pos_config=pos_config, rbasis=rbasis, alpha=alpha, a=a), k_config)
        for E, forces in Output:
            psi += E
            virial += E
            for i in range(U.N):
                U[i].H1_force += forces[i][0]
                U[i].H2_force += forces[i][1]
                U[i].M_force += forces[i][2]
                    
    setattr(U, "virial", virial)
    setattr(U, "potential_energy", psi)

def parallel_ewald(c_values: list, pos_config: list, rbasis: tuple, alpha: float, a: float):
    k = c_values[0] * rbasis[0] + c_values[1] * rbasis[1] + c_values[2] * rbasis[2]
    pre_factor = np.exp(-np.dot(k,k)/(4*alpha**2)) / np.dot(k,k)
    S = 0
    for i in range(len(pos_config)):
        S += (q[1] * np.exp(-1.j * np.dot(k, pos_config[i][1])) + q[2] * np.exp(-1.j * np.dot(k, pos_config[i][2])) + q[3] * np.exp(-1.j * np.dot(k, pos_config[i][3])))
        S_squared = abs(S)**2
    E = 1/(a**3 * pc.epsilon0) * pre_factor * S_squared
    forces = []
    for i in range(len(pos_config)):
        force_H1 = 2*q[1]/(pc.epsilon0 * a**3) * pre_factor * (np.exp(1.j * np.dot(k, pos_config[i][1])) * S).imag * k
        force_H2 = 2*q[2]/(pc.epsilon0 * a**3) * pre_factor * (np.exp(1.j * np.dot(k, pos_config[i][2])) * S).imag * k
        force_M = 2*q[3]/(pc.epsilon0 * a**3) * pre_factor * (np.exp(1.j * np.dot(k, pos_config[i][3])) * S).imag * k
        forces.append([force_H1, force_H2, force_M])
    return E, forces

def nve_verlet_run(U: Universe, dt: float, Ewald: bool = False, parallel: bool = False) -> None:
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
    U.compute_forces(Ewald=Ewald, parallel=parallel)
    F_n_1 = np.zeros((U.N, U.dim))
    T_n_1 = np.zeros((U.N, U.dim))
    for i in range(U.N):
        F_n_1[i] = U[i].cm_force()
        T_n_1[i] = U[i].torque()
    V_n_1 = V_n_05 + F_n_1 * dt / (2*h2O.M) #O(Δt2)
    J_n_1 = J_n_05 + T_n_1 * dt/2 #O(Δt2)
    for i in range(U.N):
        U[i].cm_vel = V_n_1[i]
        U[i].rot_vel = np.linalg.solve(U[i].inertia_tensor(), J_n_1[i])

def npt_verlet_run(U: Universe, T0: float, P0: float, dt: float, Nf: int, Ewald: bool = False) -> None:
    """Effectue une ittération de verlet vitesse en ensemble NPT"""
    Q = Nf * pc.kb * simP.tau_t**2
    W = Nf * pc.kb * simP.tau_p**2
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
    Kt, Kr = tkinetic_energy(Vn), rkinetic_energy(J=Jn, omega=omegan)
    T = 2 * (Kt + Kr) / (Nf * pc.kb)
    P = U.pression(Kt)
    thermo_025 = thermo0 + dt/(4*Q) * (Nf * (T - T0) * pc.kb + W * baro0**2 - pc.kb * T0)
    baro_025 = baro0 + dt/4 * (3 * U.a**3 / W * (P - P0) - thermo_025 * baro0)

    # n+1/2
    Vt_05 = Vn - dt/2 * (baro_025 + thermo_025) * Vn
    Jt_05 = Jn - dt/2 * (thermo_025 + baro_025) * Jn
    omegat_05 = omegan - dt/2 * (thermo_025 + baro_025) * omegan
    Kt, Kr = tkinetic_energy(Vt_05), rkinetic_energy(J=Jt_05, omega=omegat_05)
    P_n_05 = U.pression(K=Kt)
    T_n_05 = 2 * (Kt + Kr) / (Nf * pc.kb)
    baro_05 = baro_025 + dt/4 * (3*U.a**3 / W * (P_n_05 - P0) - baro_025 * thermo_025)
    thermo_05 = thermo_025 + dt/(4*Q) * (Nf * (T_n_05 - T0) * pc.kb + W * baro_05**2 - pc.kb * T0)
    V_n_05 = Vt_05 + dt/(2*h2O.M) * Fn
    J_n_05 = Jt_05 + Tn * dt/2
    omega_n_05 = omegat_05 + omega_dot * dt/2
    
    # n + 1
    Rcm = U.cm_position()
    R_n_1 = np.zeros(Rn.shape)
    for i in range(U.N):
        R_n_1[i] = Rn[i] + dt * (V_n_05[i] + baro_05*(Rn[i] - Rcm))
        U[i]._H2O__update_positions(R_n_1 = R_n_1[i], omega_n_05 = omega_n_05[i], dt = dt)
    a_1 = U.a * np.exp(dt * baro_05)
    setattr(U, "a", a_1)
    U.compute_forces(Ewald=Ewald)
    Fn_1 = np.zeros((U.N, U.dim))
    Tn_1 = np.zeros((U.N, U.dim))
    for i in range(U.N):
        Fn_1[i] = U[i].cm_force()
        Tn_1[i] = U[i].torque()
    Vt_1 = V_n_05 + dt / (2 * h2O.M) * Fn_1
    Jt_1 = J_n_05 + dt/2 * Tn_1
    omegat_1 = np.zeros((U.N, U.dim))
    for i in range(U.N):
        omegat_1[i] = np.linalg.solve(U[i].inertia_tensor(), Jt_1[i])
    Kt, Kr = tkinetic_energy(Vt_1), rkinetic_energy(J=Jt_1, omega=omegat_1)
    P_n_1 = U.pression(K=Kt)
    T_n_1 = 2 * (Kt + Kr) / (Nf * pc.kb)
    thermo_075 = thermo_05 + dt/(4*Q) * (Nf * (T_n_1 - T0) * pc.kb + W * baro_05**2 - pc.kb * T0)
    baro_075 = baro_05 + dt/4 * (3*U.a**3 / W * (P_n_1 - P0) - baro_05 * thermo_075)
    V_n_1 = Vt_1 - dt/2 * (thermo_075 + baro_075) * Vt_1
    J_n_1 = Jt_1 - dt/2 * (thermo_075 + baro_075) * Jt_1
    omega_n_1 = omegat_1 - dt/2 * (thermo_075 + baro_075) * omegat_1
    for i in range(U.N):
        U[i].cm_vel = V_n_1[i]
        U[i].rot_vel = omega_n_1[i]

    Kt, Kr = tkinetic_energy(V_n_1), rkinetic_energy(J=J_n_1, omega=omega_n_1)
    P_n_1 = U.pression(K=Kt)
    T_n_1 = 2 * (Kt + Kr) / (Nf * pc.kb)
    baro_1 = baro_075 + dt/4 * (3*U.a**3 / W * (P_n_1 - P0) - baro_075 * thermo_075)
    thermo_1 = thermo_075 + dt/(4*Q) * (Nf * (T_n_1 - T0) * pc.kb + W * baro_1**2 - pc.kb * T0)
    U.baro = baro_1
    U.thermo = thermo_1
    U.pressure = P_n_1
    U.temp = T_n_1
    
def nvt_verlet_run(U: Universe, T0: float, dt: float, Nf: int, Ewald: bool = False) -> None:
    """Effectue une ittération de verlet vitesse en ensemble NVT"""
    Q = Nf * pc.kb * simP.tau_t**2
    Fn = np.zeros((U.N, U.dim))
    Tn = np.zeros((U.N, U.dim))
    Vn = np.zeros((U.N, U.dim))
    Jn = np.zeros((U.N, U.dim))
    Rn = np.zeros((U.N, U.dim))
    omegan = np.zeros((U.N, U.dim))
    omega_dot = np.zeros((U.N, U.dim))

    # n
    thermo0 = U.thermo
    for i in range(U.N):
        Fn[i] = U[i].cm_force()
        Tn[i] = U[i].torque()
        Vn[i] = U[i].cm_vel
        Jn[i] = U[i].ang_momentum()
        Rn[i] = U[i].cm_pos()
        omegan[i] = U[i].rot_vel
        omega_dot[i] = U[i].omega_dot()

    # n+1/4
    Kt, Kr = tkinetic_energy(Vn), rkinetic_energy(J=Jn, omega=omegan)
    thermo_025 = thermo0 + dt/(4*Q) * (2 * (Kt + Kr) - Nf * pc.kb * T0)

    # n+1/2
    Vt_05 = Vn - dt/2 * thermo_025 * Vn
    Jt_05 = Jn - dt/2 * thermo_025 * Jn
    omegat_05 = omegan - dt/2 * thermo_025 * omegan
    Kt, Kr = tkinetic_energy(Vt_05), rkinetic_energy(J=Jt_05, omega=omegat_05)
    thermo_05 = thermo_025 + dt/(4*Q) * (2 * (Kt + Kr) - Nf * pc.kb * T0)
    V_n_05 = Vt_05 + dt/(2*h2O.M) * Fn
    J_n_05 = Jt_05 + Tn * dt/2
    omega_n_05 = omegat_05 + omega_dot * dt/2
    
    # n + 1
    R_n_1 = Rn + dt * V_n_05
    for i in range(U.N):
        U[i]._H2O__update_positions(R_n_1 = R_n_1[i], omega_n_05 = omega_n_05[i], dt = dt)
    U.compute_forces(Ewald=Ewald)
    Fn_1 = np.zeros((U.N, U.dim))
    Tn_1 = np.zeros((U.N, U.dim))
    for i in range(U.N):
        Fn_1[i] = U[i].cm_force()
        Tn_1[i] = U[i].torque()
    Vt_1 = V_n_05 + dt / (2 * h2O.M) * Fn_1
    Jt_1 = J_n_05 + dt/2 * Tn_1
    omegat_1 = np.zeros((U.N, U.dim))
    for i in range(U.N):
        omegat_1[i] = np.linalg.solve(U[i].inertia_tensor(), Jt_1[i])
    Kt, Kr = tkinetic_energy(Vt_1), rkinetic_energy(J=Jt_1, omega=omegat_1)
    thermo_075 = thermo_05 + dt/(4*Q) * (2 * (Kt + Kr) - Nf * pc.kb * T0)
    V_n_1 = Vt_1 - dt/2 * thermo_075 * Vt_1
    J_n_1 = Jt_1 - dt/2 * thermo_075 * Jt_1
    omega_n_1 = omegat_1 - dt/2 * thermo_075 * omegat_1
    for i in range(U.N):
        U[i].cm_vel = V_n_1[i]
        U[i].rot_vel = omega_n_1[i]
    Kt, Kr = tkinetic_energy(V_n_1), rkinetic_energy(J=J_n_1, omega=omega_n_1)
    thermo_1 = thermo_075 + dt/(4*Q) * (2 * (Kt + Kr) - Nf * pc.kb * T0)
    U.temp = 2 * (Kt + Kr) / (Nf * pc.kb)
    U.thermo = thermo_1

if __name__ == "__main__":
    pass
    