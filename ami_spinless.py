"""
This modules provides functions useful to setup simulations of 2DEG systems.
"""
from functools import lru_cache

import numpy as np
import scipy.linalg as la
import sympy as sym
from sympy.physics.quantum.matrixutils import matrix_tensor_product as kron

import kwant
from kwant.continuum import discretize, sympify

import pauli


def symbolic_hamiltonian_spinless(dims)
        """
    Generate a symbolic Hamiltonian of the effective model for a 2DEG nanowire.
        Parameters:
            dims (int) : Number of dimension (1 or 2)
    """pass

def symbolic_hamiltonian_spinful(dims):
    """
    Generate a symbolic Hamiltonian of the effective model for a 2DEG nanowire.
        Parameters:
            dims (int) : Number of dimension (1 or 2)
    """
    
    # Nanowire hamiltonian
    hamiltonian = sympify(
        """ 
        k_c * (k_x**2 + k_y**2) * kron(sigma_z, sigma_0)
        + V(x, y) * kron(sigma_z, sigma_0) 
        + (+ alpha_z(x, y) * k_y) * kron(sigma_z, sigma_x)
        + (- alpha_z(x, y) * k_x) * kron(sigma_z, sigma_y)
        + (+ alpha_x(x, y) * k_y - alpha_y(x, y) * k_x) * kron(sigma_z, sigma_z)
        + (beta_x * k_x * kron(sigma_z, sigma_x) + beta_y * k_y * kron(sigma_z, sigma_y))
        + g_x * B_x * kron(sigma_0, sigma_x)
        + g_y * B_y * kron(sigma_0, sigma_y)
        + g_z * B_z * kron(sigma_0, sigma_z)                       
    """
    )

    Sigma = sympify(
        """
        gamma_Sc(x, y) * ( 
            ( - omega - 1j * Gamma_Sc) * kron(sigma_0, sigma_0) 
             + h_Sc * kron(sigma_0, sigma_z)
             + Delta_0 * Delta_mod(B_x, B_y, B_z, Bc_x, Bc_y, Bc_z) * kron(sigma_x, sigma_0) 
        ) * ( 
          (kron(sigma_0, sigma_0) + kron(sigma_0, sigma_z))/2 * d(
           omega + 1j * Gamma_Sc, 
           Delta_0 * Delta_mod(B_x, B_y, B_z, Bc_x, Bc_y, Bc_z),
           +h_Sc)
        + (kron(sigma_0, sigma_0) - kron(sigma_0, sigma_z))/2 * d(
           omega + 1j * Gamma_Sc,
           Delta_0 * Delta_mod(B_x, B_y, B_z, Bc_x, Bc_y, Bc_z),
           -h_Sc)
        )
        + (- 1j * gamma_Sm(x, y) * kron(sigma_0, sigma_0))
    """
    )

    # U = sym.simplify(sympify("kron(sigma_0, sigma_0)"))

    # Rotate the self energy in the spin space    
    U_p = sympify("exp( 1j / 2 * pi / 2 * sigma_y)")
    U_h = sympify("sigma_y") * U_p.conjugate() * sympify("sigma_y")
    U   = kron(sympify("(sigma_0 + sigma_z)/2"), U_p) + kron(sympify("(sigma_0 - sigma_z)/2"), U_h)
    
    hamiltonian += U.H * Sigma * U

    if dims == 1:
        hamiltonian = hamiltonian.subs(sym.Symbol("k_y", commutative=False), 0)
        hamiltonian = hamiltonian.subs(sym.Symbol("y", commutative=False), 0)
    elif dims == 2:
        pass
    else:
        raise Exception("Only dimensions 1 and 2 are implemented.")

    return sym.simplify(hamiltonian)


#@lru_cache
def make_1Dwire(a_x, L_x, with_leads=True, finalized=True, spinless_smatrix=False):
    """
    Create a Kwant system of a single channel 2DEG nanowire.
    """

    lat = kwant.lattice.chain(a_x, norbs=4)

    hamiltonian = wire_hamiltonian_sym(dims = 1)

    template = discretize(hamiltonian, grid=lat)

    def shape(site):
        (x) = site.pos
        return np.abs(x) <= L_x / 2

    syst = kwant.Builder()
    syst.fill(template, shape, (0.0,))

    if with_leads:

        phs = +pauli.tysy

        ### Build conservation law
        rot = la.expm(1j / 2 * np.pi / 2 * pauli.sy)
        trs = 1j * pauli.sy
        U = np.zeros((4, 4), dtype=complex)
        U[0:2, 0:2] = rot
        U[2:4, 2:4] = trs @ rot @ trs.T
        
        if spinless_smatrix:
            cl = U @ np.diag([0, 0, 1, 1]) @ np.conj(U.T)
        else:
            cl = U @ np.diag([0, 1, 2, 3]) @ np.conj(U.T)

        left_lead = kwant.Builder(
            symmetry=kwant.TranslationalSymmetry((-a_x,)),
            particle_hole=phs,
            conservation_law=cl,
        )

        left_lead.fill(
            template.substituted(
                gamma_Sc="gamma_Sc_lead", 
                gamma_Sm="gamma_Sm_lead", 
                V="V_L",
                alpha_x="alpha_x_lead",
                alpha_y="alpha_y_lead",
                alpha_z="alpha_z_lead",
            ),
            shape=(lambda site: True),
            start=[-L_x],
        )
        syst.attach_lead(left_lead)

        right_lead = kwant.Builder(
            symmetry=kwant.TranslationalSymmetry((+a_x,)),
            particle_hole=phs,
            conservation_law=cl,
        )

        right_lead.fill(
            template.substituted(
                gamma_Sc="gamma_Sc_lead", 
                gamma_Sm="gamma_Sm_lead", 
                V="V_R",
                alpha_x="alpha_x_lead",
                alpha_y="alpha_y_lead",
                alpha_z="alpha_z_lead",
            ),
            shape=(lambda site: True),
            start=[+L_x],
        )

        syst.attach_lead(right_lead)

    if finalized:
        syst = syst.finalized()

    return syst


#@lru_cache
def make_2Dwire(L_x, L_y, a_x, a_y, with_leads=True, finalized=True):
    """
    Create a Kwant system of a finite width 2DEG nanowire.
    """
    lat = kwant.lattice.Monatomic([[a_x, 0], [0, a_y]], offset=None, name='wire', norbs=4)
    hamiltonian = wire_hamiltonian_sym(dims = 2)
    template = discretize(hamiltonian, grid=lat)

    def shape(site):
        (x, y) = site.pos
        return (abs(x) <= L_x / 2) and (abs(y) <= L_y / 2) 

    syst = kwant.Builder()
    syst.fill(template, shape, (0.0, 0.0))

    if with_leads:
        phs = +pauli.tysy
        
        ### Build conservation law
        rot = la.expm(1j/2 * np.pi/2 * pauli.sy)
        trs = 1j * pauli.sy
        U = np.zeros((4, 4), dtype=complex)
        U[0:2, 0:2] = rot
        U[2:4, 2:4] = trs@rot@trs.T

        cl  = U@np.diag([0, 1, 2, 3])@np.conj(U.T)

        left_lead = kwant.Builder(
            symmetry=kwant.TranslationalSymmetry((-a_x, 0)),
            particle_hole=phs,
            conservation_law=cl,
        )

        left_lead.fill(
            template.substituted(
                gamma_Sc="gamma_Sc_lead", 
                gamma_Sm="gamma_Sm_lead", 
                V = "V_L",
                alpha_x = "alpha_x_lead",
                alpha_y = "alpha_y_lead", 
                alpha_z = "alpha_z_lead",
                beta_x = "beta_x_lead",
                beta_y = "beta_y_lead"                
            ),
            shape=(lambda site: (abs(site.pos[1]) <= L_y / 2)),
            start=[-L_x//2 - a_x, 0],
        )
        syst.attach_lead(left_lead)

        right_lead = kwant.Builder(
            symmetry=kwant.TranslationalSymmetry((+a_x, 0)),
            particle_hole=phs,
            conservation_law=cl,
        )

        right_lead.fill(
            template.substituted(
                gamma_Sc="gamma_Sc_lead", 
                gamma_Sm="gamma_Sm_lead", 
                V = "V_R",
                alpha_x = "alpha_x_lead",
                alpha_y = "alpha_y_lead", 
                alpha_z = "alpha_z_lead",
                beta_x = "beta_x_lead",
                beta_y = "beta_y_lead"
            ),
            shape=(lambda site: (abs(site.pos[1]) <= L_y / 2)),
            start=[+L_x//2 + a_x, 0],
        )

        syst.attach_lead(right_lead)

    if finalized:
        syst = syst.finalized()

    return syst


#@lru_cache
# def get_default_params():
#     """
#     Generate the default prameters dict.
#     """

#     ############### POTENTIAL LANDSCAPE #######################

#     def V(x, y):
#         return 0

#     def V_L(x, y):
#         return 0

#     def V_R(x, y):
#         return 0

#     ############## SPIN-ORBIT COUPLING #######################

#     def alpha_x(x, y):
#         return 0  # (2 * x / L_wire) ** 5

#     def alpha_y(x, y):
#         return 0  # (2 * y / W_wire) ** 3 # np.sin(-np.pi/6)

#     def alpha_z(x, y):
#         return -0.010

#     ############# SUPERCONDUCTOR #############################

#     def Delta_mod(B_x, B_y, B_z, Bc_x, Bc_y, Bc_z):
#         return 1  # np.sqrt(1 - (B_x/Bc_x)**2 - (B_y/Bc_y)**2 - (B_z/Bc_z)**2 )

# #     def d_up(omega, Delta, h_Sc):
# #         return (
# #             (1 / np.sqrt(Delta ** 2 - (+h_Sc - omega) ** 2))
# #             if (abs(+h_Sc - omega) <= abs(Delta))
# #             else (1j * np.sign(omega) / np.sqrt((+h_Sc - omega) ** 2 - Delta ** 2))
# #         )

# #     def d_down(omega, Delta, h_Sc):
# #         return (
# #             (1 / np.sqrt(Delta ** 2 - (-h_Sc - omega) ** 2))
# #             if (abs(-h_Sc - omega) <= abs(Delta))
# #             else (1j * np.sign(omega) / np.sqrt((-h_Sc - omega) ** 2 - Delta ** 2))
# #         )
# # 
# #     def d_up(omega, Delta, h_Sc):
# #         return 1 / np.sqrt(Delta ** 2 - (+h_Sc - omega) ** 2)

# #     def d_down(omega, Delta, h_Sc):
# #         return 1 / np.sqrt(Delta ** 2 - (-h_Sc - omega) ** 2)
    
#     def d(omega, Delta, h_Sc):
#         return 1 / np.sqrt(Delta ** 2 - (h_Sc - omega) ** 2)
    
#     def gamma_Sc(x, y):
#         return 0.20 * 1e-3

#     def gamma_Sm(x, y):
#         return 0 * 1e-6


#     params = {
#         # Name
#         "simname": "finite_bias",
#         # Material params
#         "k_c": 0.0380998212 / 0.026,
#         "alpha_x": alpha_x, # lambda *args: 0,  
#         "alpha_y": alpha_y, # lambda *args: 0, 
#         "alpha_z": alpha_z,
#         "beta_x" : 0,
#         "beta_y" : 0,
#         "g_x": 1,  # 250E-6/1.5,
#         "g_y": 1,  # 300E-6/0.3,
#         "g_z": 1,  # 300E-6/0.3,
#         # Electrostatics
#         "V": V,
#         "V_L": V_L,
#         "V_R": V_R,
#         # Induced terms params
#         "Delta_0": 0.25 * 1e-3,
#         "Delta_mod": Delta_mod,
#         "Bc_x": 3.5,
#         "Bc_y": 3.5,
#         "Bc_z": 0.3,
#         "d": d,
#         # "d_up": d_up,
#         # "d_down": d_down,
#         "h_Sc": 0 * 1e-12,
#         "Gamma_Sc": 1 * 1e-12,
#         "gamma_Sm": gamma_Sm, 
#         "gamma_Sc": gamma_Sc, 
#         "omega": 0,
#         # Magnetic field
#         "B_x": 0.0,
#         "B_y": 0.0,
#         "B_z": 0.0,
#         # Leads params
#         "gamma_Sc_lead": lambda *args: 0,
#         "gamma_Sm_lead": lambda *args: 0,
#         "alpha_x_lead": lambda *args: 0,
#         "alpha_y_lead": lambda *args: 0,
#         "alpha_z_lead": lambda *args: 0,
#     }

#     return params


#@lru_cache
def build_finitebias_pl(
    L_wire,
    L_barrier_L,
    L_barrier_R,
    mu_wire,
    DV_barrier_L,
    DV_barrier_R,
    DV_lead,
    V_bias_L,
    V_bias_R,
    gamma_Sc_0,
    gamma_Sm_0,
    l_smth,
):
    """
    Build the potential landscape of a finite-bias system.
    """
    
    def sig(x, l_smth=1e-3):
        return 0.5 * (1 + np.tanh(0.5 * x / l_smth))

    def V(x, y):

        Vp = (
            -mu_wire * (sig(x + L_wire / 2) - sig(x - L_wire / 2))
            + (-DV_lead + V_bias_L) * (1 - sig(x + L_wire / 2 + L_barrier_L))
            + (-DV_lead + V_bias_R) * (sig(x - L_wire / 2 - L_barrier_R))
            + (DV_barrier_L + V_bias_L * (-x - L_wire / 2) / L_barrier_L)
            * (sig(x + L_wire / 2 + L_barrier_L) - sig(x + L_wire / 2))
            + (DV_barrier_R + V_bias_R * (+x - L_wire / 2) / L_barrier_R)
            * (sig(x - L_wire / 2) - sig(x - L_wire / 2 - L_barrier_R))
        )

        return Vp

    def V_L(x, y):
        return V(-1e6, 0)

    def V_R(x, y):
        return V(+1e6, 0)

    def gamma_Sc(x, y):
        return gamma_Sc_0 * (sig(x + L_wire / 2) - sig(x - L_wire / 2))

    def gamma_Sm(x, y):
        return gamma_Sm_0 * (sig(x + L_wire / 2) - sig(x - L_wire / 2))

    return V, V_L, V_R, gamma_Sc, gamma_Sm


#@lru_cache
def build_superlattice_pl(
    L_wire, L_mod, L_barrier, mu_wire, Dmu_wire, DV_lead, DV_barrier, gamma_0, l_smth,
):
    """
    Build the potential landscape of a superlattice system.
    """
    def sig(x):
        return 1 / (1 + np.exp(-x / l_smth))

    def V(x, y):

        Vp = (
            -mu_wire * (sig(x + L_wire / 2) - sig(x - L_wire / 2))
            + (-DV_lead) * (1 - sig(x + L_wire / 2 + L_barrier))
            + (-DV_lead) * (sig(x - L_wire / 2 - L_barrier))
            + (DV_barrier) * (sig(x + L_wire / 2 + L_barrier) - sig(x + L_wire / 2))
            + (DV_barrier) * (sig(x - L_wire / 2) - sig(x - L_wire / 2 - L_barrier))
        )

        for i in range(0, L_wire // L_mod):
            Vp += +(Dmu_wire) * (
                +sig(x + L_wire / 2 - i * L_mod - L_mod / 2)
                - sig(x + L_wire / 2 - (i + 1) * L_mod)
            )

        return Vp

    def V_L(x, y):
        return V(-1e6, 0)

    def V_R(x, y):
        return V(+1e6, 0)

    def gamma(x, y):

        return gamma_0 * (sig(x + L_wire / 2) - sig(x - L_wire / 2))

    return V, V_L, V_R, gamma
        