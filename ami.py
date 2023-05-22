"""
This modules provides functions useful to setup simulations of a Multiple Andreev Interferometer.
"""

from functools import lru_cache

import kwant
import numpy as np
import pauli
import scipy.linalg as la
import sympy as sym
from kwant.continuum import discretize, sympify
from sympy.physics.quantum.matrixutils import matrix_tensor_product as kron


@lru_cache
def sed(omega, Delta, h_Sc):
    """
    Denominator of superconducting leads self-energy.
    """
    return np.sqrt(Delta ** 2 - (h_Sc - omega) ** 2)

# @lru_cache
# def symbolic_hamiltonian_spinful_complete(dims : int, self_energy : bool):
#     """
#     Generate a symbolic Hamiltonian of the effective model for a 2DEG nanowire.
#         Parameters:
#             dims (int) : Number of dimension (1 or 2)
#     """
    
#     # Nanowire hamiltonian
#     hamiltonian = sympify(
#         """ 
#         k_c * (k_x**2 + k_y**2) * kron(sigma_z, sigma_0)
#         + V(x, y) * kron(sigma_z, sigma_0) 
#         + (+ alpha_z(x, y) * k_y) * kron(sigma_z, sigma_x)
#         + (- alpha_z(x, y) * k_x) * kron(sigma_z, sigma_y)
#         + (+ alpha_x(x, y) * k_y - alpha_y(x, y) * k_x) * kron(sigma_z, sigma_z)
#         + beta * ( k_x * kron(sigma_z, sigma_x) - k_y * kron(sigma_z, sigma_y))
#         + h_x(x, y) * kron(sigma_0, sigma_x)
#         + h_y(x, y) * kron(sigma_0, sigma_y)
#         + h_z(x, y) * kron(sigma_0, sigma_z)                       
#     """
#     )
    
#     if self_energy:
#         Sigma = sympify(
#             """
#             gamma_Sc(x, y) * ( 
#                 ( - omega - 1j * Gamma_Sc) * kron(sigma_0, sigma_0) 
#                  + h_Sc * kron(sigma_0, sigma_z)
#                  + Delta(x, y) * ( cos(theta(x,y)) * kron(sigma_x, sigma_0) 
#                                  - sin(theta(x,y)) * kron(sigma_y, sigma_0))
#             ) * ( 
#               (kron(sigma_0, sigma_0) + kron(sigma_0, sigma_z))/2 / sed(
#                omega + 1j * Gamma_Sc, 
#                Delta(x, y),
#                +h_Sc)
#             + (kron(sigma_0, sigma_0) - kron(sigma_0, sigma_z))/2 / sed(
#                omega + 1j * Gamma_Sc,
#                Delta(x, y),
#                -h_Sc)
#             )
#             + (- 1j * gamma_Sm(x, y) * kron(sigma_0, sigma_0))
#         """
#         )
        
#         # Rotate the self energy in the spin space    
#         U_p = sympify("exp( 1j / 2 * pi / 2 * sigma_y)")
#         U_h = sympify("sigma_y") * U_p.conjugate() * sympify("sigma_y")
#         U   = (kron(sympify("(sigma_0 + sigma_z)/2"), U_p) +
#                kron(sympify("(sigma_0 - sigma_z)/2"), U_h))

#         hamiltonian += U.H * Sigma * U

#     else:
#         hamiltonian += sympify("""Delta(x, y) * ( cos(theta(x,y)) * kron(sigma_x, sigma_0) 
#                                                 - sin(theta(x,y)) * kron(sigma_y, sigma_0))""" )
        
#     if dims == 1:
#         hamiltonian = hamiltonian.subs(sym.Symbol("k_y", commutative=False), 0)
#         hamiltonian = hamiltonian.subs(sym.Symbol("y", commutative=False), 0)
#     elif dims == 2:
#         pass
#     else:
#         raise Exception("Only dimensions 1 and 2 are implemented.")
    
#     return sym.simplify(hamiltonian)

@lru_cache
def symbolic_hamiltonian_spinful_simplified(dims : int, self_energy : bool):
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
        + (+ alpha_z * k_y) * kron(sigma_z, sigma_x)
        + (- alpha_z * k_x) * kron(sigma_z, sigma_y)
        + h_x * kron(sigma_0, sigma_x)
        + h_y * kron(sigma_0, sigma_y)
        + h_z * kron(sigma_0, sigma_z)                       
    """
    )
    
    if self_energy:
        Sigma = sympify(
            """
            gamma_Sc(x, y) * ( 
                ( - omega - 1j * Gamma_Sc) * kron(sigma_0, sigma_0)
                 + Delta(x, y) * ( cos(phi(x,y)) * kron(sigma_x, sigma_0) 
                                 - sin(phi(x,y)) * kron(sigma_y, sigma_0))
            ) / sed(
               omega + 1j * Gamma_Sc,
               Delta(x, y),
               0)
        """
        )
        
        # Rotate the self energy in the spin space    
        U_p = sympify("exp( 1j / 2 * pi / 2 * sigma_y)")
        U_h = sympify("sigma_y") * U_p.conjugate() * sympify("sigma_y")
        U   = (kron(sympify("(sigma_0 + sigma_z)/2"), U_p) +
               kron(sympify("(sigma_0 - sigma_z)/2"), U_h))

        hamiltonian += U.H * Sigma * U

    else:
        hamiltonian += sympify("""Delta(x, y) * ( cos(phi(x,y)) * kron(sigma_x, sigma_0) 
                                                - sin(phi(x,y)) * kron(sigma_y, sigma_0))""" )
        
    if dims == 1:
        hamiltonian = hamiltonian.subs(sym.Symbol("k_y", commutative=False), 0)
        hamiltonian = hamiltonian.subs(sym.Symbol("y", commutative=False), 0)
    elif dims == 2:
        pass
    else:
        raise Exception("Only dimensions 1 and 2 are implemented.")
    
    return sym.simplify(hamiltonian)



def make_1D_system(a_x, bounds, with_leads, finalized, spinless_smatrix, self_energy):
    """
    Create a Kwant system of a single channel 2DEG nanowire.
    """

    lat = kwant.lattice.chain(a_x, norbs=4)

    hamiltonian = symbolic_hamiltonian_spinful_simplified(dims = 1, self_energy=self_energy)

    template = discretize(hamiltonian, grid=lat)

    def shape(site):
        (x) = site.pos
        return (x>bounds[0])*(x<bounds[1])

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
                V="V_L",
                alpha_z="alpha_z_lead",
                h_x="h_x_lead",
                h_y="h_y_lead",
                h_z="h_z_lead",
            ),
            shape=(lambda site: True),
            start=[bounds[0]],
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
                V="V_R",
                alpha_z="alpha_z_lead",
                h_x="h_x_lead",
                h_y="h_y_lead",
                h_z="h_z_lead",
            ),
            shape=(lambda site: True),
            start=[+bounds[1]],
        )

        syst.attach_lead(right_lead)

    if finalized:
        syst = syst.finalized()

    return syst


def build_ami_fields(
    N, 
    L_S, L_N,
    mu_S, mu_N, 
    Delta_0, Delta_phi,
    gamma_0, l_smth,
    L_B, V_B, mu_ld
    ):
    """
    Build the fields for an Andreev Multi Interferometer.
    """
    if l_smth == 0 :
        step = np.heaviside
    else:
        def step(x):
            return 1 / (1 + np.exp(-x / l_smth))

    L_w = N * (L_S + L_N) + L_N 
    
    def V(x, y):

        Vp = (
            -mu_ld
            + (mu_ld-mu_N) * (step(x) - step(x - L_w)) 
            + (mu_ld + V_B) * (step(x + L_B) - step(x)) 
            + (mu_ld + V_B) * (step(x - L_w) - step(x - L_w - L_B))
             )
        Vp += (mu_N - mu_S) * ((x-L_N) % (L_N + L_S) < L_S) * (x < L_w) * (x > 0)
        
        return Vp

    def Delta(x, y):
        return Delta_0 * ((x-L_N) % (L_N + L_S) < L_S) * (x < L_w) * (x > 0)

    def phi(x, y):
        return Delta_phi * ((x-L_N) // (L_N + L_S) )

    def gamma(x, y):
        return gamma_0 * ((x-L_N) % (L_N + L_S) < L_S) * (x < L_w) * (x > 0)

    return V, gamma, Delta, phi
