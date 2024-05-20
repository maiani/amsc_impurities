import kwant
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy.interpolate import RegularGridInterpolator
from tqdm.notebook import tqdm

from pauli import *
from plot_tools import add_tl_label, complex_plot


def bulk_amsc_system(Nx, Ny, t, t_so, t_am, V, Delta, theta, hx, hy, hz, periodic_bc):
    """
    Create a Kwant system for a bulk altermagnetic superconductor.

    Args:
        Nx (int): Number of sites in the x-direction.
        Ny (int): Number of sites in the y-direction.
        t (float): Hopping amplitude.
        t_so (float): Spin-orbit hopping amplitude.
        t_am (float): Altermagnetic hopping amplitude.
        V (float): Potential landscape.
        Delta (callable): Function describing the superconducting gap as a function of position (x, y).
        theta (callable): Function describing the superconducting phase as a function of position (x, y).
        hx (callable): Function describing the x-component of the magnetic field as a function of position (x, y).
        hy (callable): Function describing the y-component of the magnetic field as a function of position (x, y).
        hz (callable): Function describing the z-component of the magnetic field as a function of position (x, y).
        periodic_bc (bool): Wether using periodic boundary conditions.

    Returns:
        kwant.system.System: A Kwant system for the given parameters.
    """

    # Create a Kwant lattice for a square system with 4 orbitals.
    lat = kwant.lattice.square(a=1, norbs=4)
    syst = kwant.Builder()

    # Domain definition
    def square(pos):
        (x, y) = pos
        return (abs(x) <= Nx // 2) * (abs(y) <= Ny // 2)

    # Onsite Hamiltonian element
    def onsite(site1):
        (x, y) = site1.pos
        return (
            (4 * t + V(x, y)) * tzs0
            - hx(x, y) * t0sx
            - hy(x, y) * t0sy
            - hz(x, y) * t0sz
            + txs0 * Delta(x, y) * np.cos(theta(x, y))
            - tys0 * Delta(x, y) * np.sin(theta(x, y))
        )

    # Hopping Hamiltonian elements
    def hopx(site1, site2):
        return -t * tzs0 + t_am * t0sz - 1j * t_so * tzsy

    def hopy(site1, site2):
        return -t * tzs0 - t_am * t0sz + 1j * t_so * tzsx

    # Define the on-site, x-hopping, and y-hopping terms for the system.
    syst[lat.shape(square, (0, 0))] = onsite
    syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = hopx
    syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = hopy

    # Apply periodic BC
    if periodic_bc:
        for i in range(-(Ny // 2), Ny // 2 + 1):
            syst[lat(Nx // 2, i), lat(-(Nx // 2), i)] = hopx

        for i in range(-(Nx // 2), Nx // 2 + 1):
            syst[lat(i, Ny // 2), lat(i, -(Ny // 2))] = hopy

    return syst, lat


def impurity_system(
    Nx,
    Ny,
    t,
    t_so,
    t_am,
    V,
    Delta,
    theta,
    hx,
    hy,
    hz,
    periodic_bc,
    t_prime,
    t_am_prime,
    t_so_prime,
    impurity_positions,
):
    """
    Create a Kwant system for an altermagnetic superconductor with an impurity in the center.

    Args:
        Nx (int): Number of sites in the x-direction.
        Ny (int): Number of sites in the y-direction.
        t (float): Hopping amplitude.
        t_so (float): Spin-orbit hopping amplitude.
        t_am (float): Altermagnetic hopping amplitude.
        V (float): Potential landscape.
        Delta (callable): Function describing the superconducting gap as a function of position (x, y).
        theta (callable): Function describing the superconducting phase as a function of position (x, y).
        hx (callable): Function describing the x-component of the magnetic field as a function of position (x, y).
        hy (callable): Function describing the y-component of the magnetic field as a function of position (x, y).
        hz (callable): Function describing the z-component of the magnetic field as a function of position (x, y).
        periodic_bc (bool): Wether using periodic boundary conditions.

        t_prime (float): Hopping amplitude for the impurity.
        t_so_prime  (float): Spin-orbit hopping amplitude for the impurity.
        t_am_prime  (float): Altermagnetic hopping amplitude for the impurity.
        impurity_positions (list): list of the position for impurities
        
    Returns:
        kwant.system.System: A Kwant system for the given parameters.
    """

    syst, lat = bulk_amsc_system(
        Nx=Nx,
        Ny=Ny,
        t=t,
        t_so=t_so,
        t_am=t_am,
        V=V,
        Delta=Delta,
        theta=theta,
        hx=hx,
        hy=hy,
        hz=hz,
        periodic_bc=periodic_bc,
    )

    for pos in impurity_positions:
        x, y = pos

        # Changing the hopping around the impurity
        syst[lat(x, y), lat(x-1, y)] = (
            -t_prime * tzs0 + t_am_prime * t0sz - 1j * t_so_prime * tzsy
        )
        syst[lat(x-1, y), lat(x, y)] = (
            -t_prime * tzs0 + t_am_prime * t0sz - 1j * t_so_prime * tzsy
        ).conj()
    
        syst[lat(x+1, y), lat(x, y)] = (
            -t_prime * tzs0 + t_am_prime * t0sz - 1j * t_so_prime * tzsy
        )
        syst[lat(x, y), lat(x+1, y)] = (
            -t_prime * tzs0 + t_am_prime * t0sz - 1j * t_so_prime * tzsy
        ).conj()
    
        syst[lat(x, y+1), lat(x, y)] = (
            -t_prime * tzs0 - t_am_prime * t0sz + 1j * t_so_prime * tzsx
        )
        syst[lat(x, y), lat(x, y+1)] = (
            -t_prime * tzs0 - t_am_prime * t0sz + 1j * t_so_prime * tzsx
        ).conj()
        
        syst[lat(x, y), lat(x, y-1)] = (
            -t_prime * tzs0 - t_am_prime * t0sz + 1j * t_so_prime * tzsx
        )
        syst[lat(x-1, y), lat(x, y)] = (
            -t_prime * tzs0 - t_am_prime * t0sz + 1j * t_so_prime * tzsx
        ).conj()

    return syst, lat
