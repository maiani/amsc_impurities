import kwant
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from pauli import t0sx, t0sy, t0sz, txs0, tys0, tzs0, tzsx, tzsy


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
        syst[lat(x, y), lat(x - 1, y)] = (
            -t_prime * tzs0 + t_am_prime * t0sz - 1j * t_so_prime * tzsy
        )
        syst[lat(x - 1, y), lat(x, y)] = (
            -t_prime * tzs0 + t_am_prime * t0sz - 1j * t_so_prime * tzsy
        ).conj()

        syst[lat(x + 1, y), lat(x, y)] = (
            -t_prime * tzs0 + t_am_prime * t0sz - 1j * t_so_prime * tzsy
        )
        syst[lat(x, y), lat(x + 1, y)] = (
            -t_prime * tzs0 + t_am_prime * t0sz - 1j * t_so_prime * tzsy
        ).conj()

        syst[lat(x, y + 1), lat(x, y)] = (
            -t_prime * tzs0 - t_am_prime * t0sz + 1j * t_so_prime * tzsx
        )
        syst[lat(x, y), lat(x, y + 1)] = (
            -t_prime * tzs0 - t_am_prime * t0sz + 1j * t_so_prime * tzsx
        ).conj()

        syst[lat(x, y), lat(x, y - 1)] = (
            -t_prime * tzs0 - t_am_prime * t0sz + 1j * t_so_prime * tzsx
        )
        syst[lat(x - 1, y), lat(x, y)] = (
            -t_prime * tzs0 - t_am_prime * t0sz + 1j * t_so_prime * tzsx
        ).conj()

    return syst, lat


def generate_intial_Delta(
    x, y, Delta_init, vortex_positions, windings, l_core, noise=0
):
    """
    Generate the initial order parameter Delta for a simulation using the AM-SC model.

    Parameters:
        x (array-like): The x-coordinates of the grid.
        y (array-like): The y-coordinates of the grid.
        Delta_init (array-like): The initial order parameter Delta.
        vortex_positions (list of tuples): The positions of the vortices.
        windings (list of floats): The windings of the vortices.
        l_core (float): The core length of the vortices.
        noise (float, optional): The amount of noise to add to the initial order parameter. Defaults to 0.

    Returns:
        tuple: A tuple containing two functions:
            - Delta (function): The order parameter Delta as a function of x and y.
            - theta (function): The phase angle theta as a function of x and y.
    """

    x_ax = x[0]
    y_ax = y[:, 0]

    Psi_n = Delta_init + 0j * x

    if l_core != 0:
        for n, pos in enumerate(vortex_positions):
            xp, yp = pos
            Psi_n *= (
                1 - np.exp(-np.sqrt((x - xp) ** 2 + (y - yp) ** 2) / l_core)
            ) * np.exp(1j * windings[n] * np.arctan2(y - yp, x - xp))

    if noise != 0:
        Psi_n += np.random.normal(
            0, noise / np.sqrt(2), Psi_n.shape
        ) + 1j * np.random.normal(0, noise / np.sqrt(2), Psi_n.shape)

    # Create the new interpolation functions
    Delta_interp = RegularGridInterpolator((y_ax, x_ax), abs(Psi_n))
    theta_interp = RegularGridInterpolator((y_ax, x_ax), np.angle(Psi_n))

    # Update the order parameter
    def Delta(x, y):
        return Delta_interp((y, x))

    def theta(x, y):
        return theta_interp((y, x))

    return Delta, theta


def setup_gaussian_impurities(
    x,
    y,
    mu,
    hx0,
    hy0,
    hz0,
    impurity_positions,
    impurity_sizes,
    impurity_eccentricities,
    impurity_orientations,
    V_imp,
    hx_imp,
    hy_imp,
    hz_imp,
):
    """
    Setup the potential and fields for Gaussian impurities.

    Args:
        x (numpy.ndarray): The x-coordinates.
        y (numpy.ndarray): The y-coordinates.
        mu (float): The chemical potential.
        hx0 (float): The x-component of the magnetic field at the origin.
        hy0 (float): The y-component of the magnetic field at the origin.
        hz0 (float): The z-component of the magnetic field at the origin.
        impurity_positions (List[Tuple[float, float]]): The positions of the impurities.
        impurity_sizes (List[float]): The sizes of the impurities.
        impurity_eccentricities (List[float]): The eccentricities of the impurities.
        impurity_orientations (List[float]): The orientations of the impurities.
        V_imp (List[float]): The potential values of the impurities.
        hx_imp (List[float]): The x-component of the magnetic field values of the impurities.
        hy_imp (List[float]): The y-component of the magnetic field values of the impurities.
        hz_imp (List[float]): The z-component of the magnetic field values of the impurities.

    Returns:
        Tuple[Callable[[float, float], float], Callable[[float, float], float], Callable[[float, float], float], Callable[[float, float], float]]:
            - V (Callable[[float, float], float]): The potential function.
            - hx (Callable[[float, float], float]): The x-component of the magnetic field function.
            - hy (Callable[[float, float], float]): The y-component of the magnetic field function.
            - hz (Callable[[float, float], float]): The z-component of the magnetic field function.
    """

    x_ax = x[0]
    y_ax = y[:, 0]

    # Calculation of the A matrix for each Gaussian potential
    A_matrices = []
    for sigma, e, alpha in zip(
        impurity_sizes, impurity_eccentricities, impurity_orientations
    ):
        R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        Sigma = (
            sigma**2 / 2 * np.array([[np.sqrt(1 - e**2), 0], [0, np.sqrt(1 + e**2)]])
        )
        Sigma_prime = R @ Sigma @ R.T
        A = np.linalg.inv(Sigma_prime) / 2
        A_matrices.append(A)

    # Fields definition
    def V(x, y):
        V = -mu + 0 * x
        for i, pos in enumerate(impurity_positions):
            xp, yp = pos
            r = np.array([x - xp, y - yp])
            V += V_imp[i] * np.exp(
                -np.einsum("aij, ab, bij -> ij", r, A_matrices[i], r)
            )  # / (2 * np.pi) * np.sqrt(la.det(2*A_matrices[i]))
        return V

    def hx(x, y):
        hx = hx0 + 0 * x
        for i, pos in enumerate(impurity_positions):
            xp, yp = pos
            r = np.array([x - xp, y - yp])
            hx += hx_imp[i] * np.exp(
                -np.einsum("aij, ab, bij -> ij", r, A_matrices[i], r)
            )  # / (2 * np.pi) * np.sqrt(la.det(2*A_matrices[i]))
        return hx

    def hy(x, y):
        hy = hy0 + 0 * x
        for i, pos in enumerate(impurity_positions):
            xp, yp = pos
            r = np.array([x - xp, y - yp])
            hy += hy_imp[i] * np.exp(
                -np.einsum("aij, ab, bij -> ij", r, A_matrices[i], r)
            )  # / (2 * np.pi) * np.sqrt(la.det(2*A_matrices[i]))
        return hy

    def hz(x, y):
        hz = hz0 + 0 * x
        for i, pos in enumerate(impurity_positions):
            xp, yp = pos
            r = np.array([x - xp, y - yp])
            hz += hz_imp[i] * np.exp(
                -np.einsum("aij, ab, bij -> ij", r, A_matrices[i], r)
            )  # / (2 * np.pi) * np.sqrt(la.det(2*A_matrices[i]))
        return hz

    # Create the new interpolation functions
    V_interp = RegularGridInterpolator((y_ax, x_ax), V(x, y))
    hx_interp = RegularGridInterpolator((y_ax, x_ax), hx(x, y))
    hy_interp = RegularGridInterpolator((y_ax, x_ax), hy(x, y))
    hz_interp = RegularGridInterpolator((y_ax, x_ax), hz(x, y))

    def V(x, y):
        return V_interp((y, x))

    def hx(x, y):
        return hx_interp((y, x))

    def hy(x, y):
        return hy_interp((y, x))

    def hz(x, y):
        return hz_interp((y, x))

    return V, hx, hy, hz


def setup_Coulomb_impurities(x, y, mu, impurity_positions, V_imp, screening_length):
    """
    Set up the Coulomb impurities for a given set of positions and parameters.

    Parameters:
        x (ndarray): The x-coordinates of the grid.
        y (ndarray): The y-coordinates of the grid.
        mu (float): The chemical potential.
        impurity_positions (list): The positions of the impurities.
        V_imp (list): The potential values of the impurities.
        screening_length (float): The screening length.

    Returns:
        function: The interpolated potential function.

    """

    x_ax = x[0]
    y_ax = y[:, 0]

    # Fields definition
    def V(x, y):
        Vf = -mu + 0 * x

        for i, pos in enumerate(impurity_positions):
            xp, yp = pos
            r = np.array([x - xp, y - yp])
            r_magnitude = np.sqrt(r[0] ** 2 + r[1] ** 2)
            Vf += (
                V_imp[i]
                * np.exp(-r_magnitude / screening_length)
                / np.sqrt(r_magnitude**2 + 1)
            )
        return Vf

    # Create the potential grid
    V_grid = V(x, y)

    # Create the new interpolation function
    V_interp = RegularGridInterpolator((y_ax, x_ax), V_grid)

    def V_func(x, y):
        return V_interp((y, x))

    return V_func


def setup_spin_impurities(
    x,
    y,
    hx0,
    hy0,
    hz0,
    impurity_positions,
    impurity_sizes,
    impurity_eccentricities,
    impurity_orientations,
    hx_imp,
    hy_imp,
    hz_imp,
):
    """
    Set up the spin impurities in a tight-binding model.
    Args:
        x (ndarray): The x-coordinates of the grid points.
        y (ndarray): The y-coordinates of the grid points.
        hx0 (float): The x-component of the external magnetic field.
        hy0 (float): The y-component of the external magnetic field.
        hz0 (float): The z-component of the external magnetic field.
        impurity_positions (List[Tuple[float, float]]): The positions of the impurity centers.
        impurity_sizes (List[float]): The sizes of the impurity Gaussian potentials.
        impurity_eccentricities (List[float]): The eccentricities of the impurity Gaussian potentials.
        impurity_orientations (List[float]): The orientations of the impurity Gaussian potentials.
        hx_imp (List[float]): The x-components of the impurity magnetic fields.
        hy_imp (List[float]): The y-components of the impurity magnetic fields.
        hz_imp (List[float]): The z-components of the impurity magnetic fields.
    Returns:
        Tuple[Callable[[float, float], float], Callable[[float, float], float], Callable[[float, float], float]]:
            The x-component, y-component, and z-component of the magnetic field at each grid point.
    """

    x_ax = x[0]
    y_ax = y[:, 0]

    # Calculation of the A matrix for each Gaussian potential
    A_matrices = []
    for sigma, e, alpha in zip(
        impurity_sizes, impurity_eccentricities, impurity_orientations
    ):
        R = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
        Sigma = (
            sigma**2 / 2 * np.array([[np.sqrt(1 - e**2), 0], [0, np.sqrt(1 + e**2)]])
        )
        Sigma_prime = R @ Sigma @ R.T
        A = np.linalg.inv(Sigma_prime) / 2
        A_matrices.append(A)

    def hx(x, y):
        hx = hx0 + 0 * x
        for i, pos in enumerate(impurity_positions):
            xp, yp = pos
            r = np.array([x - xp, y - yp])
            hx += hx_imp[i] * np.exp(
                -np.einsum("aij, ab, bij -> ij", r, A_matrices[i], r)
            )  # / (2 * np.pi) * np.sqrt(la.det(2*A_matrices[i]))
        return hx

    def hy(x, y):
        hy = hy0 + 0 * x
        for i, pos in enumerate(impurity_positions):
            xp, yp = pos
            r = np.array([x - xp, y - yp])
            hy += hy_imp[i] * np.exp(
                -np.einsum("aij, ab, bij -> ij", r, A_matrices[i], r)
            )  # / (2 * np.pi) * np.sqrt(la.det(2*A_matrices[i]))
        return hy

    def hz(x, y):
        hz = hz0 + 0 * x
        for i, pos in enumerate(impurity_positions):
            xp, yp = pos
            r = np.array([x - xp, y - yp])
            hz += hz_imp[i] * np.exp(
                -np.einsum("aij, ab, bij -> ij", r, A_matrices[i], r)
            )  # / (2 * np.pi) * np.sqrt(la.det(2*A_matrices[i]))
        return hz

    # Create the new interpolation functions
    hx_interp = RegularGridInterpolator((y_ax, x_ax), hx(x, y))
    hy_interp = RegularGridInterpolator((y_ax, x_ax), hy(x, y))
    hz_interp = RegularGridInterpolator((y_ax, x_ax), hz(x, y))

    def hx(x, y):
        return hx_interp((y, x))

    def hy(x, y):
        return hy_interp((y, x))

    def hz(x, y):
        return hz_interp((y, x))

    return hx, hy, hz
