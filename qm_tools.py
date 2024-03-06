import matplotlib.collections as mcoll
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
from matplotlib.collections import LineCollection
from scipy.optimize import linear_sum_assignment


def sort_eigensystem(ws, vs):
    """Sort the eigensystem of an Hamiltonian.

    Parameters:
    -----------
    ws : eigenvalues
    vs : eigenvectos

    Returns:
    ws_sorted : sorted eigenvalues
    vs_sorted : sorted eigenvectors
    """

    def best_match(psi1, psi2, threshold=None):
        """Find the best match of two sets of eigenvectors.

        Parameters:
        -----------
        psi1, psi2 : numpy 2D complex arrays
            Arrays of initial and final eigenvectors.
        threshold : float, optional
            Minimal overlap when the eigenvectors are considered belonging to the same band.
            The default value is :math:`1/(2N)^{1/4}`, where :math:`N` is the length of each eigenvector.

        Returns:
        --------
        sorting : numpy 1D integer array
            Permutation to apply to ``psi2`` to make the optimal match.
        diconnects : numpy 1D bool array
            The levels with overlap below the ``threshold`` that should be considered disconnected.
        """
        if threshold is None:
            threshold = (2 * psi1.shape[0]) ** -0.25
        Q = np.abs(psi1.T.conj() @ psi2)  # Overlap matrix
        orig, perm = linear_sum_assignment(-Q)
        return perm, Q[orig, perm] < threshold

    N = ws.shape[0]

    e = ws[0]
    psi = vs[0]

    ws_sorted = [e]
    vs_sorted = [psi]

    for i in range(1, N):
        e2 = ws[i]
        psi2 = vs[i]
        perm, line_breaks = best_match(psi, psi2)
        e2 = e2[perm]
        intermediate = (e + e2) / 2
        intermediate[line_breaks] = None
        psi = psi2[:, perm]
        e = e2

        ws_sorted.append(intermediate)
        ws_sorted.append(e)
        vs_sorted.append(psi)

    return np.array(ws_sorted)[::2], np.array(vs_sorted)


def apply_sw(
    v0,
    v,
    w,
    d,
    H: np.ndarray = None,
    method="svd",
):
    """
    Calculate the effective low-energy Hamiltonian with the Schrieffer-Wolff transformation.

        Parameters:
            - v0 : unperturbed system eigenvectors
            - v  : eigenvectors of the full system
            - w  : eigenvalues of the full system
            - H  : Hamiltonian (needed only for method "sqrt")
            - d  : desired dimension of the effective Hamiltonian
            - method : "sqrt" for matrix square root method and "svd" for SVD version

        Returns:
            - H_eff : effective Hamiltonian in the v0 basis
            - U_sw  : Schrieffer-Wolff unitary (only for "sqrt" method)
    """

    assert v0.shape[-1] >= d
    assert v.shape[-1] >= d

    if H is not None:
        assert H.shape[0] == v.shape[-2]
        assert H.shape[1] == v.shape[-2]

    U_sw = None

    if method == "svd":
        P0 = v0[:, :d].conj().T
        P = v[:, :d]

        B = P0 @ P

        U, s, Vh = la.svd(B)
        A = U @ Vh

        H_eff = A @ np.diag(w[:d]) @ A.conj().T

    elif method == "sqrt":
        # TODO :  Rewrite P0 in the eigenvalue base to make it faster
        P0 = v0[:, :d] @ v0[:, :d].conj().T
        P = v[:, :d] @ v[:, :d].conj().T

        I = np.eye(H.shape[0])

        U_sw = la.sqrtm((2 * P0 - I) @ (2 * P - I))

        H_sw_full = U_sw @ H @ U_sw.conj().T
        H_eff = v0[:, :d].conj().T @ H_sw_full @ v0[:, :d]

    else:
        raise ValueError("""Method should be "sqrt" or "svd".""")

    return H_eff, U_sw


def plot_spectrum(x, v0, w, v, c0, **kwargs):
    """
    Plot the spectrum of some Hamiltonian with chaning linecolor depending on
    the value of the projection onto a basis v0. The eigensystem v and w need not
    to be order, but if it is ordered the results are better.

    Parameters:
    - x    : (x_N) x axis
    - v0   : (x_N, v_N, w0_N) or (v_N, w0_N)  array of the H0 eiegnvectors, needs to be ordered
    - w    : (x_N, w_N)  array of the H eigenvalues
    - v    : (x_N, v_N, w_N)  array of the H eiegnvectors
    - c0   : (w0_N)            array of the colors in RGB or RGBA
    - ax   : matplotlib axes to fill
    - **kwargs : arguments passed to LineCollection
    """

    x_N, v_N, w0_N = v0.shape
    w_N = w.shape[1]

    # If x_N=1, tile the array
    if len(v0) == 2:
        v0 = np.tile(v0, (x_N, 1, 1))

    # Check x_N
    assert (
        (x.shape[0] == v0.shape[0])
        and (x.shape[0] == w.shape[0])
        and (x.shape[0] == v.shape[0])
    )

    # CHeck v_N
    assert v0.shape[1] == v.shape[1]

    # Check w0_N
    assert v0.shape[2] == c0.shape[0]

    # Check w_N
    assert w.shape[1] == v.shape[2]

    # Reshape color matrix to match the number of states
    _c0 = np.zeros((w0_N, 4), dtype=float)
    _c0[: min(w0_N, len(c0))] = c0[: min(w0_N, len(c0))]

    # Calulate overlaps
    overlaps = np.abs(np.einsum("abc, abd -> acd", v0.conj(), v)) ** 2

    # Prepare arrays
    segments = np.zeros((w_N * (x_N - 1), 2, 2))
    segments *= np.nan
    colors = np.zeros((w_N * (x_N - 1), 4))

    # For each line
    for n in range(w_N):
        segments[n * (x_N - 1) : (n + 1) * (x_N - 1), 0, 0] = x[:-1]
        segments[n * (x_N - 1) : (n + 1) * (x_N - 1), 1, 0] = x[1:]
        segments[n * (x_N - 1) : (n + 1) * (x_N - 1), 0, 1] = w[:-1, n]
        segments[n * (x_N - 1) : (n + 1) * (x_N - 1), 1, 1] = w[1:, n]

        colors[n * (x_N - 1) : (n + 1) * (x_N - 1), :] = np.clip(
            np.einsum("bc, ab -> ac", _c0, overlaps[:, :, n])[:-1], 0, 1
        )

    lc = LineCollection(segments, edgecolors=colors, **kwargs)

    return lc


# TEST CODE
# x_N = 200

# x = np.linspace(0, 2*np.pi, x_N)

# H = np.zeros((x_N, 3, 3))
# H[:, 0, 0] = np.cos(x + 0/3 * 2 * np.pi)
# H[:, 1, 1] = np.cos(x + 1/3 * 2 * np.pi)
# H[:, 2, 2] = np.cos(x + 2/3 * 2 * np.pi)

# w0 = np.zeros((x_N, 3), dtype=float)
# v0 = np.zeros((x_N, 3, 3), dtype=complex)
# for n in range(x_N):
#     w0[n], v0[n] = np.linalg.eigh(H[n])

# w0, v0 = sort_eigenvalues(w0, v0)

# H[:, 0, 1] = 0.05
# H[:, 1, 2] = 0.05
# H[:, 0, 2] = 0.05
# H[:, 1, 0] = 0.05
# H[:, 2, 1] = 0.05
# H[:, 2, 0] = 0.05

# w = np.zeros((x_N, 3), dtype=float)
# v = np.zeros((x_N, 3, 3), dtype=complex)
# for n in range(x_N):
#     w[n], v[n] = np.linalg.eigh(H[n])

# c0 = matplotlib_default_colors[:3]

# fig, ax = plt.subplots(figsize=(3.375, 3.375))
# plot_spectrum(x, w0=w0, v0=v0, w=w, v=v, c0=c0, ax=ax)
# ax.plot(x, w0[:, 0], "C0--")
# ax.plot(x, w0[:, 1], "C1--")
# ax.plot(x, w0[:, 2], "C2--")

# ax.set_xlim(0, 2*np.pi)
# ax.set_ylim(-1, 1)

# fig.tight_layout()
