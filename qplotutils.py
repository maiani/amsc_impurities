import matplotlib.collections as mcoll
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection


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