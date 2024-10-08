import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from numpy.typing import ArrayLike
from colorsys import hls_to_rgb


def add_tl_label(ax, text):
    """
    Adds a text label to the specified axes with a specified position and appearance.

    Parameters:
        ax (Axes): The axes to add the label to.
        text (str): The text to display in the label.

    Returns:
        None
    """
    ax.text(
        0.02,
        0.98,
        text,
        horizontalalignment="left",
        verticalalignment="top",
        transform=ax.transAxes,
        bbox=dict(facecolor=[1, 1, 1, 0.90], edgecolor="none", pad=0.25),
    )


def colorize(z: ArrayLike) -> ArrayLike:
    """
    Colorize a complex-valued array.

    Parameters:
    - z (np.ndarray): Input array with complex values.

    Returns:
    - np.ndarray: Colorized array with RGB values.

    The function assigns colors to the complex values based on their phase and amplitude.
    NaN and infinity values are assigned specific colors.

    Colors are determined using the HLS color space conversion.
    """
    z = np.asarray(z)
    n, m = z.shape
    c = np.zeros((n, m, 3))

    # Assign specific colors to NaN and infinity values
    c[np.isinf(z)] = (1.0, 1.0, 1.0)
    c[np.isnan(z)] = (0.5, 0.5, 0.5)

    # Process finite values
    idx = ~(np.isinf(z) + np.isnan(z))
    phase = (np.angle(z[idx]) + np.pi) / (2 * np.pi)
    phase = (phase + 0.5) % 1.0
    amplitude = 1.0 - 1.0 / (1.0 + abs(z[idx]) ** 0.3)

    # Convert phase and amplitude to RGB using HLS color space
    color = [hls_to_rgb(p, a, 0.8) for p, a in zip(phase, amplitude)]
    c[idx] = color

    return c


def complex_plot(x, y, z, ax=None, interpolation="gaussian"):
    """
    Plot the complex field represented by z.

    Parameters:
    - x (ArrayLike): X-coordinates.
    - y (ArrayLike): Y-coordinates.
    - z (ArrayLike): Complex field values.
    - ax (Optional[Axes]): Optional matplotlib Axes to use for plotting.

    Returns:
    - Tuple[Figure, Axes, Union[None, Type[AxesImage]]]: Tuple containing matplotlib Figure, Axes, and AxesImage object.
    """
    x, y, z = map(np.asarray, (x, y, z))

    if x.shape != z.shape or y.shape != z.shape:
        raise ValueError("Input shapes of x, y, and z must be the same.")

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Plot the complex field
    img = ax.imshow(
        colorize(z),
        extent=(x.min(), x.max(), y.min(), y.max()),
        interpolation=interpolation,
    )

    return fig, ax, img


# # Example usage
# x = np.linspace(-1, 1, 100)
# y = np.linspace(-1, 1, 100)
# X, Y = np.meshgrid(x, y)

# # Assuming Delta and theta functions are defined somewhere
# Delta = lambda x, y: np.sqrt(x**2 + y**2)
# theta = lambda x, y: np.arctan2(y, x)

# # Example usage of the complex_plot function
# Z = Delta(X, Y) * np.exp(1j * theta(X, Y))
# fig, ax, img = complex_plot(X, Y, Z)

# plt.show()


def multiplot(x, ys, cs, ax=None, **kwargs):
    """
    Plot multiple lines with smoothly changing colors.

    Parameters:
    - x        : (x_N,) array-like, x-axis values
    - ys       : (x_N, y_N) array-like, y-axis values for multiple lines
    - cs       : (x_N, y_N, 4) array-like, RGBA colors for each point on the lines
    - ax       : matplotlib axes to plot on. If None, uses the current axes
    - **kwargs : additional arguments passed to LineCollection

    Returns:
    - LineCollection object added to the plot
    """

    if ax is None:
        ax = plt.gca()

    x = np.asarray(x)
    ys = np.asarray(ys)

    x_N = x.shape[0]
    y_N = ys.shape[1]

    assert (
        x.shape[0] == ys.shape[0]
    ), "x and ys must have the same length in the first dimension."

    assert (
        ys.shape[:2] == cs.shape[:2] and cs.shape[2] == 4
    ), "ys must have shape (x_N, y_N) and cs must have shape (x_N, y_N, 4)."

    # Prepare arrays
    segments = np.zeros((y_N * (x_N - 1), 2, 2))
    s_colors = np.zeros((y_N * (x_N - 1), 4))

    # For each line
    for n in range(y_N):
        start_idx = n * (x_N - 1)
        end_idx = (n + 1) * (x_N - 1)

        segments[start_idx:end_idx, 0, 0] = x[:-1]
        segments[start_idx:end_idx, 1, 0] = x[1:]
        segments[start_idx:end_idx, 0, 1] = ys[:-1, n]
        segments[start_idx:end_idx, 1, 1] = ys[1:, n]

        # Average the colors for each segment
        s_colors[start_idx:end_idx] = (cs[:-1, n] + cs[1:, n]) / 2

    lc = LineCollection(segments, colors=s_colors, **kwargs)
    ax.add_collection(lc)
    ax.autoscale()
    return lc


# Example usage with three lines and smoothly changing colors between two specified colors for each line

# # Generate data
# x = np.linspace(0, 10, 100)
# ys = np.array([np.sin(x), np.cos(x), np.sin(x) + np.cos(x)]).T  # Three lines

# # Define start and end colors for each line
# start_colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1)]  # Red, Green, Blue
# end_colors = [(1, 1, 0, 1), (0, 1, 1, 1), (1, 0, 1, 1)]    # Yellow, Cyan, Magenta

# # Generate smoothly changing colors for each line
# colors = np.zeros((100, 3, 4))  # (x_N, y_N, 4)
# for i in range(3):
#     cmap = LinearSegmentedColormap.from_list(f'line_{i}_cmap', [start_colors[i], end_colors[i]])
#     norm = plt.Normalize(vmin=0, vmax=99)
#     sm = cm.ScalarMappable(cmap=cmap, norm=norm)
#     colors[:, i, :] = sm.to_rgba(np.linspace(0, 99, 100))

# fig, ax = plt.subplots()
# multiplot(x, ys, colors=colors, ax=ax)
# plt.xlabel('x-axis')
# plt.ylabel('y-axis')
# plt.title('Multiline Plot with Smoothly Changing Colors')
# plt.show()


def multiplot_2d(xs, ys, cs=None, ax=None):
    """
    Plot multiple 2D lines with varying colors.

    Parameters:
    - xs  : list of arrays, each containing x-axis values for a line
    - ys  : list of arrays, each containing y-axis values for a line
    - cs  : list of arrays, each containing RGBA colors for points on the lines
            If None, a default colormap will be used.
    - ax  : matplotlib axes to plot on. If None, uses the current axes.

    Returns:
    - LineCollection object added to the plot
    """

    if ax is None:
        ax = plt.gca()

    if cs is None:
        cs = []
        cmap = plt.get_cmap("viridis")
        for line in ys:
            N_points = len(line)
            norm = plt.Normalize(vmin=0, vmax=N_points - 1)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            cs.append(sm.to_rgba(np.linspace(0, N_points - 1, N_points)))

    segments = []
    s_colors = []

    for x, y, c in zip(xs, ys, cs):
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([points[:-1], points[1:]], axis=1)
        segments.extend(segs)
        s_colors.extend((np.array(c[:-1]) + np.array(c[1:])) / 2)

    lc = LineCollection(segments, colors=s_colors)
    ax.add_collection(lc)
    ax.autoscale()
    return lc
