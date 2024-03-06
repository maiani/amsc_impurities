from colorsys import hls_to_rgb
from typing import Optional, Tuple, Type, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from numpy.typing import ArrayLike


def add_tl_label(ax, text):
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


def complex_plot(
    x: ArrayLike, y: ArrayLike, z: ArrayLike, ax: Optional[Axes] = None
) -> Tuple[Figure, Axes, AxesImage]:
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
        interpolation="bilinear",
    )

    return fig, ax, img


if __name__ == "__main__":
    # Example usage
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)

    # Assuming Delta and theta functions are defined somewhere
    Delta = lambda x, y: np.sqrt(x**2 + y**2)
    theta = lambda x, y: np.arctan2(y, x)

    # Example usage of the complex_plot function
    Z = Delta(X, Y) * np.exp(1j * theta(X, Y))
    fig, ax, img = complex_plot(X, Y, Z)

    plt.show()
