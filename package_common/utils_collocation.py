"""A Python module to provide the utilities for collocation methods."""


import sys

import numpy as np

from package_common.calc_heinrichs import heinrichs, heinrichs_d, heinrichs_d2
from package_common.default_logger import DefaultLogger
from package_common.spectral_deform import ComplexCoordinate
from package_common.utils_name import create_function_name_logger


def calc_collocation_point(i_l: int,
                           num_point: int) -> float:
    """Calculate the Gauss-Lobatto collocation points.

    Parameters
    ----------
    i_l : int
        The index of the collocation point.
    num_point : int
        The number of the collocation points.

    Returns
    -------
    float
        The position of the collocation point.

    Warnings
    --------
    Invalid argument
        If the input value is not within [0, num_point].
    """

    logger: DefaultLogger = create_function_name_logger()
    if 0 <= i_l <= num_point:
        return -np.cos(i_l*np.pi/num_point)

    logger.error('Invalid argument')
    sys.exit(1)


def spherical_laplacian_heinrichs(
        m_order: int,
        n_degree: int,
        s_pos: float,
        mu_complex: ComplexCoordinate) -> float | complex:
    """Calculate the spherical horizontal Laplacian of the Heinrichs
    basis at a given point.

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order).
    n_degree : int
        The degree of the Heinrichs basis.
    s_pos : float
        The position of the point.
    mu_complex : ComplexCoordinate
        The complex coordinate for spectral deformation.

    Returns
    -------
    float | complex
        The value of the spherical horizontal Laplacian of the Heinrichs
        basis at the point.
    """

    mu: float | complex
    mu_d: float | complex
    mu_d2: float | complex
    if mu_complex.check_spectral_deform():
        mu = mu_complex.value(s_pos)
        mu_d = mu_complex.value_d(s_pos)
        mu_d2 = mu_complex.value_d2(s_pos)
    else:
        mu = mu_complex.r_value(s_pos)
        mu_d = mu_complex.r_value_d(s_pos)
        mu_d2 = mu_complex.r_value_d2(s_pos)

    sin2: float | complex = 1 - (mu**2)

    return (
        sin2 * heinrichs_d2(n_degree, s_pos) / (mu_d**2)
        - (2*mu/mu_d + sin2*mu_d2/(mu_d**3))
        * heinrichs_d(n_degree, s_pos)
        - (m_order**2) * heinrichs(n_degree, s_pos) / sin2
    )
