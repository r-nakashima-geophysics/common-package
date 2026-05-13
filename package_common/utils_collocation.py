"""A Python module to provide the utilities for collocation methods."""


import sys

import numpy as np

from package_common.calc_heinrichs import heinrichs, heinrichs_d, heinrichs_d2
from package_common.common_types import ArrayComplex, ArrayFloat, Callable
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

    if 0 <= i_l <= num_point:
        return -np.cos(i_l*np.pi/num_point)

    logger: DefaultLogger = create_function_name_logger()

    logger.error('Invalid argument')
    sys.exit(1)


def chebyshev_gauss_quad(
        num_mode: int,
        *,
        vec_1: ArrayComplex,
        func_1: Callable[[int, float | complex], float | complex],
        vec_2: ArrayComplex,
        func_2: Callable[[int, float | complex], float | complex],
        weight_func: Callable[[float], float] = lambda x: 1,
        y_complex: ComplexCoordinate) -> ArrayComplex:
    """Calculate the integrals of conj(field_1) * field_2 * weight_func /
    sqrt(1-x^2) using the Chebyshev-Gauss quadrature for all eigenmodes, where
    field_1 = sum(vec_1 * func_1) and field_2 = sum(vec_2 * func_2).

    Parameters
    ----------
    vec_1 : ArrayComplex
        The first vector.
    func_1 : Callable[[int, float | complex], float | complex]
        The function associated with the first vector.
    vec_2 : ArrayComplex
        The second vector.
    func_2 : Callable[[int, float | complex], float | complex]
        The function associated with the second vector.
    weight_func : Callable[[float], float]
        The weight function for the quadrature except for the factor
        1/sqrt(1-x^2).
    y_complex : ComplexCoordinate
        The complex coordinate for spectral deformation.

    Returns
    -------
    integral : ArrayComplex
        The result of the Chebyshev-Gauss quadrature.
    """

    s_pos: float | complex

    num_degree: int = vec_1.shape[0]
    num_point: int = 3 * num_degree

    weight: float

    func_1_s: ArrayFloat | ArrayComplex
    func_2_s: ArrayFloat | ArrayComplex
    field_1: ArrayComplex
    field_2: ArrayComplex
    integral: ArrayComplex = np.zeros(num_mode, dtype=np.complex128)

    if not y_complex.check_spectral_deform():

        for i_s in range(1, num_point+1):
            s_pos = calc_collocation_point(2*i_s-1, 2*num_point)

            func_1_s = np.array(
                [func_1(i_n, s_pos) for i_n in range(num_degree)])
            func_2_s = np.array(
                [func_2(i_n, s_pos) for i_n in range(num_degree)])

            field_1 = func_1_s @ vec_1
            field_2 = func_2_s @ vec_2
            weight = weight_func(s_pos) / np.sqrt(1-(s_pos**2))

            integral += weight * np.conj(field_1) * field_2

        integral *= np.pi / num_point

    else:

        norm_y_pos: float
        y_pos: complex
        for i_y in range(1, num_point+1):
            norm_y_pos = calc_collocation_point(2*i_y-1, 2*num_point)
            y_pos = y_complex.value(norm_y_pos).real
            s_pos = y_complex.inverse(y_pos)

            func_1_s = np.array(
                [func_1(i_n, s_pos) for i_n in range(num_degree)])
            func_2_s = np.array(
                [func_2(i_n, s_pos) for i_n in range(num_degree)])

            field_1 = func_1_s @ vec_1
            field_2 = func_2_s @ vec_2
            weight = weight_func(y_pos) / np.sqrt(1-(y_pos**2))

            integral += weight * np.conj(field_1) * field_2

        integral *= np.pi / num_point

    return integral


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
