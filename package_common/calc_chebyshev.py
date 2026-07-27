"""A Python module to calculate values related to Chebyshev
polynomials.

References
----------
[1] John P. Boyd, Chebyshev and Fourier Spectral Methods. Courier Corporation,
(2001).
"""

import cmath
import math

import numpy as np

from package_common.utils_debug import under_construction_log


def chebyshev(n_degree: int,
              s_pos: complex | float) -> complex | float:
    """Calculate the value of a Chebyshev polynomial at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial.
    s_pos : complex | float
        The position of the point.

    Returns
    -------
    complex | float
        The value of the Chebyshev polynomial at the point.

    Examples
    --------
    >>> from package_common.calc_chebyshev import chebyshev
    >>> print(chebyshev(3, 0.5))
    -1.0
    """

    return calc_chebyshev(n_degree, s_pos, 0)[0]


def chebyshev_d(n_degree: int,
                s_pos: complex | float) -> complex | float:
    """Calculate the value of the first derivative of a Chebyshev
    polynomial at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial.
    s_pos : complex | float
        The position of the point.

    Returns
    -------
    complex | float
        The value of the first derivative of the Chebyshev polynomial at the
        point.

    Examples
    --------
    >>> from package_common.calc_chebyshev import chebyshev_d
    >>> print(chebyshev_d(3, 0.5))
    4.2423009548996277e-16
    """

    return calc_chebyshev(n_degree, s_pos, 1)[1]


def chebyshev_d2(n_degree: int,
                 s_pos: complex | float) -> complex | float:
    """Calculate the value of the second derivative of a Chebyshev
    polynomial at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial.
    s_pos : complex | float
        The position of the point.

    Returns
    -------
    complex | float
        The value of the second derivative of the Chebyshev polynomial at the
        point.

    Examples
    --------
    >>> from package_common.calc_chebyshev import chebyshev_d2
    >>> print(chebyshev_d2(3, 0.5))
    12.000000000000002
    """

    return calc_chebyshev(n_degree, s_pos, 2)[2]


def chebyshev_d3(n_degree: int,
                 s_pos: complex | float) -> complex | float:
    """Calculate the value of the third derivative of a Chebyshev
    polynomial at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial.
    s_pos : complex | float
        The position of the point.

    Returns
    -------
    complex | float
        The value of the third derivative of the Chebyshev polynomial at the
        point.

    Examples
    --------
    >>> from package_common.calc_chebyshev import chebyshev_d3
    >>> print(chebyshev_d3(3, 0.5))
    24.000000000000007
    """

    return calc_chebyshev(n_degree, s_pos, 3)[3]


def calc_chebyshev(n_degree: int,
                   s_pos: complex | float,
                   order: int) -> tuple[complex | float, ...]:
    """Helper function to calculate the value of a Chebyshev polynomial or its
    derivatives at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial.
    s_pos : complex | float
        The position of the point.
    order : int
        The order of the derivative.

    Returns
    -------
    tuple[complex | float, ...]
        The value of the Chebyshev polynomial or its derivative at the point.

    Notes
    -----
    This function is a helper function for chebyshev, chebyshev_d,
    chebyshev_d2, and chebyshev_d3.
    """

    is_complex: bool = isinstance(s_pos, complex)

    t: complex | float \
        = cmath.acos(s_pos) if is_complex else math.acos(s_pos)
    nt: complex | float = n_degree * t
    cos_nt: complex | float \
        = cmath.cos(nt) if is_complex else math.cos(nt)
    if order == 0:
        return (cos_nt,)

    cheb_d: complex | float
    cheb_d2: complex | float
    cheb_d3: complex | float

    n_sq: int

    sin_t: complex | float = cmath.sin(t) if is_complex else math.sin(t)
    if not np.isclose(sin_t, 0.0):

        sin_nt: complex | float \
            = cmath.sin(nt) if is_complex else math.sin(nt)
        cheb_d = n_degree * sin_nt / sin_t
        if order == 1:
            return (cos_nt, cheb_d)

        n_sq = n_degree**2
        cos_t: complex | float = s_pos
        sin_t_sq: complex | float = sin_t**2
        cheb_d2 = (-n_sq * cos_nt + cheb_d * cos_t) / sin_t_sq
        if order == 2:
            return (cos_nt, cheb_d, cheb_d2)

        cheb_d3 = ((1-n_sq) * cheb_d + 3 * cheb_d2 * cos_t) / sin_t_sq
        if order == 3:
            return (cos_nt, cheb_d, cheb_d2, cheb_d3)

    else:
        n_sq = n_degree**2
        cheb_d = (s_pos**(n_degree+1)) * n_sq
        if order == 1:
            return (cos_nt, cheb_d)

        cheb_d2 = s_pos * cheb_d * (n_sq-1)/3
        if order == 2:
            return (cos_nt, cheb_d, cheb_d2)

        cheb_d3 = s_pos * cheb_d2 * (n_sq-4)/5
        if order == 3:
            return (cos_nt, cheb_d, cheb_d2, cheb_d3)

    under_construction_log()
