"""A Python module to calculate values related to the Heinrichs basis,
(1-x**2)T_n(x), where T_n(x) is the Chebyshev polynomial of degree n.

References
----------
[1] John P. Boyd, Chebyshev and Fourier Spectral Methods. Courier Corporation,
(2001).
"""

from package_common.calc_chebyshev import calc_chebyshev
from package_common.utils_debug import under_construction_log


def heinrichs(n_degree: int,
              s_pos: complex | float) -> complex | float:
    """Calculate the value of the Heinrichs basis at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Heinrichs basis.
    s_pos : complex | float
        The position of the point.

    Returns
    -------
    complex | float
        The value of the Heinrichs basis at the point.

    Examples
    --------
    >>> from package_common.calc_heinrichs import heinrichs
    >>> print(heinrichs(3, 0.5))
    -0.75
    """

    return calc_heinrichs(n_degree, s_pos, 0)[0]


def heinrichs_d(n_degree: int,
                s_pos: complex | float) -> complex | float:
    """Calculate the value of the first derivative of the Heinrichs
    basis at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Heinrichs basis.
    s_pos : complex | float
        The position of the point.

    Returns
    -------
    complex | float
        The value of the first derivative of the Heinrichs basis at the point.

    Examples
    --------
    >>> from package_common.calc_heinrichs import heinrichs_d
    >>> print(heinrichs_d(3, 0.5))
    1.0000000000000002
    """

    return calc_heinrichs(n_degree, s_pos, 1)[1]


def heinrichs_d2(n_degree: int,
                 s_pos: complex | float) -> complex | float:
    """Calculate the value of the second derivative of the Heinrichs
    basis at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Heinrichs basis.
    s_pos : complex | float
        The position of the point.

    Returns
    -------
    complex | float
        The value of the second derivative of the Heinrichs basis at the point.

    Examples
    --------
    >>> from package_common.calc_heinrichs import heinrichs_d2
    >>> print(heinrichs_d2(3, 0.5))
    11.000000000000002
    """

    return calc_heinrichs(n_degree, s_pos, 2)[2]


def heinrichs_d3(n_degree: int,
                 s_pos: complex | float) -> complex | float:
    """Calculate the value of the third derivative of the Heinrichs
    basis at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Heinrichs basis.
    s_pos : complex | float
        The position of the point.

    Returns
    -------
    complex | float
        The value of the third derivative of the Heinrichs basis at the point.

    Examples
    --------
    >>> from package_common.calc_heinrichs import heinrichs_d3
    >>> print(heinrichs_d3(3, 0.5))
    -18.000000000000004
    """

    return calc_heinrichs(n_degree, s_pos, 3)[3]


def calc_heinrichs(n_degree: int,
                   s_pos: complex | float,
                   order: int) -> tuple[complex | float, ...]:
    """Helper function to calculate the value of the Heinrichs basis or its
    derivatives at a given point.

    Parameters
    ----------
    n_degree: int
        The degree of the Heinrichs basis.
    s_pos: complex | float
        The position of the point.
    order: int
        The order of the derivative.

    Returns
    -------
    tuple[complex | float, ...]
        The value of the Heinrichs basis or its derivatives at the point.

    Notes
    -----
    This function is a helper function for heinrichs, heinrichs_d,
    heinrichs_d2, and heinrichs_d3.
    """

    tuple_cheb: tuple[complex | float, ...] \
        = calc_chebyshev(n_degree, s_pos, order)

    cheb: complex | float = tuple_cheb[0]
    s_sin_sq: complex | float = 1 - (s_pos**2)
    hein: complex | float = s_sin_sq * cheb
    if order == 0:
        return (hein,)

    cheb_d: complex | float = tuple_cheb[1]
    hein_d: complex | float = s_sin_sq * cheb_d - 2 * s_pos * cheb
    if order == 1:
        return (hein, hein_d)

    cheb_d2: complex | float = tuple_cheb[2]
    hein_d2: complex | float \
        = s_sin_sq * cheb_d2 - 4 * s_pos * cheb_d - 2 * cheb
    if order == 2:
        return (hein, hein_d, hein_d2)

    cheb_d3: complex | float = tuple_cheb[3]
    hein_d3: complex | float \
        = s_sin_sq * cheb_d3 - 6 * s_pos * cheb_d2 - 6 * cheb_d
    if order == 3:
        return (hein, hein_d, hein_d2, hein_d3)

    under_construction_log()
