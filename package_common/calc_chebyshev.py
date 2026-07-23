"""A Python module to calculate values related to Chebyshev
polynomials.

References
----------
[1] John P. Boyd, Chebyshev and Fourier Spectral Methods. Courier Corporation,
(2001).
"""

import numpy as np

from package_common.common_types import TypeVarFloatComplex
from package_common.utils_debug import under_construction_log


def chebyshev(n_degree: int,
              s_pos: TypeVarFloatComplex) -> TypeVarFloatComplex:
    """Calculate the value of a Chebyshev polynomial at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial.
    s_pos : TypeVarFloatComplex
        The position of the point.

    Returns
    -------
    TypeVarFloatComplex
        The value of the Chebyshev polynomial at the point.

    Examples
    --------
    >>> from package_common.calc_chebyshev import chebyshev
    >>> print(chebyshev(3, 0.5))
    -1.0
    """

    return _calc_chebyshev(n_degree, s_pos, 0)[0]


def chebyshev_d(n_degree: int,
                s_pos: TypeVarFloatComplex) -> TypeVarFloatComplex:
    """Calculate the value of the first derivative of a Chebyshev
    polynomial at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial.
    s_pos : TypeVarFloatComplex
        The position of the point.

    Returns
    -------
    TypeVarFloatComplex
        The value of the first derivative of the Chebyshev polynomial at the
        point.

    Examples
    --------
    >>> from package_common.calc_chebyshev import chebyshev_d
    >>> print(chebyshev_d(3, 0.5))
    4.2423009548996277e-16
    """

    return _calc_chebyshev(n_degree, s_pos, 1)[1]


def chebyshev_d2(n_degree: int,
                 s_pos: TypeVarFloatComplex) -> TypeVarFloatComplex:
    """Calculate the value of the second derivative of a Chebyshev
    polynomial at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial.
    s_pos : TypeVarFloatComplex
        The position of the point.

    Returns
    -------
    TypeVarFloatComplex
        The value of the second derivative of the Chebyshev polynomial at the
        point.

    Examples
    --------
    >>> from package_common.calc_chebyshev import chebyshev_d2
    >>> print(chebyshev_d2(3, 0.5))
    12.000000000000002
    """

    return _calc_chebyshev(n_degree, s_pos, 2)[2]


def chebyshev_d3(n_degree: int,
                 s_pos: TypeVarFloatComplex) -> TypeVarFloatComplex:
    """Calculate the value of the third derivative of a Chebyshev
    polynomial at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial.
    s_pos : TypeVarFloatComplex
        The position of the point.

    Returns
    -------
    TypeVarFloatComplex
        The value of the third derivative of the Chebyshev polynomial at the
        point.

    Examples
    --------
    >>> from package_common.calc_chebyshev import chebyshev_d3
    >>> print(chebyshev_d3(3, 0.5))
    24.000000000000007
    """

    return _calc_chebyshev(n_degree, s_pos, 3)[3]


def _calc_chebyshev(n_degree: int,
                    s_pos: TypeVarFloatComplex,
                    order: int) -> tuple[TypeVarFloatComplex, ...]:
    """Helper function to calculate the value of a Chebyshev polynomial or its
    derivatives at a given point.

    Parameters
    ----------
    n_degree : int
        The degree of the Chebyshev polynomial.
    s_pos : TypeVarFloatComplex
        The position of the point.
    order : int
        The order of the derivative.

    Returns
    -------
    tuple[TypeVarFloatComplex, ...]
        The value of the Chebyshev polynomial or its derivative at the point.

    Notes
    -----
    This function is a helper function for chebyshev, chebyshev_d,
    chebyshev_d2, and chebyshev_d3.
    """

    t: TypeVarFloatComplex = np.acos(s_pos)
    nt: TypeVarFloatComplex = n_degree * t
    cn: TypeVarFloatComplex = np.cos(nt)
    if order == 0:
        return (cn,)

    chebyshev_d: TypeVarFloatComplex
    chebyshev_d2: TypeVarFloatComplex
    chebyshev_d3: TypeVarFloatComplex

    n_sq: int

    if not np.isclose(np.abs(s_pos.real), 1.0):

        s: TypeVarFloatComplex = np.sin(t)
        sn: TypeVarFloatComplex = np.sin(nt)
        chebyshev_d = n_degree * sn / s
        if order == 1:
            return (cn, chebyshev_d)

        n_sq = n_degree**2
        c: TypeVarFloatComplex = np.cos(t)
        chebyshev_d2 = (-n_sq * cn + chebyshev_d * c) / (s**2)
        if order == 2:
            return (cn, chebyshev_d, chebyshev_d2)

        chebyshev_d3 = ((1-n_sq) * chebyshev_d
                        + 3 * chebyshev_d2 * c) / (s**2)
        if order == 3:
            return (cn, chebyshev_d, chebyshev_d2, chebyshev_d3)

    else:
        n_sq = n_degree**2
        chebyshev_d = (s_pos**(n_degree+1)) * n_sq
        if order == 1:
            return (cn, chebyshev_d)

        chebyshev_d2 = s_pos * chebyshev_d * (n_sq-1)/3
        if order == 2:
            return (cn, chebyshev_d, chebyshev_d2)

        chebyshev_d3 = s_pos * chebyshev_d2 * (n_sq-4)/5
        if order == 3:
            return (cn, chebyshev_d, chebyshev_d2, chebyshev_d3)

    under_construction_log()
