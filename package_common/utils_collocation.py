"""A Python module to provide the utilities for collocation methods."""

import numpy as np

from package_common.calc_chebyshev import _calc_chebyshev
from package_common.common_types import (ArrayComplex, ArrayFloat, Callable,
                                         FloatFunc, Self)
from package_common.default_logger import DefaultLogger
from package_common.spectral_deform import ComplexCoordinate
from package_common.utils_name import create_function_name_logger

type Func4Quad = Callable[[int, float | int | complex], float | complex]


class ChebyshevGaussQuad:
    """Class to perform the Chebyshev-Gauss quadrature."""

    __num_degree: int
    __num_point: int
    __spectral_deform: bool
    __point_array: ArrayFloat
    __point_analytic_cont: ArrayComplex

    __flag: bool = False
    __logger: DefaultLogger = DefaultLogger(__name__)

    @classmethod
    def set_class_variable(cls: type[Self],
                           num_degree: int,
                           *,
                           y_complex: ComplexCoordinate) -> None:
        """Set the class variables.

        Parameters
        ----------
        num_degree : int
            The number of the degree.
        y_complex : ComplexCoordinate
            The complex coordinate for spectral deformation.
        """

        cls.__num_degree = num_degree
        cls.__num_point = 3 * cls.__num_degree
        cls.__spectral_deform = y_complex.check_spectral_deform()
        cls.__flag = True

        cls.__point_array = np.array(
            [calc_collocation_point(2*i_l-1, 2*cls.__num_point)
             for i_l in range(1, cls.__num_point+1)],
            dtype=np.float64
        )

        if cls.__spectral_deform:
            y_pos: complex
            guess: complex
            cls.__point_analytic_cont \
                = np.empty(cls.__num_point, dtype=np.complex128)
            for i_pos, pos in enumerate(cls.__point_array):
                y_pos = y_complex.value_without_spectral_deform(pos)
                if i_pos == 0:
                    guess = pos + 1j * 0
                else:
                    guess = cls.__point_analytic_cont[i_pos-1]
                cls.__point_analytic_cont[i_pos] \
                    = y_complex.inverse(y_pos, guess=guess)

    def __init__(self: Self,
                 *,
                 func_1: Func4Quad,
                 func_2: Func4Quad | None = None,
                 weight: FloatFunc = lambda x: 1) -> None:
        """Initialize an instance of the ChebyshevGaussQuad class.

        Parameters
        ----------
        func_1 : Func4Quad
            The function associated with the first vector.
        func_2 : Func4Quad, optional, default None
            The function associated with the second vector.
        weight : FloatFunc, optional, default lambda x: 1
            The weight function for the quadrature of the first and second
            vectors.

        Warnings
        --------
        `set_class_variable` class method has not been executed
            If `set_class_variable` class method has not been executed.

        Notes
        -----
        The weight 1/sqrt(1-x^2) for the Chebyshev-Gauss quadrature is included
        in self.__array_weight automatically.
        """

        if not ChebyshevGaussQuad.__flag:
            ChebyshevGaussQuad.__logger.error(
                '`set_class_variable` class method has not been executed')
        else:
            num_degree: int = ChebyshevGaussQuad.__num_degree
            self.__num_point: int = ChebyshevGaussQuad.__num_point
            point_array: ArrayFloat | ArrayComplex \
                = ChebyshevGaussQuad.__point_array

        self.__flag_func_2: bool = func_2 is not None

        self.__array_func_1: ArrayFloat | ArrayComplex
        self.__array_func_2: ArrayFloat | ArrayComplex

        self.__array_weight: ArrayFloat = (
            np.vectorize(weight)(point_array) * np.sqrt(1.0 - point_array**2)
        )

        dtype: type
        if not ChebyshevGaussQuad.__spectral_deform:
            dtype = np.float64
        else:
            dtype = np.complex128
            point_array = ChebyshevGaussQuad.__point_analytic_cont

        self.__array_func_1 = np.empty(
            (num_degree, self.__num_point), dtype=dtype)
        if func_2 is not None:
            self.__array_func_2 = np.empty(
                (num_degree, self.__num_point), dtype=dtype)

        for i_pos, s_pos in enumerate(point_array):
            self.__array_func_1[:, i_pos] = [
                func_1(i_n, s_pos) for i_n in range(num_degree)]
            if func_2 is not None:
                self.__array_func_2[:, i_pos] = [
                    func_2(i_n, s_pos) for i_n in range(num_degree)]

    def quadrature(self: Self,
                   *,
                   vec_1: ArrayComplex,
                   vec_2: ArrayComplex | None = None) -> ArrayComplex:
        """Calculate the integrals of conj(field_1) * field_2 * weight using
        the Chebyshev-Gauss quadrature for all eigenmodes, where field_1 =
        sum(vec_1 * func_1) and field_2 = sum(vec_2 * func_2). The weight
        1/sqrt(1-x^2) for the Chebyshev-Gauss quadrature is included in
        self.__array_weight automatically.

        Parameters
        ----------
        vec_1 : ArrayComplex
            The first vector.
        vec_2 : ArrayComplex, optional, default None
            The second vector.

        Returns
        -------
        integral : ArrayComplex
            The result of the Chebyshev-Gauss quadrature.

        Warnings
        --------
        Inconsistent input
            If vec_2 is None, although func_2 is not None.

        Notes
        -----
        This method may run inside multiprocessing workers.
        """

        if self.__flag_func_2 and (vec_2 is None):
            self.__logger.error('Inconsistent input')

        field_1: ArrayComplex = vec_1.T @ self.__array_func_1
        field_2: float | ArrayComplex
        if self.__flag_func_2 and (vec_2 is not None):
            field_2 = vec_2.T @ self.__array_func_2
        else:
            field_2 = 1

        integral: ArrayComplex \
            = np.sum(self.__array_weight * np.conj(field_1) * field_2, axis=1)
        integral *= np.pi / self.__num_point

        return integral


def calc_collocation_point(i_l: int,
                           num_point: int) -> float:
    """Calculate a Gauss-Lobatto collocation point.

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
        If the input value is not within [0, num_point], or if num_point is not
        positive.
    """

    if (0 <= i_l <= num_point) and (num_point > 0):
        return -np.cos(i_l*np.pi/num_point)

    logger: DefaultLogger = create_function_name_logger()
    logger.error('Invalid argument')


def spherical_laplacian_heinrichs(
        m_order: int,
        n_degree: int,
        s_pos: float | complex,
        mu_complex: ComplexCoordinate) -> float | complex:
    """Calculate the spherical horizontal Laplacian of the Heinrichs
    basis at a given point.

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order).
    n_degree : int
        The degree of the Heinrichs basis.
    s_pos : float | complex
        The position of the point.
    mu_complex : ComplexCoordinate
        The complex coordinate for spectral deformation.

    Returns
    -------
    float | complex
        The value of the spherical horizontal Laplacian of the Heinrichs basis
        at the point.
    """

    mu: float | complex
    mu_d: float | complex
    mu_d2: float | complex
    if mu_complex.check_spectral_deform():
        mu = mu_complex.value(s_pos)
        mu_d = mu_complex.value_d(s_pos)
        mu_d2 = mu_complex.value_d2(s_pos)
    else:
        s_pos_real: float = s_pos.real
        mu = mu_complex.r_value(s_pos_real)
        mu_d = mu_complex.r_value_d(s_pos_real)
        mu_d2 = mu_complex.r_value_d2(s_pos_real)

    sin_sq: float | complex = 1 - (mu**2)

    chebyshev: float | complex
    chebyshev_d: float | complex
    chebyshev_d2: float | complex
    chebyshev, chebyshev_d, chebyshev_d2 = _calc_chebyshev(n_degree, s_pos, 2)

    s_sin_sq: float | complex = 1 - (s_pos**2)
    heinrichs: float | complex = s_sin_sq * chebyshev
    heinrichs_d: float | complex \
        = s_sin_sq * chebyshev_d - 2 * s_pos * chebyshev
    heinrichs_d2: float | complex \
        = s_sin_sq * chebyshev_d2 - 4 * s_pos * chebyshev_d - 2 * chebyshev

    return (
        sin_sq * heinrichs_d2 / (mu_d**2)
        - (2*mu/mu_d + sin_sq*mu_d2/(mu_d**3))
        * heinrichs_d - (m_order**2) * heinrichs / sin_sq
    )
