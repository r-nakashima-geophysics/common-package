"""A Python module to provide the utilities for collocation methods."""

import sys

import numpy as np

from package_common.calc_heinrichs import heinrichs, heinrichs_d, heinrichs_d2
from package_common.common_types import (ArrayComplex, ArrayFloat, Callable,
                                         FloatFunc, Self)
from package_common.default_logger import DefaultLogger
from package_common.spectral_deform import ComplexCoordinate
from package_common.utils_name import create_function_name_logger

type Func4Quad = Callable[[int, float | int | complex], float | complex]


class ChebyshevGaussQuad:
    """Class to perform the Chebyshev-Gauss quadrature."""

    __num_mode: int
    __num_degree: int
    __num_point: int
    __point_array: ArrayFloat

    __flag: bool = False
    __logger: DefaultLogger = DefaultLogger(__name__)

    @classmethod
    def set_class_variable(cls: type[Self],
                           num_mode: int,
                           num_degree: int) -> None:
        """Set the class variables.

        Parameters
        ----------
        num_mode : int
            The number of the eigenmodes.
        num_degree : int
            The number of the degree.
        """

        cls.__num_mode = num_mode
        cls.__num_degree = num_degree
        cls.__num_point = 3 * cls.__num_degree
        cls.__flag = True

        cls.__point_array = np.array(
            [calc_collocation_point(2*i_l-1, 2*cls.__num_point)
             for i_l in range(1, cls.__num_point+1)]
        )

    def __init__(self: Self,
                 *,
                 func_1: Func4Quad,
                 func_2: Func4Quad | None = None,
                 weight: FloatFunc = lambda x: 1,
                 y_complex: ComplexCoordinate) -> None:
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
        y_complex : ComplexCoordinate
            The complex coordinate for spectral deformation.

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
            sys.exit(1)
        else:
            self.__num_mode: int = ChebyshevGaussQuad.__num_mode
            self.__num_degree: int = ChebyshevGaussQuad.__num_degree
            self.__num_point: int = ChebyshevGaussQuad.__num_point
            self.__point_array: ArrayFloat = ChebyshevGaussQuad.__point_array

        self.__flag_func_2: bool = (func_2 is not None)

        self.__array_func_1: ArrayFloat | ArrayComplex
        self.__array_func_2: ArrayFloat | ArrayComplex

        self.__array_weight: ArrayFloat \
            = np.empty(self.__num_point, dtype=np.float64)

        if not y_complex.check_spectral_deform():

            self.__array_func_1 = np.empty(
                (self.__num_degree, self.__num_point), dtype=np.float64)
            self.__array_func_2 = np.empty(
                (self.__num_degree, self.__num_point), dtype=np.float64)

            for i_pos, pos in enumerate(self.__point_array):

                self.__array_func_1[:, i_pos] = [
                    func_1(i_n, pos) for i_n in range(self.__num_degree)]
                if func_2 is not None:
                    self.__array_func_2[:, i_pos] = [
                        func_2(i_n, pos) for i_n in range(self.__num_degree)]

                self.__array_weight[i_pos] = weight(pos) * np.sqrt(1-(pos**2))

        else:

            self.__array_func_1 = np.empty(
                (self.__num_degree, self.__num_point), dtype=np.complex128)
            self.__array_func_2 = np.empty(
                (self.__num_degree, self.__num_point), dtype=np.complex128)

            y_pos: complex
            s_pos: complex | None = None

            for i_pos, pos in enumerate(ChebyshevGaussQuad.__point_array):

                y_pos = y_complex.value_without_spectral_deform(pos)
                if s_pos is None:
                    s_pos = pos + 1j * 0
                s_pos = y_complex.inverse(y_pos, guess=s_pos)

                self.__array_func_1[:, i_pos] = [
                    func_1(i_n, s_pos) for i_n in range(self.__num_degree)]
                if func_2 is not None:
                    self.__array_func_2[:, i_pos] = [
                        func_2(i_n, s_pos) for i_n in range(self.__num_degree)]

                self.__array_weight[i_pos] = weight(pos) * np.sqrt(1-(pos**2))

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

        Notes
        -----
        This method may run inside multiprocessing workers.
        """

        field_1: ArrayComplex = vec_1.T @ self.__array_func_1
        field_2: ArrayComplex
        if (self.__flag_func_2) and (vec_2 is not None):
            field_2 = vec_2.T @ self.__array_func_2
        else:
            field_2 = np.ones((self.__num_mode, self.__num_point),
                              dtype=np.complex128)

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
    sys.exit(1)


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

    sin2: float | complex = 1 - (mu**2)

    return (
        sin2 * heinrichs_d2(n_degree, s_pos) / (mu_d**2)
        - (2*mu/mu_d + sin2*mu_d2/(mu_d**3))
        * heinrichs_d(n_degree, s_pos)
        - (m_order**2) * heinrichs(n_degree, s_pos) / sin2
    )
