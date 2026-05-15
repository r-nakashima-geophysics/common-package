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
                 func_1a: Func4Quad,
                 func_1b: Func4Quad | None = None,
                 func_2a: Func4Quad | None = None,
                 func_2b: Func4Quad | None = None,
                 weight_1: FloatFunc | None = None,
                 weight_2: FloatFunc | None = None,
                 y_complex: ComplexCoordinate) -> None:
        """Initialize an instance of the ChebyshevGaussQuad class.

        Parameters
        ----------
        func_1a : Func4Quad
            The function associated with the first vector in the first term.
        func_1b : Func4Quad, optional, default None
            The function associated with the second vector in the first term.
        func_2a : Func4Quad, optional, default None
            The function associated with the first vector in the second term.
        func_2b : Func4Quad, optional, default None
            The function associated with the second vector in the second term.
        weight_1 : FloatFunc, optional, default None
            The weight function for the quadrature of the first and second
            vectors.
        weight_2 : FloatFunc, optional, default None
            The weight function for the quadrature of the third and fourth
            vectors.
        y_complex : ComplexCoordinate
            The complex coordinate for spectral deformation.

        Warnings
        --------
        `set_class_variable` class method has not been executed.
            If `set_class_variable` class method has not been executed.
        """

        if not ChebyshevGaussQuad.__flag:
            ChebyshevGaussQuad.__logger.error(
                '`set_class_variable` class method has not been executed.')
            sys.exit(1)
        else:
            self.__num_mode: int = ChebyshevGaussQuad.__num_mode
            self.__num_degree: int = ChebyshevGaussQuad.__num_degree
            self.__num_point: int = ChebyshevGaussQuad.__num_point
            self.__point_array: ArrayFloat = ChebyshevGaussQuad.__point_array

        self.__func_1b: Func4Quad | None = func_1b
        self.__func_2a: Func4Quad | None = func_2a
        self.__func_2b: Func4Quad | None = func_2b
        self.__weight_1: FloatFunc | None = weight_1
        self.__weight_2: FloatFunc | None = weight_2

        self.__array_func_1a: ArrayFloat | ArrayComplex
        self.__array_func_1b: ArrayFloat | ArrayComplex
        self.__array_func_2a: ArrayFloat | ArrayComplex
        self.__array_func_2b: ArrayFloat | ArrayComplex

        self.__array_weight_1: ArrayFloat \
            = np.empty(self.__num_point, dtype=np.float64)
        self.__array_weight_2: ArrayFloat \
            = np.empty(self.__num_point, dtype=np.float64)

        if not y_complex.check_spectral_deform():

            self.__array_func_1a = np.empty(
                (self.__num_degree, self.__num_point), dtype=np.float64)
            self.__array_func_1b = np.empty(
                (self.__num_degree, self.__num_point), dtype=np.float64)
            self.__array_func_2a = np.empty(
                (self.__num_degree, self.__num_point), dtype=np.float64)
            self.__array_func_2b = np.empty(
                (self.__num_degree, self.__num_point), dtype=np.float64)

            for i_pos, pos in enumerate(self.__point_array):

                self.__array_func_1a[:, i_pos] = [
                    func_1a(i_n, pos) for i_n in range(self.__num_degree)]
                if func_1b is not None:
                    self.__array_func_1b[:, i_pos] = [
                        func_1b(i_n, pos) for i_n in range(self.__num_degree)]
                if func_2a is not None:
                    self.__array_func_2a[:, i_pos] = [
                        func_2a(i_n, pos) for i_n in range(self.__num_degree)]
                if func_2b is not None:
                    self.__array_func_2b[:, i_pos] = [
                        func_2b(i_n, pos) for i_n in range(self.__num_degree)]

                if weight_1 is not None:
                    self.__array_weight_1[i_pos] \
                        = weight_1(pos) / np.sqrt(1-(pos**2))
                if weight_2 is not None:
                    self.__array_weight_2[i_pos] \
                        = weight_2(pos) / np.sqrt(1-(pos**2))

        else:

            self.__array_func_1a = np.empty(
                (self.__num_degree, self.__num_point), dtype=np.complex128)
            self.__array_func_1b = np.empty(
                (self.__num_degree, self.__num_point), dtype=np.complex128)
            self.__array_func_2a = np.empty(
                (self.__num_degree, self.__num_point), dtype=np.complex128)
            self.__array_func_2b = np.empty(
                (self.__num_degree, self.__num_point), dtype=np.complex128)

            y_pos: complex
            s_pos: complex

            for i_pos, pos in enumerate(ChebyshevGaussQuad.__point_array):

                y_pos = y_complex.value(pos)
                s_pos = y_complex.inverse(y_pos)

                self.__array_func_1a[:, i_pos] = [
                    func_1a(i_n, s_pos) for i_n in range(self.__num_degree)]
                if func_1b is not None:
                    self.__array_func_1b[:, i_pos] = [
                        func_1b(i_n, s_pos)
                        for i_n in range(self.__num_degree)]
                if func_2a is not None:
                    self.__array_func_2a[:, i_pos] = [
                        func_2a(i_n, s_pos)
                        for i_n in range(self.__num_degree)]
                if func_2b is not None:
                    self.__array_func_2b[:, i_pos] = [
                        func_2b(i_n, s_pos)
                        for i_n in range(self.__num_degree)]

                if weight_1 is not None:
                    self.__array_weight_1[i_pos] \
                        = weight_1(pos) / np.sqrt(1-(pos**2))
                if weight_2 is not None:
                    self.__array_weight_2[i_pos] \
                        = weight_2(pos) / np.sqrt(1-(pos**2))

    def quadrature(self: Self,
                   *,
                   vec_1a: ArrayComplex,
                   vec_1b: ArrayComplex | None = None,
                   vec_2a: ArrayComplex | None = None,
                   vec_2b: ArrayComplex | None = None) -> ArrayComplex:
        """Calculate the integrals of (conj(field_1a) * field_1b * weight_1
        + conj(field_2a) * field_2b * weight_2) using the Chebyshev-Gauss
        quadrature for all eigenmodes, where field_1a = sum(vec_1a * func_1a),
        field_1b = sum(vec_1b * func_1b), field_2a = sum(vec_2a * func_2a), and
        field_2b = sum(vec_2b * func_2b).

        Parameters
        ----------
        vec_1a : ArrayComplex
            The first vector.
        vec_1b : ArrayComplex, optional, default None
            The second vector.
        vec_2a : ArrayComplex, optional, default None
            The third vector.
        vec_2b : ArrayComplex, optional, default None
            The fourth vector.

        Returns
        -------
        integral : ArrayComplex
            The result of the Chebyshev-Gauss quadrature.

        Notes
        -----
        This method may run inside multiprocessing workers.
        """

        field_1a: ArrayComplex = np.empty(self.__num_mode, dtype=np.complex128)
        field_1b: ArrayComplex = np.ones(self.__num_mode, dtype=np.complex128)
        field_2a: ArrayComplex = np.zeros(self.__num_mode, dtype=np.complex128)
        field_2b: ArrayComplex = np.ones(self.__num_mode, dtype=np.complex128)

        weight_1: ArrayFloat = np.ones(self.__num_mode, dtype=np.float64)
        weight_2: ArrayFloat = np.ones(self.__num_mode, dtype=np.float64)

        integral: ArrayComplex = np.zeros(self.__num_mode, dtype=np.complex128)

        for i_pos in range(len(self.__point_array)):

            field_1a = self.__array_func_1a[:, i_pos] @ vec_1a

            if (self.__func_1b is not None) and (vec_1b is not None):
                field_1b = self.__array_func_1b[:, i_pos] @ vec_1b

            if (self.__func_2a is not None) and (vec_2a is not None):
                field_2a = self.__array_func_2a[:, i_pos] @ vec_2a

            if (self.__func_2b is not None) and (vec_2b is not None):
                field_2b = self.__array_func_2b[:, i_pos] @ vec_2b

            if self.__weight_1 is not None:
                weight_1 = self.__array_weight_1[i_pos]
            if self.__weight_2 is not None:
                weight_2 = self.__array_weight_2[i_pos]

            integral += weight_1 * np.conj(field_1a) * field_1b
            integral += weight_2 * np.conj(field_2a) * field_2b

        integral *= np.pi / self.__num_point

        return integral


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
