"""A Python module to provide the utilities for collocation methods."""

import numpy as np

from package_common.calc_heinrichs import calc_heinrichs
from package_common.common_types import (ArrayComplex, ArrayFloat, Callable,
                                         FuncFloat, cast)
from package_common.default_logger import DefaultLogger
from package_common.spectral_deform import ComplexCoordinate
from package_common.utils_name import create_function_name_logger

type Func4Quad = Callable[[int, float | int | complex], complex | float]


class ChebyshevGaussQuad:
    """Class to perform the Chebyshev-Gauss quadrature."""

    __num_degree: int
    __num_point: int
    __spectral_deform: bool
    __point_array: ArrayFloat
    __point_analytic_cont: ArrayComplex
    __jacobian: ArrayFloat

    __cache_dict_array: dict[Func4Quad, ArrayComplex | ArrayFloat] = {}

    __flag: bool = False
    __logger: DefaultLogger = DefaultLogger(__name__)

    @classmethod
    def set_class_variable(cls,
                           num_degree: int,
                           *,
                           y_complex: ComplexCoordinate,
                           y_unuse_spectral_deform: ComplexCoordinate) -> None:
        """Set the class variables.

        Parameters
        ----------
        num_degree : int
            The number of the degree.
        y_complex : ComplexCoordinate
            The complex coordinate for spectral deformation.
        y_unuse_spectral_deform : ComplexCoordinate
            The complex coordinate without spectral deformation.
        """

        if cls.__flag:
            return

        cls.__cache_dict_array.clear()

        cls.__num_degree = num_degree
        cls.__num_point = 3 * cls.__num_degree
        cls.__spectral_deform = y_complex.use_spectral_deform
        cls.__flag = True

        cls.__point_array = np.array(
            [calc_collocation_point(2*i_l-1, 2*cls.__num_point)
             for i_l in range(1, cls.__num_point+1)], dtype=np.float64
        )

        if cls.__spectral_deform:
            y_pos: complex
            guess: complex
            cls.__point_analytic_cont \
                = np.empty(cls.__num_point, dtype=np.complex128)
            for i_pos, pos in enumerate(cls.__point_array):
                y_pos = y_unuse_spectral_deform.r_value(pos)
                if i_pos == 0:
                    guess = pos + 1j * 0
                else:
                    guess = cls.__point_analytic_cont[i_pos-1]
                cls.__point_analytic_cont[i_pos] \
                    = y_complex.inverse(y_pos, guess=guess)

        cls.__jacobian = np.array(
            [y_unuse_spectral_deform.r_value_d(pos)
             for pos in cls.__point_array],
            dtype=np.float64
        )

    @classmethod
    def __create_array_func(cls,
                            func: Func4Quad) -> ArrayComplex | ArrayFloat:
        """Create an array of the function values at all collocation points.

        Parameters
        ----------
        func : Func4Quad
            The function to be evaluated.

        Returns
        -------
        ArrayComplex | ArrayFloat
            The array of the function values at all collocation points.
        """

        cached_array: ArrayComplex | ArrayFloat | None \
            = cls.__cache_dict_array.get(func, None)
        if cached_array is not None:
            return cached_array

        array_func: ArrayComplex | ArrayFloat
        point_array: ArrayComplex | ArrayFloat
        if not cls.__spectral_deform:
            array_func = np.empty(
                (cls.__num_degree, cls.__num_point), dtype=np.float64)
            point_array = cls.__point_array
        else:
            array_func = np.empty(
                (cls.__num_degree, cls.__num_point), dtype=np.complex128)
            point_array = cls.__point_analytic_cont

        list_point_array: list[complex | float] = point_array.tolist()
        for i_n in range(cls.__num_degree):
            array_func[i_n, :] = [
                func(i_n, s_pos) for s_pos in list_point_array]

        cls.__cache_dict_array[func] = array_func
        return array_func

    def __init__(self,
                 *,
                 func_1: Func4Quad,
                 func_2: Func4Quad | None = None,
                 weight: FuncFloat = lambda x: 1.0) -> None:
        """Initialize an instance of the ChebyshevGaussQuad class.

        Parameters
        ----------
        func_1 : Func4Quad
            The function associated with the first vector.
        func_2 : Func4Quad | None, optional, default None
            The function associated with the second vector.
        weight : FuncFloat, optional, default lambda x: 1.0
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

        self.__num_point: int = ChebyshevGaussQuad.__num_point
        point_array: ArrayFloat = ChebyshevGaussQuad.__point_array

        self.__flag_func_2: bool = func_2 is not None

        self.__array_weight: ArrayFloat = (
            np.array([weight(pos) for pos in point_array], dtype=np.float64)
            * np.sqrt(1.0 - point_array**2) * ChebyshevGaussQuad.__jacobian
        )

        self.__array_func_1: ArrayComplex | ArrayFloat \
            = ChebyshevGaussQuad.__create_array_func(func_1)
        if func_2 is not None:
            self.__array_func_2: ArrayComplex | ArrayFloat \
                = ChebyshevGaussQuad.__create_array_func(func_2)

    def quadrature(self,
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
        vec_2 : ArrayComplex | None, optional, default None
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
        integral: ArrayComplex
        if self.__flag_func_2 and (vec_2 is not None):
            field_2: ArrayComplex = vec_2.T @ self.__array_func_2
            integral = (np.conj(field_1) * field_2) @ self.__array_weight
        else:
            integral = cast(ArrayComplex,
                            np.conj(field_1) @ self.__array_weight)

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
        s_pos: complex | float,
        mu_complex: ComplexCoordinate) -> complex | float:
    """Calculate the spherical horizontal Laplacian of the Heinrichs
    basis at a given point.

    Parameters
    ----------
    m_order : int
        The zonal wavenumber (order).
    n_degree : int
        The degree of the Heinrichs basis.
    s_pos : complex | float
        The position of the point.
    mu_complex : ComplexCoordinate
        The complex coordinate for spectral deformation.

    Returns
    -------
    complex | float
        The value of the spherical horizontal Laplacian of the Heinrichs basis
        at the point.
    """

    mu: complex | float
    mu_d: complex | float
    mu_d2: complex | float
    if mu_complex.use_spectral_deform:
        mu = mu_complex.value(s_pos)
        mu_d = mu_complex.value_d(s_pos)
        mu_d2 = mu_complex.value_d2(s_pos)
    else:
        s_pos_real: float = s_pos.real
        mu = mu_complex.r_value(s_pos_real)
        mu_d = mu_complex.r_value_d(s_pos_real)
        mu_d2 = mu_complex.r_value_d2(s_pos_real)

    sin_sq: complex | float = 1 - (mu**2)

    heinrichs: complex | float
    heinrichs_d: complex | float
    heinrichs_d2: complex | float
    heinrichs, heinrichs_d, heinrichs_d2 = calc_heinrichs(n_degree, s_pos, 2)

    return (
        sin_sq * heinrichs_d2 / (mu_d**2)
        - (2*mu/mu_d + sin_sq*mu_d2/(mu_d**3))
        * heinrichs_d - (m_order**2) * heinrichs / sin_sq
    )
