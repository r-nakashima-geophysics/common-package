"""A Python module for the spectral deformation method.

References
----------
[1] John D. Crawford and Peter D. Hislop, Application of the method of spectral
deformation to the Vlasov-Poisson system. Annals of Physics 189, 265-317
(1989). doi: 10.1016/0003-4916(89)90166-8

[2] John P. Boyd, Chebyshev and Fourier Spectral Methods. Courier Corporation,
(2001).
"""

import sys

import numpy as np
from scipy import optimize

from package_common.background_field import BackgroundField
from package_common.common_types import ArrayFloat, ComplexFunc, Self
from package_common.default_logger import DefaultLogger
from package_common.utils_name import create_function_name_logger


class ComplexCoordinate(BackgroundField):
    """Subclass of the BackgroundField class to define the complex
    coordinate transformation.

    Attributes
    ----------
    name : str
        The name of the complex coordinate transformation.
    value : ComplexFunc
        The profile of the complex coordinate transformation.
    value_d : ComplexFunc | None
        The first derivative of the profile of the complex coordinate
        transformation.
    value_d2 : ComplexFunc | None
        The second derivative of the profile of the complex coordinate
        transformation.
    tex : str | None
        The LaTeX text of the complex coordinate transformation.
    params : dict[str, float]
        The parameters for the complex coordinate transformation.
    """

    def __init__(self: Self,
                 name: str,
                 *,
                 value: ComplexFunc,
                 value_d: ComplexFunc | None = None,
                 value_d2: ComplexFunc | None = None,
                 tex: str | None = None,
                 params: dict[str, float]) -> None:
        """Initialize an instance of the ComplexCoordinate class.

        Parameters
        ----------
        name : str
            The name of the complex coordinate transformation.
        value : ComplexFunc
            The profile of the complex coordinate transformation.
        value_d : ComplexFunc | None, optional, default None
            The first derivative of the profile of the complex coordinate
            transformation.
        value_d2 : ComplexFunc | None, optional, default None
            The second derivative of the profile of the complex coordinate
            transformation.
        tex : str | None, optional, default None
            The LaTeX text of the complex coordinate transformation.
        params : dict[str, float]
            The parameters for the complex coordinate transformation.
        """

        self.params: dict[str, float] = params

        super().__init__(name,
                         value=value,
                         value_d=value_d,
                         value_d2=value_d2,
                         tex=tex)

        self.__logger: DefaultLogger = DefaultLogger(self.name)

    def inverse(self: Self,
                y_pos: complex) -> complex:
        """Solve y = y(s) for s numerically.

        Parameters
        ----------
        y_pos : complex
            The target value in the complex coordinate.

        Returns
        -------
        complex
            A numerical solution of y = y(s) for s.

        Warnings
        --------
        Not converge
            If no solution converges to the target value.
        """

        def _residual(s_vec: ArrayFloat) -> ArrayFloat:
            s_pos: complex = s_vec[0] + 1j*s_vec[1]
            residual: complex = self.value(s_pos) - y_pos
            return np.array([residual.real, residual.imag], dtype=np.float64)

        def _jacobian(s_vec: ArrayFloat) -> ArrayFloat:
            s_pos: complex = s_vec[0] + 1j*s_vec[1]
            dy: complex = self.value_d(s_pos)
            return np.array([[np.real(dy), np.real(dy*1j)],
                             [np.imag(dy), np.imag(dy*1j)]], dtype=np.float64)

        init_guess: ArrayFloat \
            = np.array([y_pos.real, y_pos.imag], dtype=np.float64)
        sol = optimize.root(_residual, init_guess, jac=_jacobian)

        if (not sol.success) and (not np.allclose(sol.fun, [0, 0])):
            self.__logger.warning(f'Not converge at y = {y_pos}')

        return sol.x[0] + 1j*sol.x[1]

    def check_spectral_deform(self) -> bool:
        """Check whether the spectral deformation method is used or not.

        Returns
        -------
        check : bool
            The boolean value to check whether the spectral deformation method
            is used or not.
        """

        return any((not np.isclose(value, 0))
                   for value in self.params.values())


def init_complex_coordinate_simple(
        y_start: float,
        y_end: float,
        *,
        alpha: float,
        beta_0: float,
        beta_1: float) -> ComplexCoordinate:
    """Construct an instance of the ComplexCoordinate class for the
    simple complex coordinate transformation, e.g. y = s + i(1-s^2), in the
    spectral deformation method.

    Parameters
    ----------
    y_start : float
        The starting point.
    y_end : float
        The ending point.
    alpha : float
        A parameter for the complex coordinate transformation.
    beta_0 : float
        A parameter for the complex coordinate transformation.
    beta_1 : float
        A parameter for the complex coordinate transformation.

    Returns
    -------
    ComplexCoordinate
        The instance of the ComplexCoordinate class for the transformation to
        complex coordinates.

    Warnings
    --------
    Invalid argument
        If `y_start` and `y_end` are equal.
    """

    logger: DefaultLogger = create_function_name_logger()

    if y_start == y_end:
        logger.error('Invalid argument')
        sys.exit(1)

    name: str = f'[simple_a={alpha}_b0={beta_0}_b1={beta_1}]'

    params: dict[str, float] = {
        "alpha": alpha,
        "beta_0": beta_0,
        "beta_1": beta_1
    }

    def y_complex(s_pos: complex) -> complex:
        return (
            y_start + (y_end-y_start)*(s_pos+1)/2
            - (alpha+1j) * (beta_0+beta_1*s_pos) * ((s_pos**2)-1)
        )

    def y_complex_d(s_pos: complex) -> complex:
        return (
            (y_end-y_start) / 2
            - (alpha+1j) * (beta_1*(3*(s_pos**2)-1)+2*beta_0*s_pos)
        )

    def y_complex_d2(s_pos: complex) -> complex:
        return (
            - 2 * (alpha+1j) * (3*beta_1*s_pos+beta_0)
        )

    return ComplexCoordinate(name,
                             value=y_complex,
                             value_d=y_complex_d,
                             value_d2=y_complex_d2,
                             params=params)
