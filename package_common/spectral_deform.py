"""A Python module for the spectral deformation method.

References
----------
[1] John D. Crawford and Peter D. Hislop, Application of the method of spectral
deformation to the Vlasov-Poisson system. Annals of Physics 189, 265-317
(1989). doi: 10.1016/0003-4916(89)90166-8

[2] John P. Boyd, Chebyshev and Fourier Spectral Methods. Courier Corporation,
(2001).
"""

import numpy as np
from scipy import optimize

from package_common.background_field import BackgroundField
from package_common.common_types import FuncComplex
from package_common.default_logger import DefaultLogger
from package_common.utils_name import create_function_name_logger

type OptimizeResult = optimize.OptimizeResult


class ComplexCoordinate(BackgroundField):
    """Subclass of the BackgroundField class to define the complex
    coordinate transformation.

    Attributes
    ----------
    name : str
        The name of the complex coordinate transformation.
    value : FuncComplex
        The profile of the complex coordinate transformation.
    value_d : FuncComplex | None
        The first derivative of the profile of the complex coordinate
        transformation.
    value_d2 : FuncComplex | None
        The second derivative of the profile of the complex coordinate
        transformation.
    tex : str | None
        The LaTeX text of the complex coordinate transformation.
    params : dict[str, float]
        The parameters for the complex coordinate transformation.
    use_spectral_deform : bool
        The boolean value to check whether the spectral deformation method is
        used or not.
    """

    def __init__(self,
                 name: str,
                 *,
                 value: FuncComplex,
                 value_d: FuncComplex,
                 value_d2: FuncComplex | None = None,
                 tex: str | None = None,
                 params: dict[str, float]) -> None:
        """Initialize an instance of the ComplexCoordinate class.

        Parameters
        ----------
        name : str
            The name of the complex coordinate transformation.
        value : FuncComplex
            The profile of the complex coordinate transformation.
        value_d : FuncComplex
            The first derivative of the profile of the complex coordinate
            transformation.
        value_d2 : FuncComplex | None, optional, default None
            The second derivative of the profile of the complex coordinate
            transformation.
        tex : str | None, optional, default None
            The LaTeX text of the complex coordinate transformation.
        params : dict[str, float]
            The parameters for the complex coordinate transformation.
        """

        self.params: dict[str, float] = params
        self.use_spectral_deform: bool = self.__check_spectral_deform()

        super().__init__(name,
                         value=value,
                         value_d=value_d,
                         value_d2=value_d2,
                         tex=tex)

        self.__logger: DefaultLogger = DefaultLogger(self.name)

    def inverse(self,
                y_pos: complex,
                *,
                guess: complex | None = None) -> complex:
        """Solve y = y(s) for s numerically.

        Parameters
        ----------
        y_pos : complex
            The target value in the complex coordinate.
        guess : complex | None, optional, default None
            The initial guess of s.

        Returns
        -------
        complex
            A numerical solution of y = y(s) for s.

        Warnings
        --------
        Did not converge
            If no solution converges to the target value.
        """

        def _residual(s: complex) -> complex:
            residual: complex = self.value(s) - y_pos
            return residual

        init_guess: complex = guess if guess is not None else y_pos

        root: complex
        result: optimize.RootResults
        root, result = optimize.newton(
            _residual, init_guess,
            fprime=self.value_d, fprime2=self.value_d2,
            full_output=True, disp=False
        )

        if (not result.converged) \
                or (not np.isclose(self.value(root)-y_pos, 0)):
            self.__logger.warning(f'Did not converge at y = {y_pos}')

        return root

    def __check_spectral_deform(self) -> bool:
        """Check whether the spectral deformation method is used or not.

        Returns
        -------
        bool
            The boolean value to check whether the spectral deformation method
            is used or not.
        """

        return any((not np.isclose(value, 0))
                   for value in self.params.values())


def init_complex_coordinate_simple(
        y_start: float,
        y_end: float,
        *,
        alpha: float = 0,
        beta_0: float = 0,
        beta_1: float = 0) -> ComplexCoordinate:
    """Construct an instance of the ComplexCoordinate class for the
    simple complex coordinate transformation, e.g. y = s + i(1-s^2), in the
    spectral deformation method.

    Parameters
    ----------
    y_start : float
        The starting point.
    y_end : float
        The ending point.
    alpha : float, optional, default 0
        A parameter for the complex coordinate transformation.
    beta_0 : float, optional, default 0
        A parameter for the complex coordinate transformation.
    beta_1 : float, optional, default 0
        A parameter for the complex coordinate transformation.

    Returns
    -------
    ComplexCoordinate
        The instance of the ComplexCoordinate class for the transformation to
        complex coordinates.

    Warnings
    --------
    Invalid argument
        If `y_start` and `y_end` are equal, or if `alpha` is not zero when both
        `beta_0` and `beta_1` are zero.
    """

    logger: DefaultLogger = create_function_name_logger()

    if np.isclose(y_start, y_end):
        logger.error('Invalid argument')

    if (not np.isclose(alpha, 0)) \
            and np.isclose(beta_0, 0) and np.isclose(beta_1, 0):
        logger.error('Invalid argument')

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

    return ComplexCoordinate(
        name,
        value=y_complex,
        value_d=y_complex_d,
        value_d2=y_complex_d2,
        params=params)
