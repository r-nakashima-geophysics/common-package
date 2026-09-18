"""A Python module to provide the utilities for time-dependent simulations."""


import numpy as np

from package_common.common_types import ArrayFloat, Callable
from package_common.default_logger import DefaultLogger
from package_common.utils_debug import under_construction_log


class Field:
    """Class to define a field for time-dependent simulations.

    Attributes
    ----------
    name : str
        The name of the field.
    time : float
        The time.
    value : ArrayFloat
        The value of the field.

    Examples
    --------
    >>> from package_common.utils_simulation import Field
    >>> Field.set_class_variable(3, 2)
    >>> field = Field('example')
    >>> field.value
    array([[0., 0.],
        [0., 0.],
        [0., 0.]])
    """

    __num_dim1: int
    __num_dim2: int | None
    __num_dim3: int | None

    __flag: bool = False
    __logger: DefaultLogger = DefaultLogger(__name__)

    @classmethod
    def set_class_variable(cls,
                           num_dim1: int,
                           num_dim2: int | None = None,
                           num_dim3: int | None = None) -> None:
        """Set the class variables.

        Parameters
        ----------
        num_dim1 : int
            The number of elements of the first dimension.
        num_dim2 : int | None, optional, default None.
            The number of elements of the second dimension.
        num_dim3 : int | None, optional, default None.
            The number of elements of the third dimension.

        Warnings
        --------
        `set_class_variable` class method has already been executed
            If `set_class_variable` class method is called after it has already
            been executed.
        """

        if cls.__flag:
            cls.__logger.warning(
                '`set_class_variable` class method has already been executed')
            return

        cls.__num_dim1 = num_dim1
        cls.__num_dim2 = num_dim2
        cls.__num_dim3 = num_dim3
        cls.__flag = True

    def __init__(self,
                 name: str,
                 time: float = 0) -> None:
        """Initialize an instance of the Field class.

        Parameters
        ----------
        name : str
            The name of the field.
        time : float, optional, default 0
            The time.

        Warnings
        --------
        `set_class_variable` class method has not been executed yet
            If `set_class_variable` class method has not been executed yet.
        """

        if not Field.__flag:
            Field.__logger.error(
                '`set_class_variable` class method has not been executed yet')

        self.name: str = name
        self.time: float = time

        self.value: ArrayFloat
        if (Field.__num_dim2 is None) and (Field.__num_dim3 is None):
            self.value = np.zeros(Field.__num_dim1, dtype=np.float64)
        elif (Field.__num_dim2 is not None) and (Field.__num_dim3 is None):
            self.value = np.zeros(
                (Field.__num_dim1, Field.__num_dim2), dtype=np.float64)
        elif (Field.__num_dim2 is not None) and (Field.__num_dim3 is not None):
            self.value = np.zeros(
                (Field.__num_dim1, Field.__num_dim2, Field.__num_dim3),
                dtype=np.float64)
        else:
            Field.__logger.warning('`num_dim3` is interpreted as `num_dim2`')
            Field.__num_dim2 = Field.__num_dim3
            Field.__num_dim3 = None
            self.value = np.zeros(
                (Field.__num_dim1, Field.__num_dim2), dtype=np.float64)

    def value_copy(self,
                   another: Field) -> None:
        """Copy the value and time from another field.

        Parameters
        ----------
        another : Field
            The field.
        """

        self.value = np.copy(another.value)
        self.time = another.time


def time_integrate(field: Field,
                   dt: float,
                   rhs: Callable[[Field, float], Field],
                   *,
                   method: str = 'RK4') -> Field:
    """Time integrate the evolution of a field.

    Parameters
    ----------
    field : Field
        The field before time integration.
    dt : float
        The time step.
    rhs : Callable[[Field, float], Field]
        The right-hand sides of the differential equations.
    method : str, optional, default 'RK4'
        The integration method.

    Returns
    -------
    Field
        The field after time integration.
    """

    tmp_field: Field = Field(name='temporary')

    if method == 'RK4':
        tmp_field.value_copy(field)
        k1: ArrayFloat = rhs(tmp_field, tmp_field.time).value * dt

        tmp_field.time += 0.5 * dt
        tmp_field.value = field.value + 0.5 * k1
        k2: ArrayFloat = rhs(tmp_field, tmp_field.time).value * dt

        tmp_field.value = field.value + 0.5 * k2
        k3: ArrayFloat = rhs(tmp_field, tmp_field.time).value * dt

        tmp_field.time += 0.5 * dt
        tmp_field.value = field.value + k3
        k4: ArrayFloat = rhs(tmp_field, tmp_field.time).value * dt

        field.value += (k1 + 2*k2 + 2*k3 + k4) / 6
        field.time += dt

    else:
        under_construction_log()

    return field
