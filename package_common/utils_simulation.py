"""A Python module to provide the utilities for time-dependent simulations."""

import numpy as np

from package_common.common_types import (ArrayComplex, ArrayFloat, Callable,
                                         NoReturn)
from package_common.default_logger import DefaultLogger
from package_common.utils_debug import under_construction_log
from package_common.utils_name import create_function_name_logger

type Rhs = Callable[[ArrayComplex | ArrayFloat |
                     list[ArrayComplex | ArrayFloat], float],
                    ArrayComplex | ArrayFloat]


class Field:
    """Class to define a field for time-dependent simulations.

    Attributes
    ----------
    name : str
        The name of the field.
    time : float
        The time.
    value : ArrayComplex | ArrayFloat
        The value of the field.
    value_tmp : list[ArrayComplex | ArrayFloat]
        The temporary storages for the value of the field.

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
    __num_tmp: int

    __flag: bool = False
    __logger: DefaultLogger = DefaultLogger(__name__)

    @classmethod
    def set_class_variable(cls,
                           num_dim1: int,
                           num_dim2: int | None = None,
                           num_dim3: int | None = None,
                           *,
                           num_tmp: int = 3) -> None | NoReturn:
        """Set the class variables.

        Parameters
        ----------
        num_dim1 : int
            The number of elements of the first dimension.
        num_dim2 : int | None, optional, default None.
            The number of elements of the second dimension.
        num_dim3 : int | None, optional, default None.
            The number of elements of the third dimension.
        num_tmp : int, optional, default 3
            The number of temporary storages for the value of the field.

        Warnings
        --------
        `set_class_variable` class method has already been executed
            If `set_class_variable` class method is called after it has already
            been executed.
        Invalid argument
            If `num_tmp` is negative.
        """

        if cls.__flag:
            cls.__logger.warning(
                '`set_class_variable` class method has already been executed')
            return

        if num_tmp < 0:
            cls.__logger.error('Invalid argument')

        cls.__num_dim1 = num_dim1
        cls.__num_dim2 = num_dim2
        cls.__num_dim3 = num_dim3
        cls.__num_tmp = num_tmp
        cls.__flag = True

        if (cls.__num_dim2 is None) and (cls.__num_dim3 is not None):
            cls.__logger.warning('`num_dim3` is interpreted as `num_dim2`')
            cls.__num_dim2 = cls.__num_dim3
            cls.__num_dim3 = None

    @classmethod
    def num_tmp(cls) -> int:
        """Return the number of temporary storages."""

        return cls.__num_tmp

    def __init__(self,
                 name: str,
                 time: float = 0,
                 *,
                 dtype: type = np.float64) -> None:
        """Initialize an instance of the Field class.

        Parameters
        ----------
        name : str
            The name of the field.
        time : float, optional, default 0
            The time.
        dtype : type, optional, default np.float64
            The data type of the `value` array.

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

        self.value: ArrayComplex | ArrayFloat
        if Field.__num_dim2 is None:
            self.value = np.zeros(Field.__num_dim1, dtype=dtype)
        elif Field.__num_dim3 is None:
            self.value = np.zeros(
                (Field.__num_dim1, Field.__num_dim2), dtype=dtype)
        else:
            self.value = np.zeros(
                (Field.__num_dim1, Field.__num_dim2, Field.__num_dim3),
                dtype=dtype)

        self.value_tmp: list[ArrayComplex | ArrayFloat] \
            = [np.zeros_like(self.value) for _ in range(Field.__num_tmp)]

    def copy_to_tmp(self) -> None:
        """Copy the value to the temporary storage."""

        for i in range(Field.__num_tmp):
            np.copyto(self.value_tmp[i], self.value)


def time_integrate(fields: Field | list[Field],
                   dt: float,
                   rhss: Rhs | list[Rhs],
                   *,
                   method: str = 'RK4') -> Field | list[Field]:
    """Time integrate the evolution of fields.

    Parameters
    ----------
    fields : Field | list[Field]
        The field(s) before time integration.
    dt : float
        The time step.
    rhss : Rhs | list[Rhs]
        The right-hand side(s) of the differential equation(s).
    method : str, optional, default 'RK4'
        The integration method.

    Returns
    -------
    Field | list[Field]
        The field(s) after time integration.

    Warnings
    --------
    Mismatch length between `fields` and `rhss`
        If the lengths of `fields` and `rhss` mismatch.
    Lack of sufficient temporary storage
        If the number of temporary storages is less than a required number for
        the chosen method (e.g., 3 for RK4).

    Examples
    --------
    >>> import numpy as np
    >>> from package_common.utils_simulation import Field
    >>> from package_common.utils_simulation import time_integrate
    >>> Field.set_class_variable(2)
    >>> field = Field(name='example')
    >>> field.value = np.array([1.0, 1.0])
    >>> rhs = lambda x, t: np.array([[0, 1], [-1, 0]]) @ x
    >>> for _  in range(10):
    ...     field = time_integrate(field, 0.01, rhs)
    ...     print(field.value)
    [1.00994983 0.98995017]
    [1.01979867 0.97980134]
    [1.02954553 0.96955453]
    [1.03918944 0.95921077]
    [1.04872943 0.94877109]
    [1.05816455 0.93823653]
    [1.06749385 0.92760815]
    [1.0767164  0.91688701]
    [1.08583128 0.90607418]
    [1.09483758 0.89517075]
    """

    is_list: bool = isinstance(fields, (list, tuple))

    if not is_list:
        fields = [fields]
    if not isinstance(rhss, (list, tuple)):
        rhss = [rhss]
    if len(fields) != len(rhss):
        logger: DefaultLogger = create_function_name_logger()
        logger.error('Mismatch length between `fields` and `rhss`')

    if method == 'RK4':

        if Field.num_tmp() < 3:
            logger: DefaultLogger = create_function_name_logger()
            logger.error('Lack of sufficient temporary storage')

        for field in fields:
            field.copy_to_tmp()

        k: ArrayComplex | ArrayFloat
        values: ArrayComplex | ArrayFloat | list[ArrayComplex | ArrayFloat] \
            = [field.value_tmp[2] for field in fields] if is_list \
            else fields[0].value_tmp[2]
        time: float = fields[0].time
        for field, rhs in zip(fields, rhss):
            k = rhs(values, time) * dt
            field.value_tmp[0] += 0.5 * k
            field.value += k / 6

        values = [field.value_tmp[0] for field in fields] if is_list \
            else fields[0].value_tmp[0]
        time += 0.5 * dt
        for field, rhs in zip(fields, rhss):
            k = rhs(values, time) * dt
            field.value_tmp[1] += 0.5 * k
            field.value += k / 3

        values = [field.value_tmp[1] for field in fields] if is_list \
            else fields[0].value_tmp[1]
        for field, rhs in zip(fields, rhss):
            k = rhs(values, time) * dt
            field.value_tmp[2] += k
            field.value += k / 3

        values = [field.value_tmp[2] for field in fields] if is_list \
            else fields[0].value_tmp[2]
        time += 0.5 * dt
        for field, rhs in zip(fields, rhss):
            k = rhs(values, time) * dt
            field.value += k / 6

        for field in fields:
            field.time = time

    else:
        under_construction_log()

    return fields if is_list else fields[0]
