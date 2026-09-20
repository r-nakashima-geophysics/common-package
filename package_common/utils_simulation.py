"""A Python module to provide the utilities for time-dependent simulations."""

import numpy as np

from package_common.common_types import Any, ArrayComplex, ArrayFloat, Callable
from package_common.default_logger import DefaultLogger
from package_common.utils_debug import under_construction_log
from package_common.utils_name import create_function_name_logger

type Rhs = Callable[
    [ArrayComplex | ArrayFloat | list[ArrayComplex | ArrayFloat], float, ...],
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
        The temporary storage arrays.
    num_tmp : int
        The number of temporary storage arrays.

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

        if (cls.__num_dim2 is None) and (cls.__num_dim3 is not None):
            cls.__logger.warning('`num_dim3` is interpreted as `num_dim2`')
            cls.__num_dim2 = cls.__num_dim3
            cls.__num_dim3 = None

    def __init__(self,
                 name: str,
                 time: float = 0,
                 *,
                 num_tmp: int = 3,
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
        num_tmp : int, optional, default 3
            The number of temporary storage arrays.

        Warnings
        --------
        `set_class_variable` class method has not been executed yet
            If `set_class_variable` class method has not been executed yet.
        Invalid argument
            If `num_tmp` is negative.
        """

        if not Field.__flag:
            Field.__logger.error(
                '`set_class_variable` class method has not been executed yet')

        if num_tmp < 0:
            DefaultLogger(name).__logger.error('Invalid argument')

        self.name: str = name
        self.time: float = time
        self.num_tmp: int = num_tmp

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
            = [np.zeros_like(self.value) for _ in range(self.num_tmp)]

    def copy_to_tmp(self,
                    *,
                    num_copy: int | None = None) -> None:
        """Copy the values to the temporary storage arrays.

        Parameters
        ----------
        num_copy : int | None, optional, default None
            The number of temporary storage arrays to which the values are
            copied.

        Notes
        -----
        If `num_copy` is None, the values are copied to all temporary storage
        arrays.
        """

        if num_copy is None:
            num_copy = self.num_tmp

        for i in range(num_copy):
            np.copyto(self.value_tmp[i], self.value)


def time_integrate(fields: Field | list[Field],
                   dt: float,
                   rhss: Rhs | list[Rhs],
                   *,
                   method: str = 'RK4',
                   args: tuple[Any, ...] = ()) -> Field | list[Field]:
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
    args : tuple[Any, ...], optional, default ()
        Additional arguments.

    Returns
    -------
    Field | list[Field]
        The field(s) after time integration.

    Warnings
    --------
    Mismatch length between `fields` and `rhss`
        If the lengths of `fields` and `rhss` mismatch.
    Lack of sufficient temporary storage arrays
        If the number of temporary storage arrays is less than the required
        number for the chosen method (e.g., 3 for RK4).

    Examples
    --------
    >>> import numpy as np
    >>> from package_common.utils_simulation import Field
    >>> from package_common.utils_simulation import time_integrate
    >>> Field.set_class_variable(2)
    >>> field = Field(name='example')
    >>> field.value = np.array([1.0, 1.0])
    >>> rhs = lambda x, t: np.array([[0, 1], [-1, 0]]) @ x
    >>> for _ in range(10):
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

    Notes
    -----
    The first argument of `rhs` must be the list of `field.value` (or a single
    `field.value`), the second argument must be the current time, and the
    remaining arguments are optional additional arguments.
    """

    is_list: bool = isinstance(fields, (list, tuple))

    list_field: list[Field]
    if isinstance(fields, (list, tuple)):
        list_field = fields
    else:
        list_field = [fields]
    if isinstance(rhss, (list, tuple)):
        list_rhs = rhss
    else:
        list_rhs = [rhss]

    logger: DefaultLogger
    if len(list_field) != len(list_rhs):
        logger = create_function_name_logger()
        logger.error('Mismatch length between `fields` and `rhss`')

    if method == 'RK4':

        for field in list_field:
            if field.num_tmp < 3:
                logger = create_function_name_logger()
                logger.error('Lack of sufficient temporary storage arrays')
            field.copy_to_tmp(num_copy=3)

        k: ArrayComplex | ArrayFloat
        values: ArrayComplex | ArrayFloat | list[ArrayComplex | ArrayFloat] \
            = [field.value_tmp[2] for field in list_field] if is_list \
            else list_field[0].value_tmp[2]
        time: float = list_field[0].time
        for field, rhs in zip(list_field, list_rhs):
            k = rhs(values, time, *args) * dt
            field.value_tmp[0] += 0.5 * k
            field.value += k / 6

        values = [field.value_tmp[0] for field in list_field] if is_list \
            else list_field[0].value_tmp[0]
        time += 0.5 * dt
        for field, rhs in zip(list_field, list_rhs):
            k = rhs(values, time, *args) * dt
            field.value_tmp[1] += 0.5 * k
            field.value += k / 3

        values = [field.value_tmp[1] for field in list_field] if is_list \
            else list_field[0].value_tmp[1]
        for field, rhs in zip(list_field, list_rhs):
            k = rhs(values, time, *args) * dt
            field.value_tmp[2] += k
            field.value += k / 3

        values = [field.value_tmp[2] for field in list_field] if is_list \
            else list_field[0].value_tmp[2]
        time += 0.5 * dt
        for field, rhs in zip(list_field, list_rhs):
            k = rhs(values, time, *args) * dt
            field.value += k / 6

        for field in list_field:
            field.time = time

    else:
        under_construction_log()

    return list_field if is_list else list_field[0]
