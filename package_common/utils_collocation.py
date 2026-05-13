"""A Python module to provide the utilities for collocation methods."""


import sys

import numpy as np

from package_common.default_logger import DefaultLogger
from package_common.utils_name import create_function_name_logger


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

    logger: DefaultLogger = create_function_name_logger()
    if 0 <= i_l <= num_point:
        return -np.cos(i_l*np.pi/num_point)

    logger.error('Invalid argument')
    sys.exit(1)
