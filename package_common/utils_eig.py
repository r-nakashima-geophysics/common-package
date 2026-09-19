"""A Python module to provide the utilities for eigenvalue problems."""

import numpy as np

from package_common.common_types import (ArrayAny, ArrayBool, ArrayComplex,
                                         ArrayInt, NoReturn)
from package_common.default_logger import DefaultLogger
from package_common.utils_name import create_function_name_logger


def sort_eig(eigenvalues: ArrayComplex,
             eigenvectors: ArrayComplex) -> ArrayComplex:
    """Sort the eigenvalues and eigenvectors.

    Parameters
    ----------
    eigenvalues : ArrayComplex
        The eigenvalues.
    eigenvectors : ArrayComplex
        The eigenvectors.

    Returns
    -------
    matrix_eig : ArrayComplex
        The matrix storing the eigenvalues and eigenvectors.

    Examples
    --------
    >>> import numpy as np
    >>> from package_common.utils_eig import sort_eig
    >>> matrix = np.array([[1, 2], [3, 4]])
    >>> eigenvalues, eigenvectors = np.linalg.eig(matrix)
    >>> sort_eig(eigenvalues, eigenvectors)
    array([[-0.82456484+0.j, -0.41597356+0.j],
           [ 0.56576746+0.j, -0.90937671+0.j],
           [-0.37228132+0.j,  5.37228132+0.j]])
    """

    size_matrix: int = len(eigenvalues)

    idx: ArrayInt = np.argsort(eigenvalues.real, kind='stable')

    matrix_eig: ArrayComplex = np.empty(
        (size_matrix+1, size_matrix), dtype=np.complex128)
    matrix_eig[:size_matrix, :] = eigenvectors[:, idx]
    matrix_eig[size_matrix, :] = eigenvalues[idx]

    return matrix_eig


def screening_eig(matrix_eig: ArrayComplex,
                  check: ArrayBool,
                  *phys_qtys: ArrayAny) \
    -> tuple[ArrayComplex,
             tuple[ArrayAny, ...]] | NoReturn:
    """Exclude invalid eigenmodes.

    Parameters
    ----------
    matrix_eig : ArrayComplex
        The matrix storing the eigenvalues and eigenvectors.
    check : ArrayBool
        The validity of the eigenmodes.
    *phys_qtys : ArrayAny
        The physical quantities.

    Returns
    -------
    matrix_eig : ArrayComplex
        The matrix storing the eigenvalues and eigenvectors. This is masked by
        `check`.
    phys_qtys : tuple[ArrayAny, ...]
        The physical quantities. This is masked by `check`.

    Warnings
    --------
    Invalid shape of the input arrays
        If the shapes of the input arrays do not match.

    Examples
    --------
    >>> import numpy as np
    >>> from package_common.utils_eig import sort_eig, screening_eig
    >>> matrix = np.array([[1, 2], [3, 4]])
    >>> check = np.array([True, False])
    >>> phys_qtys = np.array([10.0, 20.0])
    >>> eigenvalues, eigenvectors = np.linalg.eig(matrix)
    >>> matrix_eig = sort_eig(eigenvalues, eigenvectors)
    >>> screening_eig(matrix_eig, check, phys_qtys)
    (array([[-0.82456484+0.j,         nan+0.j],
           [ 0.56576746+0.j,         nan+0.j],
           [-0.37228132+0.j,         nan+0.j]]), (array([10., nan]),))

    Notes
    -----
    Arrays of type `ArrayInt` are not supported in `phys_qtys`.
    """

    logger: DefaultLogger

    size_matrix: int = matrix_eig.shape[1]

    if (matrix_eig.shape[0] != size_matrix + 1) \
            or (len(check.ravel()) != size_matrix):
        logger = create_function_name_logger()
        logger.error('Invalid shape of the input arrays')

    for phys_qty in phys_qtys:
        if len(phys_qty.ravel()) != size_matrix:
            logger = create_function_name_logger()
            logger.error('Invalid shape of the input arrays')

    invalid: ArrayBool = ~check.ravel()
    matrix_eig[:, invalid] = np.nan
    for phys_qty in phys_qtys:
        phys_qty[invalid] = np.nan

    return matrix_eig, phys_qtys
