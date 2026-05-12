"""A Python module to provide the utilities for eigenvalue problems."""

import sys

import numpy as np

from package_common.common_types import (ArrayAny, ArrayBool, ArrayComplex,
                                         ArrayInt)
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
    >>> from package_common.utils_eig import sort_eig
    >>> eigenvalues, eigenvectors = np.linalg.eig(matrix)
    >>> matrix_eig = sort_eig(eigenvalues, eigenvectors)
    """

    size_matrix: int = len(eigenvalues)

    idx: ArrayInt = np.argsort(eigenvalues.real)
    eigenvalues_sorted: ArrayComplex = eigenvalues[idx]
    eigenvectors_sorted: ArrayComplex = eigenvectors[:, idx]

    matrix_eig: ArrayComplex = np.empty(
        (size_matrix+1, size_matrix), dtype=np.complex128)
    matrix_eig[0*size_matrix:1*size_matrix, :] \
        = eigenvectors_sorted
    matrix_eig[size_matrix, :] = eigenvalues_sorted

    return matrix_eig


def screening_eig(matrix_eig: ArrayComplex,
                  check: ArrayBool,
                  *phys_qtys) -> tuple[ArrayComplex,
                                       tuple[ArrayAny, ...]]:
    """Exclude invalid eigenmodes.

    Parameters
    ----------
    matrix_eig : ArrayComplex
        The matrix storing the eigenvalues and eigenvectors.
    check : ArrayBool
        The validity of the eigenmodes.
    *phys_qtys
        The physical quantities.

    Returns
    -------
    matrix_eig : ArrayComplex
        The eigenvalues and normalized eigenvectors.
    phys_qtys : tuple[ArrayAny, ...]
        The physical quantities.

    Warnings
    --------
    Invalid shape of the input arrays
        If the shapes of the input arrays do not match.

    Examples
    --------
    >>> from package_common.utils_eig import sort_eig, screening_eig
    >>> eigenvalues, eigenvectors = np.linalg.eig(matrix)
    >>> matrix_eig = sort_eig(eigenvalues, eigenvectors)
    >>> matrix_eig, phys_qtys = screening_eig(matrix_eig, check,
    phys_qtys)
    """

    logger: DefaultLogger = create_function_name_logger()

    size_mat: int = matrix_eig.shape[1]

    if (matrix_eig.shape[0] != size_mat + 1) \
            or (len(check.ravel()) != size_mat):
        logger.error('Invalid shape of the input arrays')
        sys.exit(1)

    for phys_qty in phys_qtys:
        if len(phys_qty.ravel()) != size_mat:
            logger.error('Invalid shape of the input arrays')
            sys.exit(1)

    list_phys_qtys: list[ArrayAny] = list(phys_qtys)

    invalid: ArrayBool = np.logical_not(check)
    matrix_eig[:, invalid] = np.nan
    for phys_qty in list_phys_qtys:
        phys_qty[invalid] = np.nan

    phys_qtys = tuple(list_phys_qtys)

    return matrix_eig, phys_qtys
