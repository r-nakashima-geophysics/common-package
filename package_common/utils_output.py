"""A Python module to assist the output of data and information."""

from pathlib import Path

from package_common.default_logger import DefaultLogger
from package_common.utils_name import create_function_name_logger


def output_info(path_source_file: Path,
                path_output_file: Path,
                *,
                marker_start: str = '# --- [start] output info --- #',
                marker_end: str = '# --- [end] output info --- #') -> None:
    """Output code fragment of a source file.

    Parameters
    ----------
    path_source_file : Path
        The path to the source file.
    path_output_file : Path
        The path to the output file.
    marker_start : str, optional, default '# --- [start] output info --- #'
        The start marker of the code fragment
    marker_end : str, optional, default '# --- [end] output info --- #'
        The end marker of the code fragment

    Warnings
    --------
    Markers not found
        If the start or end markers are not found in the source file.
    Start marker is after end marker
        If the end marker does not appear after the start marker in the source
        file.
    """

    source: str = path_source_file.read_text(encoding='utf-8')

    start: int = source.find(marker_start)
    end: int = source.rfind(marker_end)

    logger: DefaultLogger
    if (start == -1) or (end == -1):
        logger = create_function_name_logger()
        logger.error('Markers not found')

    if start > end:
        logger = create_function_name_logger()
        logger.error('Start marker is after end marker')

    fragment: str = source[start+len(marker_start):end]
    path_output_file.write_text(fragment, encoding='utf-8')

    name_output: str = path_output_file.name
    DefaultLogger(name_output).info('Saved')
