"""A Python module to define a class for measuring the computational
time of a Python script.

Notes
-----
The caffeine module will be imported in the initializer of the DefaultTimer
class when this class is used on macOS.
"""

import importlib
import sys
from time import perf_counter
from types import ModuleType

from package_common.common_types import Self
from package_common.default_logger import DefaultLogger


class DefaultTimer:
    """Class to measure the computational time of a Python script.

    Examples
    --------
    >>> from package_common.default_timer import DefaultTimer
    >>> timer = DefaultTimer('my_timer')
    >>> timer.start()
    >>> timer.show()
    >>> _ = timer.lap()  # None
    >>> lap_time = timer.lap()
    >>> timer.stop()
    >>> timer.start()
    >>> timer.end()
    """

    __import_caffeine: bool = False

    def __init__(self: Self,
                 name: str) -> None:
        """Initialize an instance of the DefaultTimer class.

        Parameters
        ----------
        name : str
            The name of the timer.
        """

        self.__running: bool = False
        self.__start_time: float | None = None
        self.__elapsed_time: float | None = None
        self.__split_time: float | None = None
        self.__net_time: float | None = None

        self.__logger: DefaultLogger = DefaultLogger(name)

        if (sys.platform == "darwin") \
                and (not DefaultTimer.__import_caffeine):

            caffeine: ModuleType = importlib.import_module('caffeine')
            caffeine.on(display=False)
            DefaultTimer.__import_caffeine = True

    def start(self) -> None:
        """(Re)start the timer."""

        if self.__start_time is None:
            self.__logger.info('Start')
            self.__start_time = perf_counter()
        elif not self.__running:
            self.__start_time = perf_counter()
        self.__running = True

    def show(self) -> None:
        """Show the elapsed time.

        Warnings
        --------
        Timer has not been started
            If `start()` has not been called before `show()` or `end()` are
            called.
        """

        if self.__start_time is None:
            self.__logger.warning('Timer has not been started')
        elif self.__net_time is None:
            self.__elapsed_time = perf_counter() - self.__start_time
            self.__logger.info(f'Elapsed time: {self.__elapsed_time:.1f} sec.')
        else:
            if self.__running:
                self.stop()
                self.start()
            self.__logger.info(f'Net time: {self.__net_time:.1f} sec.')

    def end(self) -> None:
        """End the timer."""

        self.show()
        self.__logger.info('End')
        self.__running = False
        self.__start_time = None
        self.__elapsed_time = None
        self.__split_time = None
        self.__net_time = None

    def stop(self) -> None:
        """Stop the timer.

        Warnings
        --------
        Timer has not been (re)started
            If `start()` has not been called before `stop()` is called.
        """

        now: float = perf_counter()
        if (self.__start_time is None) or (not self.__running):
            self.__logger.warning('Timer has not been (re)started')
        elif self.__net_time is None:
            self.__net_time = now - self.__start_time
        elif self.__net_time is not None:
            self.__net_time += now - self.__start_time
        self.__running = False

    def lap(self) -> float | None:
        """Measure the lap time.

        Returns
        -------
        float | None
            The lap time.
        """

        now: float = perf_counter()
        if self.__split_time is None:
            self.__split_time = now
            return None
        lap_time: float = now - self.__split_time
        self.__split_time = now
        return lap_time
