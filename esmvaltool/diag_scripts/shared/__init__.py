"""Code that is shared between multiple diagnostic scripts."""
from . import plot
from ._base import get_cfg, run_diagnostic
from .python_diag import *

__all__ = [
    'DAY_Y',
    'DAY_M',
    'HEIGHT',
    'LAT',
    'LON',
    'MONTH',
    'TIME',
    'YEAR',
    'EXP',
    'LONG_NAME',
    'MODEL',
    'OBS',
    'PROJECT',
    'SHORT_NAME',
    'STANDARD_NAME',
    'UNITS',
    'OUTPUT_FILE_TYPE',
    'PLOT_DIR',
    'SCRIPT',
    'VERSION',
    'WORK_DIR',
    'WRITE_NETCDF',
    'WRITE_PLOTS',
    'Variable',
    'Variables',
    'Models',
    'get_cfg',
    'plot',
    'run_diagnostic',
]
