import numpy as np
from os import path, cpu_count
from time2graph.utils.base_utils import Debugger

module_path = path.dirname(path.abspath(__file__))
njobs = cpu_count()
if njobs >= 40:
    njobs = int(njobs / 2)





__all__ = [
    'np',
    'path',
    'Debugger',
    'module_path',
    'njobs'
]
