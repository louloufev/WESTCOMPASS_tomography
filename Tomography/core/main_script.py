from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
sys.path.append('/Home/LF285735/Documents/Python/WESTCOMPASS_tomography')
from Tomography.core import result_inversion, metadata_handling, fonction_tomo

exec(open("Tomography/core/input_west.py").read())



path = metadata_handling.get_or_create_inversion(ParamsMachine, ParamsGrid, ParamsVid)
