from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
sys.path.append('/compass/home/fevre/WESTCOMPASS_tomography/') #input directory path of the package
from Tomography.core import result_inversion, metadata_handling, fonction_tomo

exec(open("Tomography/core/input_compass.py").read())



path = metadata_handling.get_or_create_inversion(ParamsMachine, ParamsGrid, ParamsVid)
inv_results = xr.open_zarr(path)
