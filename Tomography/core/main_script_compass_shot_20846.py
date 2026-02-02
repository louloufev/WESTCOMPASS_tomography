from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
sys.path.append('/compass/home/fevre/WESTCOMPASS_tomography/') #input directory path of the package
from Tomography.core import result_inversion, metadata_handling, fonction_tomo, input_compass_20879, utility_functions

ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion, ParamsDenoising = input_compass_20879.load_input()

# inversion step : obtain inversion results (inversion matrix saved for further processing elsewhere, but not loaded for memory usage)
inv_path = metadata_handling.get_or_create_inversion(ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion)
inv_ds = xr.open_zarr(inv_path)

# denoising step 
denoised_path = metadata_handling.get_or_create_denoising(ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion, ParamsDenoising)
denoised_ds = xr.open_zarr(denoised_path)