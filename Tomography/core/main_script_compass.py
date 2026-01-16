from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
sys.path.append('/compass/home/fevre/WESTCOMPASS_tomography/') #input directory path of the package
from Tomography.core import result_inversion, metadata_handling, fonction_tomo, input_compass

ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion, ParamsDenoising = input_compass.load_input()

ParamsDenoising.c = 5
ParamsInversion.inversion_method = 'Bob'
rt_path = metadata_handling.get_or_create_raytracing(ParamsMachine, ParamsGrid)
rt_ds = xr.open_zarr(rt_path)

treated_videos_path = metadata_handling.get_or_create_treated_videos(ParamsVideo)
treated_videos_ds = xr.open_zarr(treated_videos_path)

inv_path = metadata_handling.get_or_create_inversion(ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion)
inv_ds = xr.open_zarr(inv_path)

denoised_path = metadata_handling.get_or_create_denoising(ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion, ParamsDenoising)
denoised_ds = xr.open_zarr(denoised_path)