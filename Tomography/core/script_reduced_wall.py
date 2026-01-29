import importlib
import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
sys.path.append('/Home/LF285735/Documents/Python/WESTCOMPASS_tomography')
from Tomography.core import utility_functions, metadata_handling, input_west, fonction_tomo, input_west_Halpha
sys.path.append('/Home/LF285735/Documents/Python')
# from mcpherson import setup
from PIL import Image
import xarray as xr



#### setting parameters to scan
diffuse_coefficients = [1, 3, 5, 7, 9]
name_materials = [f'Tomography/ressources/components_west_0{coeff}' for coeff in diffuse_coefficients]
name_materials.append("absorbing_surface")
ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion = input_west_Halpha.load_input()
# ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion = input_west.load_input()

ParamsMachine.path_wall = "Tomography/ressources/RZ_WEST_reduced.npy"
ParamsMachine.name_material = name_materials[1]

path_ds = metadata_handling.get_or_create_inversion(ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion)

