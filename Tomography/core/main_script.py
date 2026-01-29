
import importlib
import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
sys.path.append('/Home/LF285735/Documents/Python/WESTCOMPASS_tomography')
from Tomography.core import utility_functions, metadata_handling, input_west, fonction_tomo, input_west_Halpha, input_west_61537
sys.path.append('/Home/LF285735/Documents/Python')
# from mcpherson import setup
from PIL import Image
import xarray as xr


import os 
connection = os.listdir('/Home/LF285735/Documents/Python/mnt/nunki/')
if connection == []:
    raise ConnectionError("Not connected to nunki servers !")

sys.path.append('/Home/LF285735/Documents/Python/WESTCOMPASS_tomography')
diffuse_coefficients = [1, 3, 5, 7, 9]
name_materials = [f'Tomography/ressources/components_west_0{coeff}' for coeff in diffuse_coefficients]
name_materials.append("absorbing_surface")

####default
#Halpha 
ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion = input_west_Halpha.load_input()
ParamsVideo.dict_vid.update(reduce_frames = 10)
ParamsVideo.frame_input = None
path_61357 = metadata_handling.get_or_create_inversion(ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion)
inv_ds_61357 = xr.open_zarr(path_61357)

#Azote
ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion = input_west_61537.load_input()
path_61537= metadata_handling.get_or_create_inversion(ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion)
inv_ds_61537 = xr.open_zarr(path_61537)


####reflection
#Halpha 
ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion = input_west_Halpha.load_input()
ParamsVideo.dict_vid.update(reduce_frames = 10)
ParamsVideo.frame_input = None
ParamsMachine.name_material = name_materials[1]
path_61357_refl = metadata_handling.get_or_create_inversion(ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion)
inv_ds_61357_refl = xr.open_zarr(path_61357_refl)

#Azote
ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion = input_west_61537.load_input()
ParamsMachine.name_material = name_materials[1]
path_61537_refl= metadata_handling.get_or_create_inversion(ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion)
inv_ds_61537_refl = xr.open_zarr(path_61537_refl)



#### reducing walls
#Halpha
ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion = input_west_Halpha.load_input()
ParamsInversion.inversion_parameter = {"new_wall" : "Tomography/ressources/RZ_WEST_reduced.npy"}
ParamsVideo.dict_vid.update(reduce_frames = 10)
ParamsVideo.frame_input = None
ParamsMachine.name_material = name_materials[1]
path_61357_refl_reduced_wall = metadata_handling.get_or_create_inversion(ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion)
inv_ds_61357_refl_reduced_wall = xr.open_zarr(path_61357_refl_reduced_wall)

#azote
ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion = input_west_61537.load_input()
ParamsInversion.inversion_parameter = {"new_wall" : "Tomography/ressources/RZ_WEST_reduced.npy"}
ParamsMachine.name_material = name_materials[1]
path_61537_refl_reduced_wall= metadata_handling.get_or_create_inversion(ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion)
inv_ds_61537_refl_reduced_wall = xr.open_zarr(path_61537_refl_reduced_wall)

