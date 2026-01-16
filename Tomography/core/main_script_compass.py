from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import sys
sys.path.append('/compass/home/fevre/WESTCOMPASS_tomography/') #input directory path of the package
from Tomography.core import result_inversion, metadata_handling, fonction_tomo, input_compass, utility_functions

ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion, ParamsDenoising = input_compass.load_input()

#raytracing step : obtain transfert matrix
rt_path = metadata_handling.get_or_create_raytracing(ParamsMachine, ParamsGrid)
rt_ds = xr.open_zarr(rt_path)

# video loading step : obtain filtered camera video
treated_videos_path = metadata_handling.get_or_create_treated_videos(ParamsVideo)
treated_videos_ds = xr.open_zarr(treated_videos_path)

# inversion step : obtain inversion results (inversion matrix saved for further processing elsewhere, but not loaded for memory usage)
inv_path = metadata_handling.get_or_create_inversion(ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion)
inv_ds = xr.open_zarr(inv_path)

# denoising step 
denoised_path = metadata_handling.get_or_create_denoising(ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion, ParamsDenoising)
denoised_ds = xr.open_zarr(denoised_path)


time = inv_ds.t_inv.to_numpy() #load time values of the 399 frames. NB: inv_ds.t_inv is a xarray dataset (with coordinates and metadata), the to_numpy method returns only its values

###plot image
#plot reconstructed image from inversion (inversion result turned back into synthetic camera image)
inv_ds.image2d.plot(var = 'images_retrofit', t_inv = time[197], index_dim = 'pixel') 
#var : name of the variable 
    # inversion for inversion results
    # images_retrofit for reconstructed camera image
    # images for filtered camera image
# t_inv : time you want to plot
# index_dim : on what plan you need to reconstruct your data 
#     pixel : camera plan
#     node : poloidal plan

####same plot image, but without the implemented method for plotting, and instead the method for reconstructing the full 2D image, saving it as image_retrofit
image_retrofit = inv_ds.reconstruct(var = 'images_retrofit', t_inv = time[197], index_dim = 'pixel')
fig, ax = plt.subplots()
ax.imshow(image_retrofit[197])

plt.savefig("retrofitted_image.png") #save image

#same but with inversion results; NB : Z axis is flipped !
inversion = inv_ds.reconstruct(var = 'inversion', t_inv = time[197], index_dim = 'node')
fig, ax = plt.subplots()
ax.imshow(inversion[197])

plt.savefig("inversion.png") #save image

##### reconstruct not an image but the full video
#var : name of the variable 
    # inversion for inversion results
    # images_retrofit for reconstructed camera image
    # images for filtered camera image
# index_dim : on what plan you need to reconstruct your data 
#     pixel : camera plan
#     node : poloidal plan
#fill_value : what value to fill the masked part with (default NaN)
inversion_full_video = inv_ds.videoreconstruct.reconstruct(var = 'inversion', index_dim = 'node', fill_value = 0)

###### create and save full video. Returns the new position R, Z of the grid if the video needs clipping (can only make even values of pixels)
#filename : optional string to save the videos in another folder (here will be in working directory)
#keep_only  : possibility to keep only positive or negatives values :
    # all (default) : keep all the values, but rescale them on 8 bits(min, max)-> (0, 255)
    # peaks : keep only positive values, set negative values to 0 and rescale it on 8 bits (min, max) -> (0, max)-> (0, 255)
    # holes : keep only negative values, set positive values to 0, take the absolute value and rescale it on 8 bits (min, max) -> (min, 0)-> (0, abs(min)) -> (0, 255)
R, Z = inv_ds.videomaker.process(filename = '', var = 'images', index_dim = 'pixel', keep_only = "peaks", fill_value = 0)