# import fonction_tomo
from Tomography.core.fonction_tomo import full_inversion_toroidal


import importlib
import numpy as np
import matplotlib.pyplot as plt
import pdb


#for easy debugging
import Tomography
importlib.reload(Tomography.core.fonction_tomo)

#import relevant function from tomography package
from Tomography.core.fonction_tomo import full_inversion_toroidal

###### path parameters to look for calibrations, 3D models, etc..

# mask for the camera
path_mask =  '/compass/home/fevre/version_pleque/data_file/15487_mask_20190712.mat'
# path_mask = None

# paths_calibration = [path calibration 1, path calibration 2, ...] : path for calcam camera calibrations. Here it is put in an array to loop over different calibrations for testing
# if the directory name has already been set in Tomography/ressources/folder_paths.yaml, you only need to put the name of the file, instead of the whole path
path_calibration ='/compass/Shared/Common/COMPASS/Diagnostics/Cameras/Calibrations/Calcam calibration/15487-15482/From Alexandra/2022_10_05 - 15487 recalibration_C.ccc'

# path for the limits of the vessel/ 3D model of the vessel. 
path_wall = '/compass/home/fevre/WESTCOMPASS_tomography/Tomography/ressources/COMPASS_RZ_vessel.mat'
path_CAD ='/compass/home/fevre/WESTCOMPASS_tomography/models_and_calibrations/models/compass/compass 20879 view camera.ccm'
######
###### raytracing parameters : parameters for the calculation of the geometry matrix
machine = 'COMPASS'
symetry = 'magnetic' #hypothesis on the emissivity uniformity. Can be set to 'toroidal'
# parameters for dimension of the 2D plane
phi_grid = 145 #toroidal angle (in degrees)
n_polar = 360 # number of toroidal points in 1 revolution for magnetic lines(only relevant for magnetic symmetry. Set to 1 for toroidal symmetry)
dr_grid = 2e-3 #radius step of 2D grid
dz_grid = 2e-3 #height step of 2D grid
# This dictionnary is there to add more parameters to the raytracing. See the function full_inversion_toroidal for help
dict_transfert_matrix = {'grid_precision_multiplier':4, 'variant_mag':'V4_std_O', 'revision':21}
dict_denoising = {'c_c' :3, 'sigma' : 2, 'median' : 10}
variant = '2018_11 - with midplane' # parameters for the variant of the 3D model
# parameters to specify the model for the reflection of the walls
name_material =     'absorbing_surface'


######

###### inversion parameters : if a geometry matrix has already been measured with the previous parameters, will skip the raytracing and go straight into the inversion

inversion_method = 'Bob' # see inversion_and_thresolding function in inversion_module module for list of choices 
inversion_parameter = {'min_visibility_node': 1, 'rcond':1e-4}
# inversion_parameter = {}

    # min_visibility_node : 

decimation = 1 # int : used to average camera data into blocks of pixels; useful for large number of pixels. 
    # decimation = 1 : takes all pixels
    # decimation = 2 : takes the mean value of 2*2 pixel block, effectively dividing by 4 the number of pixels



# input for video : specify what section of the video to load. 
time_input = [1150, 1250]
# time_input = None # [t0, t1] in milliseconds
frame_input = None
# frame_input = [54001, 54400] # number of the frames
            # if left at none, will treat the whole video
            # if both specified, will take time_input over frame_input


param_fit = 'vid'# 
Verbose = False #if set to True, will plot additionnal figures along the raytracing process to vizualize if the process runs well
# can add a long time, best set to false once raytracing gives satisfactory results.



# parameter for the number of the shot

nshot =15487
path_vid = None


#####



[transfert_matrix, 
vid,
images_retrofit_full,
inversion_results_full, 
inversion_results_thresolded_full,
pixels, 
noeuds, 
dr_grid, 
dz_grid, 
nb_noeuds_r, 
nb_noeuds_z, 
RZwall, 
R_wall, 
Z_wall, 
world, 
full_wall,
R_noeud, 
Z_noeud, 
mask_pixel,
mask_noeud,
path_parameters, 
path_transfert_matrix] = full_inversion_toroidal(nshot, 
                                                path_calibration, 
                                                path_mask, 
                                                path_wall, 
                                                machine, symetry, 
                                                time_input = time_input, 
                                                frame_input = frame_input, 
                                                dr_grid = dr_grid, 
                                                dz_grid = dr_grid, 
                                                verbose = Verbose, 
                                                inversion_method = inversion_method, 
                                                name_material = name_material,  
                                                path_vid = path_vid, 
                                                path_CAD = path_CAD, 
                                                variant = variant,
                                                inversion_parameter = inversion_parameter, 
                                                phi_grid = phi_grid,
                                                decimation =decimation,
                                                param_fit = param_fit,
                                                real_inv_flag= 1,
                                                synth_inv_flag=0,
                                                dict_transfert_matrix=dict_transfert_matrix,
                                                dict_denoising = dict_denoising,
                                                n_polar= n_polar
                                                )
