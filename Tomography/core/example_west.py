from Tomography.core.fonction_tomo_test import full_inversion_toroidal



import importlib
import numpy as np
import matplotlib.pyplot as plt
import pdb
from Tomography.core import utility_functions, result_inversion


#for easy debugging
import Tomography
importlib.reload(Tomography.core.fonction_tomo_test)

#import relevant function from tomography package
from Tomography.core.fonction_tomo_test import full_inversion_toroidal

###### path parameters to look for calibrations, 3D models, etc..

# mask for the camera
path_mask = '/Home/LF276573/Zone_Travail/Python/CHERAB/masks/west/60990/custom_frame_421_1.npy'
# nshot = '61363'

path_wall = '/Home/LF276573/Zone_Travail/Python/CHERAB/models_and_calibration/models/west/WEST_wall.npy'

# paths_calibration = [path calibration 1, path calibration 2, ...] : path for calcam camera calibrations. Here it is put in an array to loop over different calibrations for testing
# if the directory name has already been set in Tomography/ressources/folder_paths.yaml, you only need to put the name of the file, instead of the whole path
path_calibration ='/Home/LF276573/Zone_Travail/Python/CHERAB/models_and_calibration/camera calibrations/west/treated_calibrations/calibration_60851_retry.npz'
# path for the limits of the vessel/ 3D model of the vessel. 
path_CAD ='/Home/LF276573/Zone_Travail/Python/CHERAB/models_and_calibration/models/west/20250429 full model.ccm'
###### raytracing parameters : parameters for the calculation of the geometry matrix
symetry = 'toroidal'
machine = 'WEST'
# parameters for dimension of the 2D plane

phi_grid = None
dr_grid = 1e-2
dz_grid = 1e-2
decimation = 4
crop_center = True
# This dictionnary is there to add more parameters to the raytracing. See the function full_inversion_toroidal for help

dict_denoising = {'c' :3, 'sigma' : 2, 'median' : 10}
variant_CAD = 'Default' # parameters for the variant of the 3D model
# parameters to specify the model for the reflection of the walls
name_material =     'absorbing_surface'


######

###### inversion parameters : if a geometry matrix has already been measured with the previous parameters, will skip the raytracing and go straight into the inversion

inversion_method = 'SparseBob' # see inversion_and_thresolding function in inversion_module module for list of choices 
inversion_parameter = {}
# inversion_parameter = {}

    # min_visibility_node : 

decimation = 1 # int : used to average camera data into blocks of pixels; useful for large number of pixels. 
    # decimation = 1 : takes all pixels
    # decimation = 2 : takes the mean value of 2*2 pixel block, effectively dividing by 4 the number of pixels



# input for video : specify what section of the video to load. 
# time_input = [1.150, 1.151]
time_input = None # [t0, t1] in seconds
# frame_input = None
frame_input = None
 # number of the frames
            # if left at none, will treat the whole video
            # if both specified, will take time_input over frame_input

# paths_vid = ['/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/61357 avant XPR _S0001/61357 avant XPR _S0001',
#              '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/61357 apres XPR _S0001/61357 apres XPR _S0001',
#              '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/61537 avant xpr filtre azote_S0001/61537 avant xpr filtre azote_S0001',
#              '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/61537 apres xpr filtre azote_S0001/61537 apres xpr filtre azote_S0001']
#             #  '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/60637 ohmic apres xpr sans filtre_S0001/60637 ohmic apres xpr sans filtre_S0001',
#             #  '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/60637 ohmic avant xpr sans filtre_S0001/60637 ohmic avant xpr sans filtre_S0001']

param_fit = None# 
Verbose = False #if set to True, will plot additionnal figures along the raytracing process to vizualize if the process runs well
# can add a long time, best set to false once raytracing gives satisfactory results.



# parameter for the number of the shot

nshot =61357
path_vid = '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/61357 avant XPR _S0001/61357 avant XPR _S0001'
 


#####
ParamsMachine = result_inversion.ParamsMachine(machine  = 'WEST',
                                                    path_calibration = path_calibration,
                                                    path_wall = path_wall,
                                                    path_CAD = path_CAD,
                                                    variant_CAD = variant_CAD,
                                                    path_mask = path_mask,
                                                    name_material = name_material,
                                                    param_fit = param_fit,     
                                                    decimation = decimation,
                                                    class_name  = 'ParamsMachine')

ParamsGrid= result_inversion.ParamsGrid(dr_grid = dr_grid,
                                                    dz_grid = dz_grid,
                                                    symetry =  symetry,
                                                    crop_center = crop_center,
                                                    class_name = 'ParamsGrid')


ParamsVid = result_inversion.ParamsVid(inversion_method = inversion_method,
                                                    nshot = nshot,
                                                    path_vid = path_vid,
                                                    dict_denoising = dict_denoising,
                                                    time_input =time_input,
                                                    frame_input = frame_input,
                                                    inversion_parameter = inversion_parameter,
                                                    class_name = 'ParamsVid')


Inversion_results = full_inversion_toroidal(ParamsMachine,ParamsGrid, ParamsVid)  