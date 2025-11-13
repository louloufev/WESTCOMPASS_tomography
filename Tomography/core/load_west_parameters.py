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
from Tomography.core.fonction_tomo_test import full_inversion_toroidal, geometry_matrix_spectro

###### path parameters to look for calibrations, 3D models, etc..

# mask for the camera
# path_mask =  '/Home/LF285735/Zone_Travail/Python/CHERAB/masks/west/60990/custom_frame_421_1.npy'

path_mask = None

# if the directory name has already been set in Tomography/ressources/folder_paths.yaml, you only need to put the name of the file, instead of the whole path
path_calibration = '/Home/LF285735/Zone_Travail/Python/CHERAB/models_and_calibration/camera calibrations/west/calibration_west_4_with_mask.ccc'
# path for the limits of the vessel/ 3D model of the vessel. 
# path_wall = '/compass/home/fevre/WESTCOMPASS_tomography/Tomography/ressources/COMPASS_RZ_vessel.mat'
path_wall = '/Home/LF285735/Zone_Travail/Python/CHERAB/models_and_calibration/models/west/WEST_wall.npy'
path_CAD ='/Home/LF285735/Zone_Travail/Python/CHERAB/models_and_calibration/models/west/20250429 full model.ccm'
######
###### raytracing parameters : parameters for the calculation of the geometry matrix
machine = 'WEST'
symetry = 'toroidal' #hypothesis on the emissivity uniformity. Can be set to 'toroidal'
# parameters for dimension of the 2D plane
phi_grid = None #toroidal angle (in degrees)
n_polar = None # number of toroidal points in 1 revolution for magnetic lines(only relevant for magnetic symmetry. Set to 1 for toroidal symmetry)
dr_grid = 2e-3 #radius step of 2D grid
dz_grid = 2e-3 #height step of 2D grid
extra_steps = None
# This dictionnary is there to add more parameters to the raytracing. See the function full_inversion_toroidal for help
grid_precision_multiplier = None
variant_mag=None
revision = None
dict_vid = {'sigma' : 2, 'median' : 0}
variant_CAD = 'Default' # parameters for the variant of the 3D model
# parameters to specify the model for the reflection of the walls
name_material =     'absorbing_surface'
c = 3


######

###### inversion parameters : if a geometry matrix has already been measured with the previous parameters, will skip the raytracing and go straight into the inversion

inversion_method = 'OPENSART' # see inversion_and_thresolding function in inversion_module module for list of choices 
inversion_parameter = {}
# inversion_parameter = {}

    # min_visibility_node : 

decimation = 4# int : used to average camera data into blocks of pixels; useful for large number of pixels. 
    # decimation = 1 : takes all pixels
    # decimation = 2 : takes the mean value of 2*2 pixel block, effectively dividing by 4 the number of pixels



# input for video : specify what section of the video to load. 
# time_input = [1.150, 1.151]
time_input = None # [t0, t1] in seconds
# frame_input = None
frame_input = [100, 200] # number of the frames
            # if left at none, will treat the whole video
            # if both specified, will take time_input over frame_input


param_fit = None# 
Verbose = False #if set to True, will plot additionnal figures along the raytracing process to vizualize if the process runs well
# can add a long time, best set to false once raytracing gives satisfactory results.



# parameter for the number of the shot
nshot_grid =61357
nshot =None
path_vid = '/Home/LF285735/Zone_Travail/Python/CHERAB/videos/west/61357 avant XPR _S0001/61357 avant XPR _S0001'


#####
ParamsMachine = result_inversion.ParamsMachine(machine  = machine,
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
                                                    variant_mag = variant_mag,
                                                    revision = revision,
                                                    phi_grid = phi_grid,
                                                    grid_precision_multiplier =grid_precision_multiplier,
                                                    n_polar = n_polar,
                                                    crop_center = False,
                                                    extra_steps = extra_steps,
                                                    nshot = nshot_grid,
                                                    class_name = 'ParamsGrid')


ParamsVid = result_inversion.ParamsVid(inversion_method = inversion_method,
                                                    nshot = nshot,
                                                    path_vid = path_vid,
                                                    dict_vid = dict_vid,
                                                    time_input =time_input,
                                                    frame_input = frame_input,
                                                    inversion_parameter = inversion_parameter,
                                                    c = c, 
                                                    class_name = 'ParamsVid')



R1, R2, Z1, Z2, tf, grid_mask, extent_RZ, R_wall, Z_wall = geometry_matrix_spectro(ParamsMachine, ParamsGrid)

plt.imshow(utility_functions.reconstruct_2D_image(np.sum(tf[:, :], 0), grid_mask), extent=extent_RZ, origin= "lower", vmin = 100)
plt.plot(R_wall, Z_wall, 'k')
plt.show()