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
path_mask = '/compass/home/fevre/WESTCOMPASS_tomography/Tomography/ressources/test_mask.npy'



# paths_calibration = [path calibration 1, path calibration 2, ...] : path for calcam camera calibrations. Here it is put in an array to loop over different calibrations for testing
# if the directory name has already been set in Tomography/ressources/folder_paths.yaml, you only need to put the name of the file, instead of the whole path
paths_calibration = [
    '/compass/home/fevre/WESTCOMPASS_tomography/models_and_calibrations/calibrations/compass/20879_2021_03_31 - first trial.ccc',
    # '/compass/Shared/Common/COMPASS/Diagnostics/Cameras/Calibrations/Calcam calibration/15487-15482/From Alexandra/2022_10_05 - 15487 recalibration_C.ccc',

    # '/compass/Shared/Common/COMPASS/Diagnostics/Cameras/Calibrations/Calcam calibration/15487-15482/From Sarah/15478_last_test_18_10_2021_115.ccc',
        ]



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
path_mask = None 
# if the directory name has already been set in Tomography/ressources/folder_paths.yaml, you only need to put the name of the file, instead of the whole path
path_calibration = '/compass/home/fevre/WESTCOMPASS_tomography/models_and_calibrations/calibrations/compass/20879_2021_03_31 - first trial.ccc'
# path for the limits of the vessel/ 3D model of the vessel. 
path_wall = None #path for limits of the wall in the 2D poloidal plane. Not necessary for compass as it can load from pleque 

path_CAD ='/compass/home/fevre/WESTCOMPASS_tomography/models_and_calibrations/models/compass/compass 20879 view camera.ccm'
variant_CAD = 'view camera' # parameters for the variant of the 3D model

######
###### raytracing parameters : parameters for the calculation of the geometry matrix
machine = 'COMPASS'
symetry = 'magnetic' #hypothesis on the emissivity uniformity. Can be set to 'toroidal'
# parameters for dimension of the 2D plane
phi_grid = 'auto' #toroidal angle (in degrees). auto tries to find the angle of the plane in the middle of the field of view of the camera.
n_polar = 1800 # number of toroidal points in 1 revolution for magnetic lines(only relevant for magnetic symmetry. Set to 1 for toroidal symmetry)
dr_grid = 2e-3 #radius step of 2D grid
dz_grid = 2e-3 #height step of 2D grid
extra_steps = 4 #the programm tries to optimize the size of the poloidal plane. this adds a number of points in the R and Z directions to make sure no field lines are missed.
grid_precision_multiplier = 4 #subdivises the cell by this number into smaller cells to better match the positions of the field lines.
variant_mag=None
revision = None

dict_vid = {'sigma' : 3, 'median' : 80} 
# this dictionnary holds parameter for video treatent :
#       sigma : sigma parameter of the gaussian filter
#       median : number of frames of the median filter

# parameters to specify the model for the reflection of the walls
name_material =     'absorbing_surface'
c = 3


######

###### inversion parameters : if a geometry matrix has already been measured with the previous parameters, will skip the raytracing and go straight into the inversion

inversion_method = 'SparseBob' # see inversion_and_thresolding function in inversion_module module for list of choices 
inversion_parameter = {}
# inversion_parameter = {}

    # min_visibility_node : 

decimation = None # int : used to average camera data into blocks of pixels; useful for large number of pixels. 
    # decimation = 1 : takes all pixels
    # decimation = 2 : takes the mean value of 2*2 pixel block, effectively dividing by 4 the number of pixels



# input for video : specify what section of the video to load. 
# time_input = [1.150, 1.151]
time_input = None # [t0, t1] in seconds
# frame_input = None
frame_input = [151200, 153200] # number of the frames
            # if left at none, will treat the whole video
            # if both specified, will take time_input over frame_input


param_fit = 'vid'# calibration, mask and videos may not have the same size. Choose the one that should be the final size.
#crops or extends with 0 (keep the same center) the other two to fit the same size. Leave to None if all have the same size.
Verbose = False #if set to True, will plot additionnal figures along the raytracing process to vizualize if the process runs well
# can add a long time, best set to false once raytracing gives satisfactory results.



# parameter for the number of the shot

nshot =20846
path_vid = None


#####
ParamsMachine = result_inversion.ParamsMachine(machine  = 'COMPASS',
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
                                                    extra_steps = extra_steps,
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


Inversion_results = full_inversion_toroidal(ParamsMachine,ParamsGrid, ParamsVid)  

