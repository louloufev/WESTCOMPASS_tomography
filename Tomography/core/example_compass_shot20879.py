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

ignore_mask_calibration = False # if False, the mask saved in the calcam calibration will be prioritized over the mask from path_mask, unless this parameter is set to true


# paths_calibration = [path calibration 1, path calibration 2, ...] : path for calcam camera calibrations. Here it is put in an array to loop over different calibrations for testing
# if the directory name has already been set in Tomography/ressources/folder_paths.yaml, you only need to put the name of the file, instead of the whole path
paths_calibration = [
    '/compass/home/fevre/WESTCOMPASS_tomography/models_and_calibrations/calibrations/compass/20879_2021_03_31 - first trial.ccc',
    # '/compass/Shared/Common/COMPASS/Diagnostics/Cameras/Calibrations/Calcam calibration/15487-15482/From Alexandra/2022_10_05 - 15487 recalibration_C.ccc',

    # '/compass/Shared/Common/COMPASS/Diagnostics/Cameras/Calibrations/Calcam calibration/15487-15482/From Sarah/15478_last_test_18_10_2021_115.ccc',
        ]

# path for the limits of the vessel/ 3D model of the vessel. 
path_wall = '/compass/home/fevre/WESTCOMPASS_tomography/Tomography/ressources/COMPASS_RZ_vessel.mat'
paths_CAD = [
   # '/compass/home/fevre/WESTCOMPASS_tomography/models_and_calibrations/models/compass/COMPASS.ccm', 
    '/compass/home/fevre/WESTCOMPASS_tomography/models_and_calibrations/models/compass/compass 20879 view camera.ccm', 
]
######
###### raytracing parameters : parameters for the calculation of the geometry matrix
machine = 'COMPASS'
symetry = 'magnetic' #hypothesis on the emissivity uniformity. Can be set to 'toroidal'

# parameters for dimension of the 2D plane
phi_grid = 220 #toroidal angle (in degrees)
n_polar = 360 # number of toroidal points in 1 revolution for magnetic lines(only relevant for magnetic symmetry. Set to 1 for toroidal symmetry)
dr_grid = 5e-3 #radius step of 2D grid
dz_grid = 5e-3 #height step of 2D grid
# This dictionnary is there to add more parameters to the raytracing. See the function full_inversion_toroidal for help
dict_transfert_matrix = {'grid_precision_multiplier':4, 'variant':'', 'revision':-1}

variant = 'view_camera' # parameters for the variant of the 3D model
# parameters to specify the model for the reflection of the walls
materials = [
        'absorbing_surface', 
        # 'tungsten001', 
        # 'tungsten03',
        # 'tungsten05',
        # 'tungsten1',
             ]

######

###### inversion parameters : if a geometry matrix has already been measured with the previous parameters, will skip the raytracing and go straight into the inversion

inversion_method = 'lstsq' # see inversion_and_thresolding function in inversion_module module for list of choices 
decimation = 1 # int : used to average camera data into blocks of pixels; useful for large number of pixels. 
    # decimation = 1 : takes all pixels
    # decimation = 2 : takes the mean value of 2*2 pixel block, effectively dividing by 4 the number of pixels



# input for video : specify what section of the video to load.  
time_input = None # [t0, t1] in milliseconds
frame_input = [54001, 54400] # number of the frames
            # if left at none, will treat the whole video
            # if both specified, will take time_input over frame_input

# reduce_frame = int;  if the video is too long to inverse, the programm can average over every few frames. Input here how much frames should be averaged.
reduce_frames = [10, 100, 10, 10, 400, 400, 4] # array to loop over different inversion parameters

params_fit = ['vid'] # 
Verbose = False #if set to True, will plot additionnal plots along the raytracing process to vizualize if the process runs well
# can add a long time, best set to false once raytracing gives satisfactory results.



# parameter for the number of the shot, in this example it is put in an array to loop over it

nshots = [
    20827,
    # 15487
    ]

#####

# Since the programm does not need to repeat the raytracing if the raytracing parameters stay the same,
# it might be useful to do 2 loops, one on the raytracing parameters, the other on the inversion parameters


for k in range(len(materials)): # loop over the raytracing parameters
    name_material = materials[k]
    # name_material = 'tungsten05'
    for j in range(len(paths_calibration)):
        path_calibration = paths_calibration[j]
        path_CAD = paths_CAD[0]
        nshot = nshots[i]

        for i in range(len(params_fit)):# loop over the inversion parameters

            path_vid = None
            inversion_parameter = {}
            param_fit = params_fit[i]
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
            mask_noeud] = full_inversion_toroidal(nshot, 
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
                                                            ignore_mask_calibration = ignore_mask_calibration,
                                                            param_fit = param_fit,
                                                            real_inv_flag= 1,
                                                            synth_inv_flag=0,
                                                            dict_transfert_matrix=dict_transfert_matrix,
                                                            n_polar= n_polar
                                                            )
