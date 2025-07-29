import fonction_tomo
import importlib
import numpy as np
import matplotlib.pyplot as plt
import inversion_module
import utility_functions
importlib.reload(inversion_module)
importlib.reload(fonction_tomo)
importlib.reload(utility_functions)
import pdb
path_mask = '/compass/home/fevre/WESTCOMPASS_tomography/functions/test_mask.npy'
symetry = 'magnetic'
# symetry = 'toroidal'
machine = 'COMPASS'
path_wall = '/compass/home/fevre/WESTCOMPASS_tomography/ressources/COMPASS_RZ_vessel.mat'
ignore_mask_calibration = 0
dict_transfert_matrix = {'grid_precision_multiplier':4, 'variant':'V4_std_O', 'revision':21, 'phi_grid': 145}
inversion_method = 'lstsq' 
phi_grid = 145
n_polar = 360
dr_grid = 50e-3
dz_grid = 50e-3
decimation = 1
# time_input = [1070, 1110] 
time_input = None
frame_input = [54001, 54400] 
# frame_input = None
nshots = [
    # 20827,
    15487
    ]
reduce_frames = [100, 100, 10, 10, 400, 400, 4]
params_fit = ['vid']
paths_calibration = [
    # '/compass/home/fevre/WESTCOMPASS_tomography/models_and_calibrations/calibrations/compass/20879_2021_03_31 - first trial.ccc',
    '/compass/Shared/Common/COMPASS/Diagnostics/Cameras/Calibrations/Calcam calibration/15487-15482/From Alexandra/2022_10_05 - 15487 recalibration_C.ccc',
    # '/compass/Shared/Common/COMPASS/Diagnostics/Cameras/Calibrations/Calcam calibration/15487-15482/From Sarah/15478_last_test_18_10_2021_115.ccc',
        ]
paths_CAD = [
   # '/compass/home/fevre/WESTCOMPASS_tomography/models_and_calibrations/models/compass/COMPASS.ccm', 
    '/compass/home/fevre/WESTCOMPASS_tomography/models_and_calibrations/models/compass/compass 20879 view camera.ccm', 
]

variant = '2018_11 - with midplane'
materials = [
        'absorbing_surface', 
        # 'tungsten001', 
        # 'tungsten03',
        # 'tungsten05',
        # 'tungsten1',
             ]
for k in range(len(materials)): 
    name_material = materials[k]
    # name_material = 'tungsten05'
    for j in range(len(paths_calibration)):
        path_calibration = paths_calibration[j]
        path_CAD = paths_CAD[j]
        for i in range(len(params_fit)):

            # path_vid = '/compass/home/fevre/WESTCOMPASS_tomography/functions/image calib.png'
            path_vid = None
            inversion_parameter = {}
            nshot = nshots[i]
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
            mask_noeud] = fonction_tomo.full_inversion_toroidal(nshot, 
                                                            path_calibration, 
                                                            path_mask, 
                                                            path_wall, 
                                                            machine, symetry, 
                                                            time_input = time_input, 
                                                            frame_input = frame_input, 
                                                            dr_grid = dr_grid, 
                                                            dz_grid = dr_grid, 
                                                            verbose = 0, 
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
