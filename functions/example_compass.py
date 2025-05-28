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
path_mask = None
symetry = 'magnetic'
machine = 'COMPASS'
path_wall = '/compass/home/fevre/WESTCOMPASS_tomography/ressources/COMPASS_RZ_vessel.mat'
frame_input = None
ignore_mask_calibration = 0
param_reflexion = None
inversion_method = 'SART' 
phi_grid = 1
dr_grid = 1e-2
dz_grid = 1e-2
decimation = 1
time_input =[1, 1.001] 
nshots = [20879]
reduce_frames = [100, 100, 10, 10, 400, 400, 4]
params_fit = ['camera', None, None, None, 'camera', 'camera', 'camera']
paths_calibration = ['/compass/home/fevre/WESTCOMPASS_tomography/models_and_calibrations/calibrations/compass/20879_2021_03_31 - first trial.ccc',
                   ]
paths_CAD = [None, 
]
materials = ['absorbing_surface', 'tungsten001', 'tungsten03','tungsten05','tungsten1']
for k in range(len(materials)): 
    name_material = materials[k]
    for j in range(len(paths_calibration)):
        path_calibration = paths_calibration[j]
        path_CAD = paths_CAD[j]
        for i in range(len(params_fit)):

            path_vid = 'None'
            inversion_parameter = {}
            nshot = nshots[i]
            param_fit = params_fit[i]
            [transfert_matrix, 
            inversion_results, 
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
                                                            inversion_parameter = inversion_parameter, 
                                                            phi_grid = phi_grid,
                                                            decimation =decimation,
                                                            ignore_mask_calibration = ignore_mask_calibration,
                                                            param_fit = param_fit,
                                                            real_inv_flag= 1,
                                                            synth_inv_flag=0,
                                                            param_reflexion=param_reflexion
                                                            )
