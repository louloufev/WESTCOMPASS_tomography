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
# path_calibration = '/Home/LF276573/Zone_Travail/Python/CHERAB/models_and_calibration/camera calibrations/west/treated_calibrations/calibration_60851_retry.npz'
# path_calibration = '/Home/LF276573/Zone_Travail/Python/CHERAB/models_and_calibration/camera calibrations/west/calibration_west_4_with_mask.npz'
path_mask = '/Home/LF276573/Zone_Travail/Python/CHERAB/masks/west/60990/custom_frame_421_1.npy'
# nshot = '61363'
symetry = 'toroidal'
machine = 'WEST'
path_wall = '/Home/LF276573/Zone_Travail/Python/CHERAB/models_and_calibration/models/west/WEST_wall.npy'
# path_vid = '/Home/LF276573/Documents/Python/CHERAB/images/west/images_reelles/med180.png'
# path_vid = '61363_before_nitrogen_inject_mean_between6_9s_and_7_0s.png'
# path_vid = '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/61528 extrait2_C001H001S0001/61528 extrait2_C001H001S0001'
# path_vid = '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/61363 extrait transition halpha_S0001/61363 extrait transition halpha_S0001'
# path_vid = '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/60783 extrait moyen ohmic Halpha_S0001/60783 extrait moyen ohmic Halpha_S0001'
# path_vid = '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/61363 extrait XPR Halpha_S0001/61363 extrait XPR Halpha_S0001'
# path_vid = 'Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/61537 extrait transition azote_S0001/61537 extrait transition azote_S0001'
# path_vid ='/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/61363 extrait long XPR Halpha_S0001/61363 extrait long XPR Halpha_S0001'
# path_vid ='/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/61363 extrait long avant XPR_S0001/61363 extrait long avant XPR_S0001'
# path_CAD = '/Home/LF276573/Zone_Travail/Python/CHERAB/models_and_calibration/models/west/west_detailed_correct_rotation/' new model
# path_CAD = '/Home/LF276573/Zone_Travail/Python/CHERAB/models_and_calibration/models/west/WEST_WALL_STL_20230718_rotated/' old model
# path_CAD = '/Home/LF276573/Zone_Travail/Python/CHERAB/models_and_calibration/models/west/west_detailed_correct_rotation/stl/' new model with stl
# name_material = 'tungsten03'
frame_input = None
ignore_mask_calibration = 0
# inversion_parameter = {'reduce_frames':100}
# inversion_parameter = {'reduce_frames':100, 'nobaffle':1}
# inversion_parameter = dict({})
param_reflexion = 'crop_center'
# param_reflexion = None
# inversion_parameter = {"extrait" : 'XPR'}
phi_grid = 1
dr_grid = 1e-2
dz_grid = 1e-2
decimation = 4
paths_vid = [
            # '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/61357 transition XPR_S0001/61357 transition XPR_S0001',
            #  '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/61542 transition XPR_S0001/61542 transition XPR_S0001',
            #  '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/61566 transition XPR_S0001/61566 transition XPR_S0001']
            #  '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/60637 ohmic apres xpr sans filtre_S0001/60637 ohmic apres xpr sans filtre_S0001',
            #  '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/60637 ohmic avant xpr sans filtre_S0001/60637 ohmic avant xpr sans filtre_S0001',
            # '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/61813 ohmic azote avant xpr sans filtre_S0001/61813 ohmic azote avant xpr sans filtre_S0001',
            # '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/61813 ohmic azote apres xpr sans filtre_S0001/61813 ohmic azote apres xpr sans filtre_S0001',
            '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/61359 transitions XPR _S0001/61359 transitions XPR _S0001',
            '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/61705 transition XPR_S0001/61705 transition XPR_S0001',
            '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/61363 extrait transition halpha_S0001/61363 extrait transition halpha_S0001',
            '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/61710_C001H001S0001/61710_C001H001S0001',
            '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/61600 first XPR_S0001/61600 first XPR_S0001',
            '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/61600 second XPR_S0001/61600 second XPR_S0001',
            '/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/60701 transition XPR_S0001/60701 transition XPR_S0001',
]
# 
# nshots = []
nshots = [61359, 61705, 61363,  61710, 61600, 61600, 60701]
reduce_frames = [100, 100, 10, 10, 400, 400]
params_fit = [None, None, None, None, 'camera', 'camera']
paths_calibration = ['/Home/LF276573/Zone_Travail/Python/CHERAB/models_and_calibration/camera calibrations/west/treated_calibrations/calibration_60851_retry.npz',
                    '/Home/LF276573/Zone_Travail/Python/CHERAB/models_and_calibration/camera calibrations/west/calibration_west_4_with_mask.npz']
# paths_CAD = ['/Home/LF276573/Zone_Travail/Python/CHERAB/models_and_calibration/models/west/WEST_WALL_STL_20230718_rotated/',
#              '/Home/LF276573/Zone_Travail/Python/CHERAB/models_and_calibration/models/west/20240429 full model stl extracted/']
paths_CAD = ['/Home/LF276573/Zone_Travail/Python/CHERAB/models_and_calibration/models/west/WEST_WALL_STL_20230718_rotated/',
             '/Home/LF276573/Zone_Travail/Python/CHERAB/models_and_calibration/models/west/16052025antennamonoblock/']
materials = ['tungsten001', 'tungsten03','tungsten05','tungsten1']
for k in range(len(materials)): 
    name_material = materials[k]
    for j in range(len(paths_calibration)):
    # for j in range(1,2):
        path_calibration = paths_calibration[j]
        path_CAD = paths_CAD[j]
        for i in range(len(paths_vid)):

            path_vid = paths_vid[i]
            # inversion_parameter = {'reduce_frames':reduce_frames[i]}
            inversion_parameter = {}
            nshot = nshots[i]
            param_fit = None
            # param_fit = params_fit[i]
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
                                                            time_input = None, 
                                                            frame_input = frame_input, 
                                                            dr_grid = dr_grid, 
                                                            dz_grid = dr_grid, 
                                                            verbose = 0, 
                                                            inversion_method = 'SART', 
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
