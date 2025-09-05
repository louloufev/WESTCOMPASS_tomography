import numpy as np
from scipy.io import loadmat, savemat
from scipy.sparse import csr_matrix, save_npz, csc_matrix, load_npz, isspmatrix
from matplotlib import pyplot as plt
from raysect.primitive import import_stl, import_obj
from raysect.optical import World, translate, rotate, ConstantSF, Point3D
from raysect.optical.observer import PinholeCamera, FullFrameSampler2D, RGBPipeline2D, VectorCamera
from raysect.optical.material import  InhomogeneousVolumeEmitter, UniformVolumeEmitter, AbsorbingSurface, Lambert
from cherab.tools.primitives.axisymmetric_mesh import axisymmetric_mesh_from_polygon
from raysect.core.math.polygon import triangulate2d
from cherab.tools.raytransfer import RayTransferPipeline2D, RayTransferCylinder
from cherab.tools.raytransfer import RoughNickel, RoughTungsten
import os 
from raysect.primitive import Cylinder, Subtract
from tomotok.core.inversions import Bob, SparseBob, CholmodMfr, Mfr
from cherab.tools.inversions import invert_regularised_nnls
from tomotok.core.derivative import compute_aniso_dmats
from tomotok.core.geometry import RegularGrid
from scipy.interpolate import RegularGridInterpolator
import subprocess
import pdb
import pickle
import sys
import time
import importlib
from . import utility_functions

import tkinter as tk
from tkinter import filedialog

from .inversion_module import prep_inversion, inverse_vid, inversion_and_thresolding, synth_inversion, plot_results_inversion, reconstruct_2D_image, plot_results_inversion_simplified
from Tomography.ressources import components, paths

#### yaml files
# The folder_paths.yaml file can be used to easily input in which files you wish to load/save your inputs/outputs
# if left blank, will create folders in your working directory
# main_folder_transfert_matrices : folder for geometry matrixes as well as their inputs for the raytracing
# main_folder_calibration : folder for your camera calibrations (.ccc format, from Calcam software)
# main_folder_CAD : folder for your tokamak model (.ccm format, from Calcam software)
# main_folder_resultat : for the results of the inversion, save them in .mat format for easiest use with matlab
# main_folder_image : this folder is for images that get automatically drawn by the function to quickly check your results


# with open("/ressources/folder_paths.yaml", "r") as f:
#     paths = yaml.safe_load(f)

main_folder_transfert_matrices = paths["main_folder_transfert_matrices"]
main_folder_resultat = paths["main_folder_resultat"]
main_folder_image = paths["main_folder_image"]
main_folder_calibration = paths["main_folder_calibration"]
main_folder_CAD = paths["main_folder_CAD"]
main_folder_processing = paths["main_folder_processing"]

def full_inversion_toroidal(nshot, 
                            path_calibration, 
                            path_mask, 
                            path_wall, 
                            machine, 
                            symetry,
                            time_input = None, 
                            frame_input = None, 
                            dr_grid = 1e-2, 
                            dz_grid = 1e-2, 
                            verbose = 0, 
                            inversion_method = 'lstsq', 
                            name_material = 'absorbing_surface', 
                            path_vid = None, 
                            path_CAD = None,
                            variant = 'Default',
                            inversion_parameter = {"rcond" : 1e-3}, 
                            phi_grid = None, 
                            decimation = 1, 
                            param_fit = None,  
                            dict_transfert_matrix = {}, 
                            dict_denoising = {'c_c' :3, 'sigma' : 0, 'median' : 10},
                            synth_inv_flag = 0, 
                            real_inv_flag = 1, 
                            n_polar = 1):
    """
    This function does the inversion for a given shot, for a given time or frame span.
    It only needs access to the video, its mask, the calibration of the camera, and the CAD model of the machine or the coordinates of the wall
    If the transfert matrix has already been calculated for the given parameters, it won't recalculate it. 
   
    
    Inputs :
        nshot : integer
            number of the shot
        path_calibration : string
            File path to the chosen calibration saved in compressed npz format. It needs to have been created by the load_calibration function from the calcam_module using the Calcam environment. 
        path_wall : string
            File path to the coordinates of the wall. The path_wall string should lead to either a mat or a npy file, containing a 2D array of the form [N*2, (R, Z)] (N being the number of points that describe the wall) 
        path_mask : string
            File path to the chosen mask. Use the create_mask function to create the wanted mask.If it is an empty string, will pick up the mask from the Calcam calibration. 

 
        time_input/frame input : array of two values [t_start, t_end]
            can choose to specify the time or frames on which to do the inversion. 
            if left at none, will treat the whole video
            if both specified, will take time_input over frame_input
            NB : time input is in milliseconds
        dr_grid/dz_grid : float, specify the step of the grid in both directions (in meter)
        symetry : string
            hypothesis on the emissivity uniformity. Can be set to 'toroidal' or 'magnetic
        n_polar : int
            number of toroidal points in 1 revolution for magnetic lines(only relevant for magnetic symmetry. Set to 1 for toroidal symmetry)
            Magnetic reconstruction on Compass has a 0.2 degrees precision for the toroidal angle (n_polar = 1800)
        Verbose : Boolean
            set to True if you want to have verification steps in the inversion and raytracing progress 
            (will show plots of the different process to check if they are no glaring issues)
        real_inv_flag : Boolen
            set to True if you want to have invert the video, set to 0 if you only want the raytracing
        synth_inv_flag : Boolen
            set to True if you want to have a random synthetic inversion 
            with a bit of noise to see if the inversion method holds well
        name_material : String 
            parameters to specify the model for the reflection of the walls
        inversion_method : String
            Choose the inversion method (refer to the inversion_module for options)
        inversion_parameter : dictionnary
            dictionnary for additionnal inversion parameters that are specific to the inversion method(refer to the inversion_module for options)
        path_CAD : string, 
            the string should lead to a zip file containing the CAD model, or a .ccm file.Can be left on None if you want to use just the wall coordinates to create the wall object. (more accurate but slower)
        variant : string
            specify the variant of model in the calcam CADmodel given by path_CAD. Default is 'Default'
        path_vid : string, 
            path to the video folder, can be left on None if working on Compass. Can put the path of one image, in png format (full name with extension .png)
        params_fit : either 'vid', 'camera', 'mask' or None (default)
            The size of the video/camera/mask might be different. This allows which one should dictate the final dimensions.
            Will crop the others or extend them with 0 to fit the final size
        dict_transfert_matrix : dictionnary 
        dictionnary for additionnal inversion parameters that are specific for each machine or for a shot
            Compass :
                grid_precision_multiplier : int, powers of 2, default 1
                    If the uniformity hypothesis is on the magnetic field lines, the 3d grid on which the raytracing is done is not the same as the coordinates on which the field lines are calculated.
                    The 3d raytracing grid is set by default to have the same precision as the magnetic field lines, which may lead to errors.
                    The grid precision multiplier multiplies the number of points in each dimension R, Z (see n_polar for the angular precision) 
                    To ensure a fine enough mesh, raise the grid_precision_multiplier to fully capture on the 3d grid the position of the field lines.

                variant_mag : string
                    the variant of the equilibrium reconstruction given by pleque (default is '')
                revision : int
                    the number of the revision of the equilibrium reconstruction given by pleque (default is 1)
                phi_grid : scalar
                    angle chosen for the emissivity 2D RZ-plane (in degrees) 

                
        gaussian filter : scalar, default 0
            if not 0, applies a gaussian filter to the video frames, with its standard deviation equals to gaussian filter
            scipy.ndimage.gaussian_filter
        median filter : integer, default 0
            if not 0, applies a time median filter to the video, the size of its windows given by median filter


            
            Outputs :
            NB : some results are saved automatically
            
            transfert_matrix : csr matrix
            Output of the raytracing, gives the geomatry matrix that can create an image from an emissivity profile

            vid : 3D array (time, X, Y)
            Video treated in the inversion (gaussian filter, median filter, etc..)
            NB : Frames are saved in the (1st dimension : Horizontal, 2nd dimension: Vertical) format, contrary to python (vertical then horizontal) format
            For exemple, to plot using the imshow command, you first have to transpose the frame; plt.ismhow(vid[0, :, :].T)

            images_retrofit_full : 3D array (time, X, Y)
            Synthetic images reconstructed from the emissivity profiles found from inversion
            Same remark as previous one.

            inversion_results_full : 3D array (time, R, Z)
            emissivity plane from inversion, in the R, Z format
            NB : the nodes are saved in increasing in R and Z values, careful in using imshow to invert the Y axis of the plot as its default places the first element of the Y axis in the top

            inversion_results_thresolded_full : 3D array (time, R, Z)
            Same as previous one, but after a denoising effect has been applied on the result

            pixels : 1D array
                indicates in the 2D grid of the pixels, which one are used/see the plasma

            noeuds 1D array
                indicates in the 2D grid of the emissivity nodes, which one are seen by the camera

            
            dr_grid, dz_grid : float
                final values of the steps of the emissivity grid.
            nb_noeuds_r, nb_noeuds_z : int
                number of points in each dimensions

            RZwall, R_wall, Z_wall : arrays
                coordinates of the wall
            world : World (from Cherab  module) :
                environment in which the raytracing was done

            full_wall : Mesh (from cherab module) :
                3D boundaries of the wall

            R_noeud, Z_noeud : 1D array
                coordinates of the emissivity grid
            mask_pixel, mask_noeud : 2D arrays
                mask on respectively the camera and the emissivity grid

    """
    #import functions
    kwargs = locals()
    #create folder where to save the outputs
    start_time_get_parameters = time.time()
    #import input parameters
    name_calib = get_name(path_calibration)
    name_wall = get_name(path_wall)
    name_machine = get_name_machine(machine)
    grid_precision_multiplier = dict_transfert_matrix.get('grid_precision_multiplier')
    if not path_vid:
        path_vid = 'None'
    #import camera calibrations
    world = World()
    real_pipeline = RayTransferPipeline2D()


    realcam = load_camera(path_calibration)
    if name_machine == 'WEST':
        check_shot_and_video(nshot, path_vid)
    mask_pixel, name_mask = load_mask(path_calibration, path_mask)

    #load image data
    ##### handle the cases for treating simple images
    [name_vid, ext] = os.path.splitext(path_vid)

#check surface
    name_material, wall_material = recognise_material(name_material)
    if path_CAD:
        type_wall = 'CAD'
        name_CAD = os.path.splitext(os.path.basename(path_CAD))[0]
        if not os.path.exists(path_CAD):
            # if os.path.dirname(path_CAD) != main_folder_CAD:
            path_CAD = main_folder_CAD + path_CAD

    else:
        type_wall = 'coords'
        name_CAD = 'coords'


    if not phi_grid:
        n_polar = 1
    else:
        n_polar = n_polar

    main_name_parameters = ( 
        'mask_' + name_mask + 
        '_calibration_' + name_calib+ 
        '_wall_' + name_CAD + 
        '_material_' + name_material + 
        '_dz_' + str(int(dz_grid*1e3)) + 
        '_dr_' + str(int(dr_grid*1e3)) + 
        '_decimation_' + str(decimation)
    )
    pairs_str = "_".join(f"{k}_{v}" for k, v in dict_transfert_matrix.items())
    secondary_name_parameters = (
        'cropping_' + str(param_fit) + 
        '_reflexion_dict_' + pairs_str +
        '_npolar_' +str(n_polar)
    )
#read path of folders
    

    name_parameters = main_name_parameters + '/' + secondary_name_parameters
    #prepare path and folder to save the data
    # path of parameters and inversion
    folder_parameters = main_folder_processing + name_machine + '/' + str(nshot)+  '/' + name_parameters + '/'
    path_parameters = folder_parameters + 'parameters.npz'
    path_transfert_matrix = folder_parameters +  'transfert_matrix.npz'
    os.makedirs(folder_parameters, exist_ok=True)


    # path inverse matrix
    name_dict_parameter_inversion = get_name_parameters_inversion(inversion_parameter)
    name_inversion = 'inversion_method_' +str(inversion_method)+ '_inversion_parameter_' + name_dict_parameter_inversion
 
    folder_inverse_matrix = folder_parameters + name_inversion + '/'
 
    os.makedirs(folder_inverse_matrix, exist_ok = True)


    # path of inversion results
    pairs_str = "_".join(f"{k}_{v}" for k, v in dict_denoising.items())
    name_parameters_vid = (
        'denoising' + pairs_str 
    )
    name_vid = os.path.basename(path_vid)
    folder_save_result = folder_inverse_matrix + name_parameters_vid + '/'
    os.makedirs(folder_save_result, exist_ok = True)
    name_resultat_inversion =  'vid_' + name_vid + '_frames_'+ str(frame_input[0]) + '_'+ str(frame_input[1])
    save_resultat_inversion = folder_save_result + name_resultat_inversion


    # path images
    folder_images_reelles =  folder_save_result + 'images_reelles/' 
    os.makedirs(folder_images_reelles, exist_ok=True)
    name_images_reelles = folder_images_reelles + 'vid_' + str(name_vid)

    folder_images_synthetiques = folder_save_result + 'images_synthetiques/' 
    os.makedirs(folder_images_synthetiques, exist_ok=True)
    name_images_synthetiques = folder_images_synthetiques + 'vid_' + str(name_vid)

#loading wall coordinates
    try:
        fwall = loadmat(path_wall)
        RZwall = fwall['RZwall']

    except:
        RZwall = np.load(path_wall, allow_pickle=True)
    #check that the last element is the neighbor of the first one and not the same one
    if(RZwall[0]==RZwall[-1]).all():
        RZwall = RZwall[:-1]
        print('erased last element of wall')
    #check that the wall coordinates are stocked in a counter clockwise position. If not, reverse it
    R_mid = (np.max(RZwall[:, 0])+np.min(RZwall[:, 0]))/2
    Z_mid = (np.max(RZwall[:, 1])+np.min(RZwall[:, 1]))/2
    theta1 = np.arctan2(RZwall[0, 1]-Z_mid, RZwall[0, 0]-R_mid)
    theta2 = np.arctan2(RZwall[1, 1]-Z_mid, RZwall[1, 0]-R_mid)
    print(RZwall[:5])
    # if theta2-theta1<0:
    #     RZwall = RZwall[::-1]
    #     print('wall reversed')
    print(RZwall[:5])
    R_wall = RZwall[:, 0]
    Z_wall = RZwall[:, 1]

    if os.path.exists(save_resultat_inversion + '.mat'):
        dict_results = loadmat(save_resultat_inversion + '.mat')
        inversion_results_full = dict_results['inversion_results_full']
        inversion_results_thresolded_full = dict_results['inversion_results_thresolded_full']
        images_retrofit_full = dict_results['images_retrofit_full']
        mask_pixel = dict_results['mask_pixel']
        mask_noeud = dict_results['mask_noeud']
        pixels = dict_results['pixels']
        noeuds = dict_results['noeuds']
        frame_input = dict_results['frame_input']
        t_start = dict_results['t_start']
        t0 = dict_results['t0']
        t_inv = dict_results['t_inv']
        # path_transfert_matrix = dict_results['path_transfert_matrix']
        # path_parameters = dict_results['path_parameters']
        path_vid = dict_results['path_vid']
        vid = dict_results['vid']
        transfert_matrix = load_npz(path_transfert_matrix)
        path_parameters_save, ext = os.path.splitext(path_parameters)
        parameters = loadmat(path_parameters_save)
        pixels = parameters['pixels']
        noeuds = parameters['noeuds']
        
        nb_noeuds_r = parameters['nb_noeuds_r']
        nb_noeuds_z = parameters['nb_noeuds_z']
        world = World()
        full_wall = axisymmetric_mesh_from_polygon(RZwall)
        R_noeud = parameters['R_noeud']
        Z_noeud = parameters['Z_noeud']
        dr_grid = np.mean(np.diff(R_noeud))
        dz_grid = np.mean(np.diff(R_noeud))
        # mask_pixel = parameters['mask_pixel']
        # mask_noeud = parameters['mask_noeud']
        pdb.set_trace()
        transfert_matrix, pixels, noeuds, mask_pixel, mask_noeud = prep_inversion(transfert_matrix, mask_pixel, mask_noeud,pixels, noeuds, inversion_parameter, R_noeud, Z_noeud)
        return transfert_matrix, vid, images_retrofit_full, inversion_results_full, inversion_results_thresolded_full, pixels, noeuds, dr_grid, dz_grid, nb_noeuds_r, nb_noeuds_z, RZwall, R_wall, Z_wall, world, full_wall, R_noeud, Z_noeud, mask_pixel, mask_noeud

    if ext == '.png':
        vid, len_vid,image_dim_y,image_dim_x, fps, frame_input = get_img(path_vid = path_vid, nshot = nshot)
        t0 = 0
        tstart = 0
        tinv = 0
    ####
    else: #load videos
        vid, len_vid,image_dim_y,image_dim_x, fps, frame_input, name_time, t_start, t0, t_inv = get_vid(time_input, frame_input, path_vid = path_vid, nshot = nshot, inversion_parameter=dict_denoising)
    #load mask, check size
    utility_functions.save_array_as_img(vid, main_folder_image + 'image_vid_mid.png')
    utility_functions.save_array_as_gif(vid, gif_path=main_folder_image + 'quickcheck_cam.gif', num_frames=100, cmap='gray')
    mask_pixel = mask_pixel.T
    vid = np.swapaxes(vid, 1,2)
    utility_functions.save_array_as_img(vid, main_folder_image + 'image_vid_mid_rotated.png')

    if machine == 'WEST':
        vid = np.swapaxes(vid, 1,2)
        if image_dim_y == mask_pixel.shape[0] and param_fit==None:
            vid = np.swapaxes(vid, 1,2)
        
        utility_functions.save_array_as_gif(vid, gif_path=main_folder_image + 'quickcheck_cam_after_rotation.gif', num_frames=100, cmap='viridis')

    realcam, mask_pixel, vid = fit_size_all(realcam, mask_pixel, vid, param_fit)
    utility_functions.save_array_as_gif(vid, gif_path=main_folder_image + 'quickcheck_cam_after_rezizing.gif', num_frames=100, cmap='viridis')


    if decimation != 1:
        realcam, mask_pixel, vid = reduce_camera_precision(realcam, mask_pixel, vid, decimation = decimation)
        utility_functions.save_array_as_gif(vid, gif_path=main_folder_image + 'quickcheck_cam_after_decimation.gif', num_frames=100, cmap='viridis')

    # corner_min_y, corner_max_y,corner_min_x, corner_max_x = FULL_MASK(mask,realcam.pixel_origins.shape[0],realcam.pixel_origins.shape[1],0,path_mask)
    """
    if image_dim_y!= realcam.pixel_origins.shape[0]:
        realcam = VectorCamera( realcam.pixel_origins[corner_min_y:corner_max_y,corner_min_x: corner_max_x] ,
                realcam.pixel_directions[corner_min_y:corner_max_y,corner_min_x: corner_max_x])
                """
#    realcam = VectorCamera( realcam.pixel_origins[corner_min_y:corner_max_y,corner_min_x: corner_max_x],realcam.pixel_directions[corner_min_y:corner_max_y,corner_min_x: corner_max_x])
#    vid = vid[0, corner_min_y:corner_max_y,corner_min_x: corner_max_x]
    realcam.frame_sampler=FullFrameSampler2D(mask_pixel)
    realcam.pipelines=[real_pipeline]
    realcam.parent=world
    realcam.pixel_samples = 100
    realcam.min_wavelength = 640
    realcam.max_wavelength = realcam.min_wavelength +1
    realcam.render_engine.processes = 16

    
    


    if os.path.exists(path_parameters):
        """
        f = h5py.File(name_transfert_matrix+'.hdf5', 'r')
        R_noeud = f["R_noeud"][()]
        Z_noeud = f["Z_noeud"][()]
        R_max_noeud = f["R_max_noeud"][()]
        R_min_noeud = f["R_min_noeud"][()]
        Z_max_noeud = f["Z_max_noeud"][()]
        Z_min_noeud = f["Z_min_noeud"][()]
        seuil = f["seuil"][()]
        #FL_MATRIX = f["FL_MATRIX"][()]
        #phi_max = f["phi_max"][()]
        #phi_min = f["phi_min"][()]
        dPhirad = f["dPhirad"][()]
        pixels = f["pixels"][()]
        noeuds = f["noeuds"][()]
        f.close()
        StepsInPhi = dPhirad*180/np.pi
        """
        [transfert_matrix, 
            pixels,
            noeuds, 
            nb_noeuds_r, 
            nb_noeuds_z, 
            R_max_noeud, 
            R_min_noeud, 
            Z_max_noeud, 
            Z_min_noeud, 
            R_noeud,
            Z_noeud, 
            mask_pixel, 
            mask_noeud,
            path_parameters_new] = load_transfert_matrix_and_parameters(path_parameters = path_parameters, path_transfert_matrix = path_transfert_matrix)
        if path_parameters_new != path_parameters:
            print('something went wrong with the loading of parameters')
            pdb.set_trace()
        
        #skips the loading of the walls, go straight to inversion
    else:

        #load wall models
        # check how the wall is described; either CAD or coord
        
        if path_CAD:
            try:
                full_wall, name_material = read_CAD_from_calcam_module(path_CAD, world, name_material, wall_material, variant = variant)
            except:
                full_wall, name_material = read_CAD(path_CAD, world, name_material, wall_material, variant = variant)
        else: 
            full_wall = axisymmetric_mesh_from_polygon(RZwall)
            full_wall.material = wall_material
            full_wall.parent = world
        #calculate transfert matrix
        [transfert_matrix, 
         pixels, 
         noeuds, 
         R_noeud, 
         Z_noeud, 
         nb_noeuds_r, 
         nb_noeuds_z, 
         mask_pixel, 
         mask_noeud] = get_transfert_matrix(mask_pixel, 
                                            realcam, 
                                            real_pipeline, 
                                            RZwall, 
                                            dr_grid, 
                                            dz_grid, 
                                            image_dim_y,
                                            image_dim_x, 
                                            world, 
                                            full_wall, 
                                            verbose, 
                                            path_transfert_matrix, 
                                            path_parameters, 
                                            symetry, 
                                            nshot,
                                            dict_transfert_matrix, 
                                            path_CAD,
                                            variant,
                                            phi_grid,
                                            grid_precision_multiplier = grid_precision_multiplier,
                                            n_polar=n_polar,
                                            t_inv = t_inv)
    end_time_get_parameters = time.time()-start_time_get_parameters
    start_time_get_equilibrium = time.time()

    if name_machine == 'WEST':
        magflux = imas_west.get(nshot, 'equilibrium', 0, 1)
        
        t = pywed.tsbase(nshot, 'RIGNITRON', nargout = 1)[0][0]
        magflux.time = magflux.time-t

        derivative_matrix = get_derivative_matrix(inversion_method, R_noeud, Z_noeud, magflux)
        derivative_matrix = 0
    else:
        magflux = 0
        t = 0
        derivative_matrix = 0

    # if derivative_matrix:
    #     derivative_matrix = [[matrix[noeuds, :][:, noeuds] for matrix in sublist] for sublist in derivative_matrix]
    end_time_get_equilibrium = time.time()-start_time_get_equilibrium
    #make a synthetic inversion to check results
    print(mask_noeud)
    print(mask_noeud.shape)




    if synth_inv_flag:
        if inversion_method == 'Cholmod' or inversion_method == 'Mfr_Cherab':
            (node_full, 
                inv_image_synth, 
                inv_normed_synth, 
                inv_image_thresolded_synth, 
                inv_image_thresolded_normed_synth, 
                image_retrofit_synth, 
                image_full_noise, 
                image_full) = call_module2_function("synth_inversion", 
                                                    transfert_matrix, 
                                                    mask_pixel, 
                                                    mask_noeud,
                                                    pixels, 
                                                    noeuds, 
                                                    nb_noeuds_r, 
                                                    nb_noeuds_z,
                                                    R_noeud, 
                                                    Z_noeud, 
                                                    R_wall, 
                                                    Z_wall, 
                                                    inversion_method, 
                                                    derivative_matrix)
            
        
            synth_inv_plot = plot_results_inversion_synth(node_full, 
                                    inv_image_synth, 
                                    inv_normed_synth, 
                                    inv_image_thresolded_synth, 
                                    inv_image_thresolded_normed_synth, 
                                    image_retrofit_synth, 
                                    image_full_noise, 
                                    image_full,
                                    mask_pixel, 
                                    mask_noeud, 
                                    nb_noeuds_r, 
                                    nb_noeuds_z, 
                                    R_noeud, 
                                    Z_noeud, 
                                    R_wall, 
                                    Z_wall)
            synth_inv_plot.savefig(name_images_synthetiques + '.png')
            plt.close()
        else:
            (node_full, 
                inv_image_synth, 
                inv_normed_synth, 
                inv_image_thresolded_synth, 
                inv_image_thresolded_normed_synth, 
                image_retrofit_synth, 
                image_full_noise, 
                image_full) = synth_inversion(transfert_matrix, 
                                                    mask_pixel, 
                                                    mask_noeud,
                                                    pixels, 
                                                    noeuds, 
                                                    nb_noeuds_r, 
                                                    nb_noeuds_z,
                                                    R_noeud, 
                                                    Z_noeud, 
                                                    R_wall, 
                                                    Z_wall, 
                                                    inversion_method, 
                                                    derivative_matrix)
            
        
            synth_inv_plot = plot_results_inversion_synth(node_full, 
                                    inv_image_synth, 
                                    inv_normed_synth, 
                                    inv_image_thresolded_synth, 
                                    inv_image_thresolded_normed_synth, 
                                    image_retrofit_synth, 
                                    image_full_noise, 
                                    image_full,
                                    mask_pixel, 
                                    mask_noeud, 
                                    nb_noeuds_r, 
                                    nb_noeuds_z, 
                                    R_noeud, 
                                    Z_noeud, 
                                    R_wall, 
                                    Z_wall)
            synth_inv_plot.savefig(name_images_synthetiques + '.png')
            plt.close()
        # synth_inversion(transfert_matrix, mask,  pixels, noeuds, nb_noeuds_r, nb_noeuds_z,R_noeud, Z_noeud, R_wall, Z_wall, inversion_method, noise = 0.1, num_structures = 2, size_struct = 2, inversion_parameter = {"rcond": 1e-3}, derivative_matrix = derivative_matrix)
    #Does the inversion 
    start_time_get_inversion = time.time()

    if real_inv_flag:
        if inversion_method == 'Cholmod' or inversion_method == 'Mfr_Cherab':
            
            inversion_results, inversion_results_normed, inversion_results_thresolded, inversion_results_thresolded_normed, images_retrofit, mask_noeud, transfert_matrix = call_module2_function("inverse_vid", 
                                                    transfert_matrix, 
                                                    mask_pixel, 
                                                    mask_noeud,
                                                    vid,  
                                                    R_noeud, 
                                                    Z_noeud, 
                                                    inversion_method,
                                                    inversion_parameter, 
                                                    derivative_matrix
                                                    )
        else:

            [inversion_results, 
             inversion_results_normed, 
             inversion_results_thresolded, 
             inversion_results_thresolded_normed, 
             images_retrofit, 
             mask_noeud, 
             mask_pixel, 
             transfert_matrix
              ]= inverse_vid(transfert_matrix, 
                                                    mask_pixel, 
                                                    np.squeeze(mask_noeud),
                                                    pixels, 
                                                    noeuds,
                                                    vid,  
                                                    R_noeud, 
                                                    Z_noeud, 
                                                    inversion_method,
                                                    inversion_parameter, 
                                                    folder_inverse_matrix,
                                                    dict_denoising, 
                                                    derivative_matrix
                                                    )
            #save results
        start_time_get_save = time.time()
        idx = [0, inversion_results.shape[0]//2, inversion_results.shape[0]-1]
        for i in idx:
            image = vid[i, :, :]
            inv_image = inversion_results[i, :] 
            # inv_normed =  inversion_results_normed[i, :]
            # inv_image_thresolded = inversion_results_thresolded[i, :]
            # inv_image_thresolded_normed = inversion_results_thresolded_normed[i, :]
            # fig_results = plot_results_inversion(inv_image, inv_normed, inv_image_thresolded, inv_image_thresolded_normed, transfert_matrix, image, mask_pixel, mask_noeud, pixels, noeuds, R_wall, Z_wall, nb_noeuds_r, nb_noeuds_z, R_noeud, Z_noeud) 
            try:
                sep_map = separatrix_map(magflux, t0/1000 + i/fps)

            except:
                sep_map = None
            fig_results = plot_results_inversion_simplified(inv_image, transfert_matrix, image, mask_pixel, mask_noeud, pixels, noeuds, R_wall, Z_wall, nb_noeuds_r, nb_noeuds_z, R_noeud, Z_noeud, c_c = 3, cmap = 'viridis', norm = 'linear', magflux = sep_map)

            fig_results.savefig(name_images_reelles + 'frames'+   '%02d.png' % (frame_input[0]+ i), dpi=180)

            plt.close()


        # np.savez_compressed(path_resultat_inversion + '.npz',  
        #                     inversion_results = inversion_results, 
        #                     images_retrofit = images_retrofit, 
        #                     mask_pixel = mask_pixel, 
        #                     mask_noeud = mask_noeud, 
        #                     pixels = pixels,
        #                     noeuds = noeuds,
        #                     frame_input = frame_input)
        # if isinstance(nb_noeuds_r,  np.ndarray):
        #     nb_noeuds_r = nb_noeuds_r[0]
        # if isinstance(nb_noeuds_z,  np.ndarray):
        #     nb_noeuds_z = nb_noeuds_z[0]
        if mask_noeud.ndim ==3:
            inversion_results_full = np.zeros((inversion_results.shape[0], mask_noeud.shape[0], mask_noeud.shape[2]))
            inversion_results_thresolded_full = np.zeros((inversion_results_thresolded.shape[0], mask_noeud.shape[0], mask_noeud.shape[2]))
        else:
            inversion_results_full = np.zeros((inversion_results.shape[0], mask_noeud.shape[0], mask_noeud.shape[1]))
            inversion_results_thresolded_full = np.zeros((inversion_results_thresolded.shape[0], mask_noeud.shape[0], mask_noeud.shape[1]))

        images_retrofit_full = np.zeros((images_retrofit.shape[0], mask_pixel.shape[0], mask_pixel.shape[1]))
        try:
            for i in range(inversion_results.shape[0]):
                inversion_results_full[i, :, :] = reconstruct_2D_image(inversion_results[i, :], mask_noeud)
            for i in range(images_retrofit_full.shape[0]):
                images_retrofit_full[i, :, :] = reconstruct_2D_image(images_retrofit[i, :], mask_pixel)
            for i in range(inversion_results.shape[0]):
                inversion_results_thresolded_full[i, :, :] = reconstruct_2D_image(inversion_results_thresolded[i, :], mask_noeud)

        except:
            pdb.set_trace()

        ind_mid = inversion_results.shape[0]//2
        utility_functions.plot_image(inversion_results_full[ind_mid, :, :].T, origin = 'lower', )
        plt.savefig(name_images_reelles + 'inv_result.png')
        plt.close()

        utility_functions.plot_image(inversion_results_thresolded_full[ind_mid, :, :].T, origin = 'lower')
        plt.savefig(name_images_reelles + 'inv_result_thresholded.png')
        plt.close()

        utility_functions.plot_image(images_retrofit_full[ind_mid, :, :].T )
        plt.savefig(name_images_reelles + 'retrofit.png')
        plt.close()

        utility_functions.plot_image(vid[ind_mid, :, :].T)
        plt.savefig(name_images_reelles + 'vid.png')
        plt.close()

        #setting video and retrofit back into image convention (rows of the matrix being vertical component, first element of matrix being top left point of the image)
        # vid = np.swapaxes(vid, 1,2)
        # images_retrofit_full = np.swapaxes(images_retrofit_full, 1,2)
        path_parameters_save, ext = os.path.splitext(path_parameters)
        dict_results = dict(inversion_results_full = inversion_results_full,
                            inversion_results_thresolded_full = inversion_results_thresolded_full,
                            images_retrofit_full = images_retrofit_full, 
                            mask_pixel = mask_pixel, 
                            mask_noeud = mask_noeud, 
                            pixels = pixels,
                            noeuds = noeuds,
                            frame_input = frame_input,
                            t_start = t_start,
                            t0 = t0,
                            t_inv = t_inv, 
                            path_transfert_matrix = path_transfert_matrix,
                            path_parameters = path_parameters_save,
                            path_vid = path_vid,
                            vid = vid)
        try:
            savemat(save_resultat_inversion + '.mat', dict_results)
        except:
            pdb.set_trace()
        end_time_get_save = time.time()-start_time_get_save

    # creating gif animation with ImageMagick
        # os.system("convert -delay 10 -loop 0 images/extract_results*.png ray_transfer_mask_demo.gif")

    else:
        inversion_results = None
        end_time_get_save = 0
    
    
    end_time_get_inversion = time.time()-start_time_get_inversion

  


    full_wall = axisymmetric_mesh_from_polygon(RZwall)

    print('time for input = ', end_time_get_parameters)
    print('time for equilibrium = ', end_time_get_equilibrium)
    print('time for inversion = ', end_time_get_inversion)
    print('time for save = ', end_time_get_save)
    # f = open("name_inversion_results  +'frames'+ str(frame_input[0]) + '_'+ str(frame_input[1].txt", "w")
    # f.write(str(end_time_get_parameters))
    # f.write(str(end_time_get_equilibrium))
    # f.write(str(end_time_get_inversion))
    # f.write(str(end_time_get_save))
    # # f.close()
    # with open(path_parameters[:-4]+'.pk', 'wb') as f:
        
    #     pickle.dump(kwargs, f)
    #     f.close()
    return transfert_matrix, vid, images_retrofit_full, inversion_results_full, inversion_results_thresolded_full, pixels, noeuds, dr_grid, dz_grid, nb_noeuds_r, nb_noeuds_z, RZwall, R_wall, Z_wall, world, full_wall, R_noeud, Z_noeud, mask_pixel, mask_noeud




def FULL_MASK(mask,image_dim_y,image_dim_x,plot_image_reduction,path_remove_blind_pixels = 0):
    ## gives the indices of the reduced mask inside the full mask

    #
    # 1) Reduction of the frame size from the calibration frame size to the
    # experimental one
    # 2) Suppression of the blind or non-sense pixels
    #
    # Creation 2024-02-07
    
    
    ## 1) Reduction to the right frame size
    decalage_centre=0
    Nber_pixels_y,Nber_pixels_x =np.shape(mask)
    corner_min_y=int(image_dim_y/2-Nber_pixels_y/2-decalage_centre)
    corner_max_y=int((image_dim_y/2+Nber_pixels_y/2-decalage_centre))
    corner_min_x=int((image_dim_x/2-Nber_pixels_x/2))
    corner_max_x=int(image_dim_x/2+Nber_pixels_x/2)
    full_mask = np.zeros((image_dim_y, image_dim_x))
    
    
    ## 2) Pixels blind or non-sense
    try:
        f = loadmat(path_remove_blind_pixels)
        mask = f['mask']
    except:
        mask = np.load(path_remove_blind_pixels)

    #full_mask[corner_min_y:corner_max_y,corner_min_x: corner_max_x] = mask

    ## 3) Check how the image was reduced
    """
    if plot_image_reduction==1:
        full_image=np.zeros((Nber_pixels_y,Nber_pixels_x,3))
        indice_im_reduced_y, indice_im_reduced_x = np.meshgrid(im_reduced_y, im_reduced_x, indexing = 'ij')
        full_image[indice_im_reduced_y,indice_im_reduced_x]=los_direction_reduced_and_blind
        plt.figure()
        plt.imshow(full_image)
        plt.plot([corner_min_x, corner_min_x],[corner_min_y, corner_max_y],'w-')
        plt.plot([corner_min_x, corner_max_x],[corner_max_y, corner_max_y],'w-')
        plt.plot([corner_max_x, corner_max_x],[corner_min_y, corner_max_y],'w-')
        plt.plot([corner_min_x, corner_max_x],[corner_min_y, corner_min_y],'w-')
    
        plt.title('Check the image reduction')
        
        # Diplay points corresponding to LOS
        get_non_nan=np.isnan(los_direction_reduced_and_blind[:,:,1])
        [row,col]=np.where(get_non_nan==0)
    
        I1=min(row)+corner_min_y
        I2=max(row)+corner_min_y
        J1=min(col)+corner_min_x
        J2=max(col)+corner_min_x
    
        plt.plot(J1,I1,'yo')
        plt.plot(J1,I2,'yo')
        plt.plot(J2,I1,'yo')
        plt.plot(J2,I2,'yo')
        #scale_picture
        """
    return corner_min_y, corner_max_y,corner_min_x, corner_max_x

def get_vid(time_input, frame_input, path_vid = None, nshot = None, inversion_parameter = {}):

    from scipy import ndimage
  
    if path_vid != 'None':
        
        try:
            import pyMRAW
            images, data = pyMRAW.load_video(path_vid + '.chix')
            fps = data['Record Rate(fps)']
            image_dim_y = data['Image Height']
            image_dim_x = data['Image Width']
            NF = data['Total Frame']
        except:
            outdata = np.load(path_vid + '.npz', allow_pickle=True)
            images = outdata["images"]
            # data = outdata["data"]
            # fps = data.item().get('Record Rate(fps)')
            # image_dim_y = data.item().get('Image Height')
            # image_dim_x = data.item().get('Image Width')
            # NF = data.item().get('Total Frame')         
            # t_start = data.item().get('t_start')

            fps = outdata['fps'].item()

            try:
                skip_frame = outdata['skipFrame'].item()
            except:

                skip_frame = 1
            fps = fps/skip_frame
            image_dim_y = outdata['image_dim_y'].item()
            image_dim_x = outdata['image_dim_x'].item()
            NF = outdata['NF'].item()
            t_start = outdata['t_start'].item()

        if t_start is None:
            t_start = input('no time saved for video, please enter time of first frame of video')
            try:
                t_start = float(t_start)
            except:
                raise(ValueError('Could not assign time start to video'))
            utility_functions.add_variable_to_npz(path_vid + '.npz', 't_start', t_start)
        if time_input:
            frame_input = [int((time_input[0]-t_start)/fps),int((time_input[1]-t_start)/fps)]
        if frame_input:
            images = images[frame_input[0]:frame_input[1], :, :]
        else:
            frame_input = [0, NF-1]
        t0 = t_start+frame_input[0]/fps
    else:
        RIS_number = 3
        from . import RIS
        try:
            if not frame_input:
                out = RIS.get_info(nshot, RIS_number)
                frame_input =[0, out.daq_parameters.Images]
            if time_input:
                frame_start = int(RIS.time_to_frame(nshot, time_input[0], RIS = RIS_number)) 
                frame_stop = int(RIS.time_to_frame(nshot, time_input[1], RIS = RIS_number)) 
                frame_input = [frame_start, frame_stop]
            stamp = 'frame'
            flag, memory_required, available_memory = RIS.check_memory(nshot, frame_input,
                                                            RIS = RIS_number, stamp = stamp)
            flag = 0
            if not flag:
                video, frame_bounds = RIS.load(nshot, frame_input, RIS = RIS_number, stamp = stamp)
            else:
                raise Exception('fail to load video')
            dict_video = RIS.get_info(nshot, RIS = RIS_number, origin = 'RAW')
        except:
            RIS_number = 4

            if not frame_input:
                out = RIS.get_info(nshot, RIS_number)
                frame_input =[0, out.daq_parameters.Images]
            if time_input:
                frame_start = int(RIS.time_to_frame(nshot, time_input[0], RIS = RIS_number)) 
                frame_stop = int(RIS.time_to_frame(nshot, time_input[1], RIS = RIS_number)) 
                frame_input = [frame_start, frame_stop]
            stamp = 'frame'
            flag, memory_required, available_memory = RIS.check_memory(nshot, frame_input,
                                                            RIS = RIS_number, stamp = stamp)
            flag = 0
            if not flag:
                video, frame_bounds = RIS.load(nshot, frame_input, RIS = RIS_number, stamp = stamp)
            else:
                raise Exception('fail to load video')
            dict_video = RIS.get_info(nshot, RIS = RIS_number, origin = 'RAW')
        images = video.data
        Flip = dict_video['daq_parameters']['Flip']
        if Flip == 'Vertical':
            images = np.flip(images, 1)#flipping the video back to its original state
        elif Flip == 'Both':
            images = np.flip(images, 1)
            # images = np.flip(images, 2)
        else:
            raise(NameError('reshaping of raw data not supported. Update the get_vid function to handle this new case'))  
        image_dim_y = dict_video['daq_parameters']['FrameH']
        image_dim_x = dict_video['daq_parameters']['FrameW']
        fps = dict_video['daq_parameters']['FrameRate']
        images.dtype = 'int16'
        
        frame_input = frame_bounds
        if dict_video['daq_parameters']['TrigType'] == 'Start':
            t_start = dict_video['daq_parameters']['TriggerTime'] + dict_video['daq_parameters']['TriggerDelay']
        else:
            raise(NameError('Time trigger not recognized. Update the get_vid function to handle this new case'))
 

    sigma = inversion_parameter.get('sigma')
    sigma = sigma or 0
    if sigma:
        images = utility_functions.gaussian_blur_video(images, sigma=sigma)

    median = inversion_parameter.get('median')
    median = median or 0
    if median:
        images_median = ndimage.median_filter(images, size=(median,1,1), mode = 'nearest')
        images = images-images_median

    if time_input:
        name_time = 'time' + str(time_input[0]) + '_'   + str(time_input[1])
        t0 = time_input[0]
    else:
        name_time = 'frame' + str(frame_input[0]) + '_'   + str(frame_input[1])
        t0 = t_start+frame_input[0]/fps

    if 'reduce_frames' in  inversion_parameter.keys():
        reduce_frames = inversion_parameter['reduce_frames']
        images = utility_functions.average_along_first_row(images,reduce_frames)
        fps = fps/reduce_frames
    t_inv = t0+np.arange(images.shape[0])/fps
    return images, images.shape[0], image_dim_y, image_dim_x, fps, frame_input, name_time, t_start, t0, t_inv 
        


def get_name(path):
    path = path.split('/')
    if path[-1]:
        name = path[len(path)-1]
    else:
        name = path[len(path)-2]

    name_shortened = name.split('.')

    name_shortened = name_shortened[0]


    return name_shortened

def get_name_extenstion(path):
    path = path.split('/')
    name = path[len(path)-1]
    name_shortened = name.split('.')

    name_shortened = name_shortened[-1]


    return name_shortened

def get_transfert_matrix(mask, 
                         realcam, 
                         real_pipeline, 
                         RZwall, dr_grid, 
                         dz_grid, 
                         image_dim_y,
                         image_dim_x, 
                         world, 
                         full_wall, 
                         verbose, 
                         path_transfert_matrix, 
                         path_parameters, 
                         symetry, 
                         nshot, 
                         dict_transfert_matrix, 
                         path_CAD,
                         variant,
                         phi_grid = None,
                         grid_precision_multiplier = None,
                         n_polar = 1,
                         t_inv = None):
    """
    return transfert_matrix for west"
    """
    
    #get wall coordinates to get the cherab object
    if symetry == 'magnetic': #overwrites wall with the one saved in pleque
        t_pleque = t_inv[len(t_inv)//2] #choose middle time of the inversion for time of equilibrium
        t_pleque = t_pleque*1000 # converting in ms for pleque
        import pleque.io.compass as plq
        print("magnetic symmetry")
        revision_mag = dict_transfert_matrix.get('revision')
        revision_mag = revision_mag or 1
        variant_mag =  dict_transfert_matrix.get('variant_mag')
        variant_mag = variant_mag or ''
        eq = plq.cdb(nshot, t_pleque, revision = revision_mag, variant = variant_mag)
        RZwall = np.array([eq.first_wall.R,eq.first_wall.Z]).T
        RZwall =RZwall[:-1, :] 
        RZwall = RZwall[::-1]
        

    wall_limit = axisymmetric_mesh_from_polygon(RZwall)
    R_wall = RZwall[:, 0]
    Z_wall = RZwall[:, 1]


    visible_pix = np.where(mask) 
    pos_camera = realcam.pixel_origins[visible_pix[0][0]  , visible_pix[1][0]]
    pos_camera = np.array([pos_camera.x,pos_camera.y, pos_camera.z] )
    pos_camera_RPHIZ = utility_functions.xyztorphiz(pos_camera)
    # R_max_noeud = pos_camera_RPHIZ[0] 
    # R_min_noeud = pos_camera_RPHIZ[0] 
    # Z_max_noeud = pos_camera_RPHIZ[2] 
    # Z_min_noeud = pos_camera_RPHIZ[2] 
    # R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, phi_max, phi_min = optimize_boundary_grid(realcam, world, R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud)


    R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, phi_max, phi_min, RPHIZ = optimize_grid_from_los(realcam, world, pos_camera_RPHIZ)
    R_max_noeud = R_max_noeud#+dr_grid
    R_min_noeud = R_min_noeud#-dr_grid
    Z_max_noeud = Z_max_noeud#+dz_grid
    Z_min_noeud = Z_min_noeud#-dz_grid
    # R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud =[0.9, 0.0069587420250745,0.6801025165889444, -0.4519975310227926]


    if verbose:
        fig = utility_functions.plot_cylindrical_coordinates(RPHIZ)
        fig = utility_functions.plot_line_from_cylindrical(pos_camera_RPHIZ, RPHIZ[0,0,:], fig, color = 'blue', label = 'point [0,0]')
        fig = utility_functions.plot_line_from_cylindrical(pos_camera_RPHIZ, RPHIZ[-1,0,:], fig, color = 'red', label = 'point [-1,0]')
        fig = utility_functions.plot_line_from_cylindrical(pos_camera_RPHIZ, RPHIZ[0,-1,:], fig, color = 'green', label = 'point [0,-1]')
        fig = utility_functions.plot_line_from_cylindrical(pos_camera_RPHIZ, RPHIZ[-1,-1,:], fig, color = 'yellow', label = 'point [-1,-1]')
        plt.savefig(main_folder_image + 'images line of sight and wall')
    extent_RZ =[R_min_noeud, R_max_noeud, Z_min_noeud, Z_max_noeud] 
    nb_noeuds_r = int((R_max_noeud-R_min_noeud)/dr_grid)
    nb_noeuds_z = int((Z_max_noeud-Z_min_noeud)/dz_grid)
    realcam
    cell_r, cell_z, grid_mask, cell_dr, cell_dz = get_mask_from_wall(R_min_noeud, R_max_noeud, Z_min_noeud, Z_max_noeud, nb_noeuds_r, nb_noeuds_z, wall_limit, dict_transfert_matrix)
    # The RayTransferCylinder object is fully 3D, but for simplicity we're only
    # working in 2D as this case is axisymmetric. It is easy enough to pass 3D
    # views of our 2D data into the RayTransferCylinder object: we just ues a
    # numpy.newaxis (or equivalently, None) for the toroidal dimension.
    grid_mask = grid_mask[:, np.newaxis, :]
    if symetry =='magnetic':
        n_polar = n_polar
    else:
        n_polar = 1
    grid_mask = grid_mask>0
    RZ_mask_grid = np.copy(grid_mask)
    grid_mask = np.tile(grid_mask, (1, n_polar, 1))
    # num_cells = vertex_mask.sum()
    num_points_rz = nb_noeuds_r*nb_noeuds_z


    # R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, phi_min, phi_max = optimize_grid(R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, realcam, phi_grid, RZwall)
    
    #recalculate the extremities of the grid; the grid starts at (r_min, z_min) and its last point is (r_max-dr_grid, z_max-dz_grid)

   
    if symetry =='magnetic':
                
        start = time.time()
        
        FL_MATRIX, dPhirad, phi_min, phi_mem = FL_lookup(eq, phi_grid, cell_r, cell_z, phi_min, phi_max, 0.5e-3, 0.2)
        end = time.time()

        elapsed = end - start
        print(f"Magnetic field lines calculation : {elapsed:.3f} seconds")
        grid_precision_multiplier = grid_precision_multiplier or 1
        cell_r_precision, cell_z_precision, grid_mask_precision, cell_dr_precision, cell_dz_precision = get_mask_from_wall(R_min_noeud, R_max_noeud, Z_min_noeud, Z_max_noeud, nb_noeuds_r*grid_precision_multiplier, nb_noeuds_z*grid_precision_multiplier, wall_limit, dict_transfert_matrix)
        grid_mask_precision = grid_mask_precision>0
        plasma = RayTransferCylinder(radius_outer = R_max_noeud, 
                                     height= Z_max_noeud-Z_min_noeud, 
                                     n_radius = nb_noeuds_r*grid_precision_multiplier, 
                                     n_height = nb_noeuds_z*grid_precision_multiplier, 
                                     radius_inner = R_min_noeud,  
                                     transform=translate(0., 0., Z_min_noeud), 
                                     n_polar = n_polar, 
                                     period = 360)
        # turning angle back into degrees
        phi_min = phi_min*180/np.pi
        phi_max = phi_max*180/np.pi
        # seuil = np.sqrt((plasma.material.dr*grid_precision_multiplier)**2+(plasma.material.dz*grid_precision_multiplier)**2)*3
        seuil = np.sqrt((np.mean(np.diff(cell_r)))**2+(np.mean(np.diff(cell_z)))**2)
        
        plasma.voxel_map[:] = -1 #setting all nodes to blind 
        for nphi in range(n_polar):
            if nphi*plasma.material.dphi>phi_max or nphi*plasma.material.dphi<phi_min:
                plasma.voxel_map[:, nphi, :] = -1
                print('phi out of range')
            else:
                ind_phi_closest = np.round((nphi*plasma.material.dphi-phi_min)/(dPhirad*180/np.pi)).astype('int')
                #print('phi in range')
                for i in range(nb_noeuds_r*grid_precision_multiplier):
                    for j in range(nb_noeuds_z*grid_precision_multiplier):
                        noeud_r = R_min_noeud + plasma.material.dr*i
                        noeud_z = Z_min_noeud + plasma.material.dz*j

                        if noeud_r != cell_r_precision[i] or noeud_z != cell_z_precision[j]:
                            pdb.set_trace()

                            raise(ValueError('careful, grid ill defined'))
                        pointrz = Point3D(noeud_r, 0, noeud_z)
                        if grid_mask_precision[i,j]:
                        # if wall_limit.contains(pointrz):
                            # print(pointrz)
                            # print('go')
                            dist = np.sqrt((noeud_r-FL_MATRIX[:, 0, ind_phi_closest])**2+(noeud_z-FL_MATRIX[:, 1, ind_phi_closest])**2)
                            argmin = np.nanargmin(dist)
                            minlos = dist[argmin]

                            if minlos < seuil:
                                plasma.voxel_map[i, nphi, j] = argmin
                            else:
                                #set the element of the grid to a virtual node (not related to a position R, Z of the 2D map) for debugging
                                plasma.voxel_map[i, nphi, j] = num_points_rz #careful of indexing, last real point of voxel map is num_points_rz-1
                                # pdb.set_trace()
                                # raise(ValueError('element of plasma too far from calculated magnetic lines'))
                                # plasma.voxel_map[i, nphi, j] = -1
                        else:
                            plasma.voxel_map[i, nphi, j] = -1
        print(np.max(plasma.voxel_map))
        # plasma = RayTransferCylinder(R_max_noeud, nb_noeuds_z*dz_grid, nb_noeuds_r*grid_precision_multiplier, nb_noeuds_z*grid_precision_multiplier, radius_inner = R_min_noeud,  parent = world, transform=translate(0., 0., Z_min_noeud), n_polar = n_polar, period = 360, voxel_map = plasma.voxel_map)
        # plasma.voxel_map[~grid_mask] = -1 
        # pdb.set_trace()
        plasma2 = RayTransferCylinder(
            radius_outer=R_max_noeud,
            radius_inner=R_min_noeud,
            height=nb_noeuds_z*dz_grid,
            n_radius=nb_noeuds_r*grid_precision_multiplier, 
            n_height=nb_noeuds_z*grid_precision_multiplier,  
            n_polar=n_polar,
            mask = grid_mask,
            voxel_map = plasma.voxel_map,
            period = 360,
            parent = world,
            transform=translate(0, 0, Z_min_noeud)
        )
        if verbose:
            ind_line = plasma.bins//2

    elif symetry == 'toroidal':
        
        plasma2 = RayTransferCylinder(
        radius_outer=cell_r[-1],
        radius_inner=cell_z[0],
        height=cell_z[-1] - cell_z[0],
        n_radius=nb_noeuds_r, n_height=nb_noeuds_z, 
        mask=grid_mask, n_polar=n_polar,
        parent = world,
        transform=translate(0, 0, cell_z[0])
    )
            
    else:
        raise(NameError('unrecognized symetry, write toroidal or magnetic'))
    if verbose:
        plt.figure()
        plt.imshow(np.sum(plasma.voxel_map, 1).T, extent= extent_RZ, origin = 'lower' )
        plt.show(block = False)
        plt.savefig(main_folder_image + '2D_voxel_map.png')

    #calculate inversion matrix
    print(plasma2.bins)
    print(num_points_rz)
    realcam.spectral_bins = plasma2.bins #set the grid to a size (NR, NZ) plus 1 extra node for elements of the grid too far from calculated field lines
    if realcam.spectral_bins >image_dim_y*image_dim_x:
        raise Exception("more nodes than pixels, inversion is impossible. Lower dr_grid or dz_grid")
    if realcam.spectral_bins >10000:
        print("careful, huge number of nodes")
    if verbose:
        compare_voxel_map_and_pleque(plasma2, FL_MATRIX, Z_min_noeud, phi_mem)

    if verbose:
        create_synth_cam_emitter(realcam, full_wall, R_wall, Z_wall, mask, path_CAD, variant = variant)
    
    


    # flag_integrity = verify_integrity(realcam, mask)
    # if not flag_integrity:
    #     raise(ValueError('mask of pixel inconsistent with either pixel los or origins'))
    realcam.observe()

    print('shape full transfert matrix = ' + str(real_pipeline.matrix.shape))
    flattened_matr = real_pipeline.matrix.reshape(real_pipeline.matrix.shape[0] * real_pipeline.matrix.shape[1], real_pipeline.matrix.shape[2])
    
    if flattened_matr.shape[1] > num_points_rz: 
        #some elements of the grid don't see the field lines. Checking if they are out of the field of view of the camera
        invisible_nodes = np.sum(flattened_matr, 0)[-1]
        if invisible_nodes>0:
            print('nodes not seen, choose bigger grid limits')
            # pdb.set_trace()
        flattened_matr = flattened_matr[:, :-1]
    print('flattened_matr shape', flattened_matr.shape)


    pixels,  = np.where(np.sum(flattened_matr, 1)) #sum over nodes
    noeuds,  = np.where(np.sum(flattened_matr, 0)) #sum over pixels
    #save results
    mask_pixel = np.zeros(flattened_matr.shape[0], dtype = bool)
    mask_pixel[pixels] = True
    mask_pixel = mask_pixel.reshape(real_pipeline.matrix.shape[0:2])
    # x, y, z = np.where(RZ_mask_grid)
    # x = x[noeuds]
    # y = y[noeuds]
    # z = z[noeuds]
    
    mask_noeud = np.zeros_like(RZ_mask_grid, dtype = bool)
    rows_noeud, indphi, cols_noeud = np.unravel_index(noeuds, mask_noeud.shape)

    mask_noeud[rows_noeud,indphi, cols_noeud] = True
    print('shape voxel_map ', plasma.voxel_map.shape)
    print('shape mask_noeud ', mask_noeud.shape)

    transfert_matrix = flattened_matr[pixels,:][:, noeuds]

    nb_visible_noeuds = len(np.unique(noeuds))
    nb_vision_pixel = len(np.unique(pixels))
    print('visible node = ' + str(nb_visible_noeuds) + 'out of ' + str(nb_noeuds_r*nb_noeuds_z))
    print('vision pixels = ' + str(nb_vision_pixel) + 'out of ' + str(real_pipeline.matrix.shape[0] * real_pipeline.matrix.shape[1]))

    transfert_matrix = csr_matrix(transfert_matrix)
    print(transfert_matrix.shape)
    pixels = np.squeeze(pixels)
    noeuds = np.squeeze(noeuds)

    print('shape reduced transfert matrix = ' + str(transfert_matrix.shape))
    plt.figure()
    plt.imshow(np.squeeze(mask_noeud).T, extent= extent_RZ, origin = 'lower' )
    plt.savefig(main_folder_image + '2D_map_nodes.png')
    if verbose:
        plt.show(block = False)


    try: 
        save_npz(path_transfert_matrix, transfert_matrix)
    except:
        print('save of transfert matrix failed')
        pdb.set_trace()
    try:
        dict_save_parameters = dict(pixels = pixels, 
                            noeuds = noeuds, 
                            nb_noeuds_r = nb_noeuds_r, 
                            nb_noeuds_z = nb_noeuds_z, 
                            R_max_noeud = R_max_noeud, 
                            R_min_noeud = R_min_noeud, 
                            Z_max_noeud = Z_max_noeud, 
                            Z_min_noeud = Z_min_noeud, 
                            R_noeud = cell_r,
                            Z_noeud = cell_z, 
                            mask_pixel = mask_pixel, 
                            mask_noeud= mask_noeud)
        np.savez_compressed(path_parameters, **dict_save_parameters)
        path_parameters_save ,ext = os.path.splitext(path_parameters)
        savemat(path_parameters_save + '.mat', dict_save_parameters)
    except:
        print('save of parameters failed')
        pdb.set_trace()
    
    return transfert_matrix, pixels, noeuds, cell_r, cell_z, nb_noeuds_r, nb_noeuds_z, mask_pixel, mask_noeud


                  

                                  
def project_to_camera(camera, coordinates_2d, toroidal_angle, radius):
    """
    Projects a set of 2D coordinates onto a 2D camera plane based on a toroidal angle
    after creating a 3D axisymmetric mesh.

    :param camera: VectorCamera object from Raysect.
    :param coordinates_2d: numpy array of shape (n, 2) representing the 2D coordinates.
    :param toroidal_angle: The toroidal angle in radians.
    :param radius: Radius for creating the axisymmetric mesh.
    :return: tuple containing:
        - numpy array of shape (m, 3) representing the 3D mesh coordinates.
        - numpy array of shape (m, 2) representing the 2D coordinates on the camera plane.
    :raises ValueError: If coordinates_2d are not a numpy array of shape (n, 2).
    """

    # Validate inputs
    if not isinstance(coordinates_2d, np.ndarray) or coordinates_2d.shape[1] != 2:
        raise ValueError("Coordinates must be a numpy array of shape (n, 2).")

    # Create the 3D axisymmetric mesh using Cherab
    coordinates_3d = axisymmetric_mesh_from_polygon(coordinates_2d, radius)

    # Define the rotation matrix for the toroidal angle around the Z-axis
    cos_angle = np.cos(toroidal_angle)
    sin_angle = np.sin(toroidal_angle)
    
    rotation_matrix = np.array([
        [cos_angle, -sin_angle, 0],
        [sin_angle, cos_angle, 0],
        [0, 0, 1]
    ])

    # Apply the rotation to the coordinates
    rotated_coordinates = np.dot(coordinates_3d, rotation_matrix.T)

    # Project the rotated coordinates onto the camera plane using the VectorCamera
    projected_coords_2d = []
    for coord in rotated_coordinates:
        # Transform the 3D point to the camera coordinate system
        point = camera.world_to_pixel(coord)
        if point is not None:
            projected_coords_2d.append(point)

    if not projected_coords_2d:
        raise ValueError("No coordinates were within the camera's field of view.")

    return coordinates_3d, np.array(projected_coords_2d)


def trace_field_line_toroidal(self, *coordinates, R: np.array = None, Z: np.array = None,
                         coord_type=None, in_first_wall=True, phi_end = None, phi_eval = None, **coords):
    """
    Return traced field lines starting from the given set of at least 2d coordinates.
Allows to choose the toroidal angle span on which to integrate
Outter field lines are limited by z planes given be outermost z coordinates of the first wall.
    :param coordinates:
    :param R:
    :param Z:
    :param coord_type:
    :param in_first_wall: if True the only inner part of field line is returned.
    :param phi_end : toroidal angle value at wich to stop the integration. 
    :param phi_eval : optional, allow to choose the values of the toroidal angle returned 
        along the integration of the magnetic field line. 
        Must be contained in the range [phi_start, phi_end], the minimum and maximum value of the toroidal angle.
        If none specified, allow the solver to choose these values. 
    :param coords:
    :return: list of coordinates 
    """
   
    import pleque.utils.field_line_tracers as flt
    from scipy.integrate import solve_ivp
    coords = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, **coords)

    res = []

    # XXXNOW
    coords_rz = coords.as_array(dim=2)

    

    dphifunc = flt.dphi_tracer_factory(self.B_R, self.B_Z, self.B_tor)

    r_lims = [np.min(self.first_wall.R), np.max(self.first_wall.R)]
    z_lims = [np.min(self.first_wall.Z), np.max(self.first_wall.Z)]

    for i in np.arange(len(coords)):
        
        if 1: #i==0 or i==len(coords)-1 or i == np.int(len(coords)/2):
            y0 = coords_rz[i]
            if coords.dim == 2:
                phi_start = 0
            else:
                phi_start = coords.phi[i]
               
            if phi_end is None:
                phi_end = phi_start 

            atol = 1e-6
            if self.is_xpoint_plasma:
                xp = self._x_point
                xp_dist = np.sqrt(np.sum((xp - y0) ** 2))
                atol = np.minimum(xp_dist * 1e-3, atol)

            if self._verbose:
                print('>>> tracing from: {:3f},{:3f},{:3f}'.format(y0[0], y0[1], phi_start))
                print('>>> atol = {}'.format(atol))

            # todo: define somehow sufficient tolerances
            sol = solve_ivp(dphifunc,
                            (phi_start, phi_end),
                            y0,
                            #                            method='RK45',
                            method='LSODA',
                            max_step=1e-2,  # we want high phi resolution
                            atol=atol,
                            rtol=1e-8,
                            t_eval = phi_eval
                            )

            if self._verbose:
                print("{}, {}".format(sol.message, sol.nfev))

            phi = sol.t
            R, Z = sol.y

            fl = self.coordinates(R, Z, phi)

            # XXX add condirtion to stopper
            if in_first_wall:
                mask = self.in_first_wall(fl)

                imask = mask.astype(int)
                in_idxs = np.where(imask[:-1] - imask[1:] == 1)[0]

                last_idx = False
                if len(in_idxs) >= 1:
                    # Last point is still in (+1)
                    last_idx = in_idxs[0]
                    mask[last_idx + 1:] = False

                Rs = fl.R[mask]
                Zs = fl.Z[mask]
                phis = fl.phi[mask]

                intersec = self.first_wall.intersection(fl, dim=2)
                if intersec is not None and len(in_idxs) >= 1:
                    R_last = Rs[-1]
                    Z_last = Zs[-1]

                    inter_idx = np.argmin((intersec.R - R_last) ** 2 + (intersec.Z - Z_last) ** 2)

                    Rx = intersec.R[inter_idx]
                    Zx = intersec.Z[inter_idx]
                    # last_idx = len(phis) - 1

                    coef = np.sqrt((Rx - fl.R[last_idx]) ** 2 + (Zx - fl.Z[last_idx]) ** 2 /
                                   (fl.R[last_idx + 1] - fl.R[last_idx]) ** 2 +
                                   (fl.Z[last_idx + 1] - fl.Z[last_idx]) ** 2)

                    phix = fl.phi[last_idx] + coef * (fl.phi[last_idx + 1] - fl.phi[last_idx])

                    Rs = np.append(Rs, Rx)
                    Zs = np.append(Zs, Zx)
                    phis = np.append(phis, phix)

                fl = self.coordinates(Rs, Zs, phis)

            res.append(fl)


    return res



def FL_lookup(eq, phi_grid, R_noeud, Z_noeud, phi_min, phi_max, IntegrationDistanceAlongLOS, dPhi = 1, psi_lim = 2.5):
   
    phi_grid = phi_grid/180*np.pi
    dPhirad = dPhi*np.pi/180 #conversion en radians
    phi_max = phi_max+dPhirad
    phi_min = phi_min-dPhirad
    dim_fl = int(np.ceil((phi_max-phi_min)/dPhirad))
    phi_mem = np.arange(phi_min, phi_max, dPhirad)
    ind_PHI = int(np.ceil((phi_grid-phi_min)/dPhirad))
    FL_MATRIX = np.zeros((len(R_noeud)*len(Z_noeud),  2, dim_fl))
    FL_MATRIX[:] = np.NaN
    RZ_noeud_R, RZ_noeud_Z = np.meshgrid(R_noeud, Z_noeud, indexing = 'ij')
    R_line = np.reshape(RZ_noeud_R, len(R_noeud)*len(Z_noeud))
    Z_line = np.reshape(RZ_noeud_Z, len(R_noeud)*len(Z_noeud))

    # chord = eq.coordinates(R = RZ_noeud_R, Z = RZ_noeud_Z, phi = np.ones(len(R_noeud)*len(Z_noeud))*phi_grid)
    # row_fl, col_fl = np.where(chord.psi_n<= psi_lim)

    
    chord = eq.coordinates(R = R_line, Z = Z_line, phi = np.ones(len(Z_line))*phi_grid)
    ind_psi = eq.in_first_wall(chord)

    # ind_psi = chord.psi_n<=psi_lim
    # ind_psi_ind = np.where(chord.psi_n<= psi_lim)[0]
    chord = eq.coordinates(R = R_line[ind_psi], Z = Z_line[ind_psi], phi = np.ones(len(Z_line[ind_psi]))*phi_grid)
    field_line_down = trace_field_line_toroidal(eq, chord, in_first_wall=False, phi_end = phi_min, phi_eval = np.flip(phi_mem[:ind_PHI], axis = 0))


    field_line_up = trace_field_line_toroidal(eq, chord, in_first_wall=False, phi_end = phi_max, phi_eval = phi_mem[ind_PHI:])




    FL_R = np.zeros((len(chord), len(phi_mem)))
    FL_Z = np.zeros((len(chord), len(phi_mem)))
    for i in range(len(field_line_down)):
        R_fl = np.concatenate((np.flip(field_line_down[i].R, axis=0), field_line_up[i].R))
        Z_fl = np.concatenate((np.flip(field_line_down[i].Z, axis=0), field_line_up[i].Z))
        phi_fl = np.concatenate((np.flip(field_line_down[i].phi, axis=0), field_line_up[i].phi))
        if not np.array_equiv(phi_fl, phi_mem):
            Raise("integration on toroidal angle incorrectly borned")
        chord = eq.coordinates(R = R_fl, Z = Z_fl, phi = phi_fl)
        #wall_mask = eq.in_first_wall(chord)
        #R_fl[np.invert(wall_mask)] = np.NaN
        #Z_fl[np.invert(wall_mask)] = np.NaN
        FL_R[i, :] = R_fl
        FL_Z[i, :] = Z_fl
    FL_MATRIX[ind_psi, 0, :] = FL_R
    FL_MATRIX[ind_psi, 1, :] = FL_Z
    return FL_MATRIX, dPhirad, phi_min, phi_mem
    
    
def optimize_grid(R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud,  realcam , phi_grid, RZwall):
    R_max_noeud = 0.764684705338163
    R_min_noeud = 0.6072972908368429
    Z_max_noeud = -0.054290360404222165
    Z_min_noeud = -0.22492253410696436
    phi_min = 1.9523258640362888
    phi_max = 3.0723448091615606
    return  R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, phi_min, phi_max


def get_img(path_vid, nshot):
    path_vid, ext = os.path.splitext(path_vid)

    try:
        import imageio.v3 as iio
        vid = iio.imread(path_vid + '.png')
    except:
        vid = plt.imread(path_vid + '.png')
        vid = np.mean(vid, 2)
    len_vid = 1
    image_dim_y,image_dim_x = vid.shape
    fps = 1
    frame_input = [0, 1]
    vid = vid[:, ::-1]
    vid = np.expand_dims(vid, axis = 0)
    return vid, len_vid,image_dim_y,image_dim_x, fps, frame_input


def visualize_scene(camera, wall, world):

    return camera


def get_derivative_matrix(inversion_method, R_noeud, Z_noeud, magflux):

    imid = len(magflux.time)//2
    interp = RegularGridInterpolator((magflux.interp2D.r[:, 0], magflux.interp2D.z[0, :]), magflux.interp2D.psi[imid, :, :], bounds_error = False)
    RZ_grid = RegularGrid(len(R_noeud), len(Z_noeud),[min(R_noeud), max(R_noeud)],[min(Z_noeud), max(Z_noeud)])
    #RZ are swapped because meshgrid indexing is different than the rest
    # RZ_grid = RegularGrid(len(Z_noeud), len(R_noeud),[min(Z_noeud), max(Z_noeud)],[min(R_noeud), max(R_noeud)])
    maggrid = interp((RZ_grid.center_mesh[1].T, RZ_grid.center_mesh[0].T))
    derivative_matrix = compute_aniso_dmats(RZ_grid, maggrid)
    return derivative_matrix
    

def serialize_data(data):
    """Serializes various data types into a pickle-compatible format."""
    if isspmatrix(data):  # Handle sparse matrices
        return {"type": "sparse", "data": pickle.dumps(data)}
    elif isinstance(data, np.ndarray):  # Handle NumPy arrays
        return {"type": "ndarray", "data": data}
    elif isinstance(data, (float, int, str)):  # Handle floats, ints, and strings
        return {"type": "primitive", "data": data}
    elif isinstance(data, list):
        return {"type": "list", "data": data}
    elif isinstance(data, dict):
        return {"type": "dict", "data": data}
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")
def deserialize_data(data):
    if data["type"] == "sparse":
        return pickle.loads(data["data"])
    elif data["type"] == "ndarray":
        return data["data"]
    elif data["type"] == "primitive":
        return data["data"]
    elif data["type"] == "list":
        return data["data"]
    elif data["type"] == "dict":
        return data["data"]
    else:
        raise ValueError(f"Unsupported data type: {data['type']}")

def call_module2_function(func_name, *args):
    serialized_args = [serialize_data(arg) for arg in args]
    input_data = {"func_name": func_name, "args": serialized_args}
    print(sys.version)
    # Debug: Check the size of serialized data
    serialized_input = pickle.dumps(input_data)
    print("Serialized Input Data Size:", len(serialized_input))
    # print(pickle.loads(serialized_input))
    # command = ["mamba", "run", "-n", "sparse_env", "python", "inversion_module.py"]
    command = ["bash", "-c", "source activate inversion_env && python inversion_module.py"]
    result = subprocess.run(
        command,
        input=serialized_input,  # Pass serialized data as binary input
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # result = subprocess.run(
    #     command,
    #     input=serialized_input,  # Pass serialized data as binary input
    #     stdout=None,
    #     stderr=None,
    # )
    # if result.stderr:
    #     raise RuntimeError(f"Error in subprocess: {result.stderr.decode().strip()}")
    # else:
    #     print("Subprocess completed successfully or stderr not captured.")

    if result.returncode != 0:
        raise RuntimeError(f"Error in subprocess: {result.stderr.decode().strip()}")

    return pickle.loads(result.stdout)
    

def reduce_camera_precision(camera, mask, vid, decimation =1):
    mask = downsample_with_avg(mask, decimation)
    # mask = mask[::decimation, ::decimation]
    vid_downsize = np.zeros((vid.shape[0], mask.shape[0], mask.shape[1]))
    for i in range(vid.shape[0]):
        vid_downsize[i, :, :] = downsample_with_avg(vid[i, :, :] , decimation)


    pixel_directions = downsample_with_avg(camera.pixel_directions, decimation)
    # pixel_directions = pixel_directions[::decimation, ::decimation]
    pixel_origins = camera.pixel_origins[::decimation, ::decimation]
    pixel_origins[np.invert(mask.astype('bool'))] = Point3D(np.NaN, np.NaN, np.NaN)

    camera = VectorCamera(pixel_origins, pixel_directions)

    return camera, mask, vid_downsize


def downsample_with_avg(matrix, block_size=4):
    # Get the shape of the matrix
    rows, cols = matrix.shape
    if matrix.dtype == 'int8':
        matrix = np.abs(matrix)
     # Create a result matrix initialized with the original values
    

    result = matrix[::block_size, ::block_size]
    
    
     # Loop over the matrix with the step of block_size (e.g., 4)
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            # Determine the block to average (handle edge cases where the block is smaller)
            block = matrix[i:i+block_size, j:j+block_size]
        
            # Check if the block is a full block (all 4x4 elements)
            if block.shape[0] == block_size and block.shape[1] == block_size:
                 # Compute the mean of the block
                block_mean = np.mean(block)
                 
                 # Assign the mean to the result matrix at the block's top-left corner
                if matrix.dtype == 'int8':
                    if block_mean == 1:
                        result[i//block_size, j//block_size] = block_mean
                    else:
                        result[i//block_size, j//block_size] = 0
                else:
                    result[i//block_size, j//block_size] = block_mean
     
    return result
 
def downsample_cam_position(matrix, block_size=4):
    # Get the shape of the matrix
    rows, cols = matrix.shape
    if matrix.dtype == 'int8':
        matrix = np.abs(matrix)
     # Create a result matrix initialized with the original values
    

    result = matrix[::block_size, ::block_size]
    
    
     # Loop over the matrix with the step of block_size (e.g., 4)
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            # Determine the block to average (handle edge cases where the block is smaller)
            block = matrix[i:i+block_size, j:j+block_size]
            
            # Check if the block is a full block (all 4x4 elements)
            if block.shape[0] == block_size and block.shape[1] == block_size:
                 # Compute the mean of the block
                block_mean = block.all()
                 
                 # Assign the mean to the result matrix at the block's top-left corner

            else:
                result[i//block_size, j//block_size] = block_mean
    
    return result
 


                        
# def plot_results_inversion_simplified(inv_image, transfert_matrix, image, mask, pixels, noeuds, R_wall, Z_wall, nb_noeuds_r, nb_noeuds_z, R_noeud, Z_noeud, c_c = 3):
    # extent = (R_noeud[0], R_noeud[-1], Z_noeud[0], Z_noeud[-1])
    # image_retrofit = transfert_matrix.dot(inv_image)
    # inv_image_full = reconstruct_2D_image(inv_image, noeuds, nb_noeuds_r, nb_noeuds_z)
    # image_retrofit_full = reconstruct_2D_image(image_retrofit, pixels, mask.shape[0], mask.shape[1])
    # figure_results =plt.figure()
    # #synthetic image
    # plt.subplot(2,2,1)
    # plt.imshow(image)
    # plt.colorbar()
    # plt.title('image')

    # #retro fit
    # plt.subplot(2,2,2)
    # plt.imshow(image_retrofit_full)
    # plt.colorbar()
    # plt.title('Retro fit')

    # #inversion
    # plt.subplot(2,2,3)
    # plt.imshow(inv_image_full, extent = extent, origin = 'lower')
    # plt.colorbar()
    # plt.plot(R_wall, Z_wall, 'r')
    # plt.xlabel('R [m]')
    # plt.ylabel('Z [m]')
    # plt.title('inversed image')
    # plt.show(block = False)
    # return figure_results

def write_csv_dictionnary(name_csv, dictionnary : dict, saving_folder = ''):
    import csv
    rows = [dict(zip(dictionnary.keys(), values)) for values in zip(*dictionnary.values())]
    if saving_folder:
        os.makedirs(saving_folder, exist_ok = True)

    with open(saving_folder + '%s.csv' % name_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=dictionnary.keys())
        writer.writeheader()
        writer.writerows(rows)
        f.close()


def read_stl(path_CAD = None, name_material = None):
    if not path_CAD:
        path_CAD = filedialog.askopenfilename(title="Select a file")
    import csv
    num_stl= len(os.listdir(path_CAD))
    if not name_material:
        name_material = 'absorbing_surface'
    material_stls = [name_material]*num_stl
    name_stls = []
    for stl in  (os.listdir(path_CAD)):
        name_stl = path_CAD + stl
        name_stls.append(name_stl)

    saving_folder = 'models_and_calibration/dictionnary_reflexion/'
    name_CAD = get_name(path_CAD)
    path_dict = saving_folder + name_CAD

    colors = get_consistent_colors(num_stl)
    dictionnary = {
        name: {"material": material, "color": color}
        for name, material, color in zip(name_stls, material_stls, colors)
    }

    write_csv_dictionnary(path_dict, dictionnary)
    return dictionnary

def read_csv_dictionnary(name_csv):
    import csv
    data = {}
    with open('%s.csv' % name_csv, "r") as f:
        reader = csv.DictReader(f)
    
        # Initialize lists for each column
        for field in reader.fieldnames:
            data[field] = []
        
        # Append values to corresponding lists
        for row in reader:
            for field in reader.fieldnames:
                data[field].append(row[field])
        f.close()
    return reader
def get_name_folder(path : str):
    if path[-1] == '/':
        return path
    else:
        path_split = path.split('/')
        path = '/'.join(path_split[:-1])
        path = path+'/'
        return path

def fit_size_all(camera, mask, vid, param_fit = None):
    if param_fit == 'mask':
        target_shape = mask.shape
    elif param_fit == 'camera':
        target_shape = camera.pixel_directions.shape
    elif param_fit == 'vid':
        target_shape = vid.shape[1:]
    else:
        if mask.shape != camera.pixel_directions.shape or mask.shape !=vid.shape[1:]:
            raise Exception('careful, discrepancy in elements shape')
        else:
            return camera, mask, vid
    mask = resize_matrix(mask, target_shape)
    pixel_directions = resize_matrix(camera.pixel_directions, target_shape)
    pixel_origins = resize_matrix(camera.pixel_origins, target_shape)
    camera = VectorCamera(pixel_origins, pixel_directions)
    vidnew = np.zeros((vid.shape[0], target_shape[0], target_shape[1]))
    for i in range(vid.shape[0]):
        vidnew[i, :, :] = resize_matrix(vid[i, :, :], target_shape)
    return camera, mask, vidnew 


def resize_matrix(matrix, target_shape):
    """
    Resize a matrix to match the target shape by cropping or zero-padding.
    Cropping and padding are centered along each axis.
    """
    target_rows, target_cols = target_shape
    current_rows, current_cols = matrix.shape

    # Determine row cropping/padding
    if current_rows > target_rows:
        start_row = (current_rows - target_rows) // 2
        matrix = matrix[start_row:start_row + target_rows, :]
    else:
        pad_before = (target_rows - current_rows) // 2
        pad_after = target_rows - current_rows - pad_before
        matrix = np.pad(matrix, ((pad_before, pad_after), (0, 0)), mode='constant')

    # Determine column cropping/padding
    if current_cols > target_cols:
        start_col = (current_cols - target_cols) // 2
        matrix = matrix[:, start_col:start_col + target_cols]
    else:
        pad_before = (target_cols - current_cols) // 2
        pad_after = target_cols - current_cols - pad_before
        matrix = np.pad(matrix, ((0, 0), (pad_before, pad_after)), mode='constant')

    return matrix

def get_consistent_colors(N):
    import matplotlib.colors as mcolors

    color_names = sorted(mcolors.CSS4_COLORS.keys())  # Sort to ens
    return color_names[:N]

def plot_results_inversion_synth(node_full, 
                                 inv_image, 
                                 inv_normed, 
                                 inv_image_thresolded, 
                                 inv_image_thresolded_normed, 
                                 image_retrofit, 
                                 image_full_noise, 
                                 image_full,
                                 mask_pixel, mask_noeud, 
                                 nb_noeuds_r, 
                                 nb_noeuds_z, 
                                 R_noeud, 
                                 Z_noeud, 
                                 R_wall, 
                                 Z_wall):
    c_c = 2
    inv_image_full = reconstruct_2D_image(inv_image, mask_noeud, nb_noeuds_r, nb_noeuds_z)
    inv_normed_full = reconstruct_2D_image(inv_normed, mask_noeud, nb_noeuds_r, nb_noeuds_z)    
    inv_image_thresolded_full = reconstruct_2D_image(inv_image_thresolded, mask_noeud, nb_noeuds_r, nb_noeuds_z)
    inv_image_thresolded_normed_full = reconstruct_2D_image(inv_image_thresolded_normed, mask_noeud, nb_noeuds_r, nb_noeuds_z)
    image_retrofit_full = reconstruct_2D_image(image_retrofit, mask_pixel, mask_pixel.shape[0], mask_pixel.shape[1])
    extent = (R_noeud[0], R_noeud[-1], Z_noeud[0], Z_noeud[-1])
    h = plt.figure()
    plt.subplot(2,4,1)
    plt.imshow(node_full.T, origin = 'lower', extent = extent)
    plt.colorbar()
    plt.plot(R_wall, Z_wall, 'r')

    plt.title('original profile')
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')


    #synthetic image
    plt.subplot(2,4,2)
    plt.imshow(image_full, origin = 'lower')
    plt.colorbar()
    plt.title('synthetic image')



    #synthetic image
    plt.subplot(2,4,3)
    plt.imshow(image_full_noise, origin = 'upper')
    plt.colorbar()
    plt.title('synthetic image (with noise)')

    #retro fit
    plt.subplot(2,4,4)
    plt.imshow(image_retrofit_full, origin = 'upper')
    plt.colorbar()
    plt.title('Retro fit')

    #inversion
    plt.subplot(2,4, 5)
    plt.imshow(inv_image_full, extent = extent, origin = 'lower')
    plt.colorbar()
    plt.plot(R_wall, Z_wall, 'r')
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.title('inversed image')
    
    #normed inversion
    plt.subplot(2,4, 6)
    plt.imshow(inv_normed_full, extent = extent, origin = 'upper')
    plt.colorbar()
    plt.plot(R_wall, Z_wall, 'r')
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.title('inversed image normalized')
    #normed inversion
    plt.subplot(2,4, 7)
    plt.imshow(inv_image_thresolded_full.T, extent = extent, origin = 'lower')
    plt.colorbar()
    plt.title('inversed image thresolded, c_c = '+ str(c_c))
    plt.plot(R_wall, Z_wall, 'r')
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    #normed inversion
    plt.subplot(2,4, 8)
    plt.imshow(inv_image_thresolded_normed_full.T, extent = extent, origin = 'upper')
    plt.colorbar()
    plt.title('inversed image thresolded and normalized, c_c = '+ str(c_c))
    plt.plot(R_wall, Z_wall, 'r')
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.tight_layout()

    plt.show(block = False)
    return h


def compare_reflection_results(nshot, 
                               path_calibration, 
                               path_mask, 
                               path_wall, 
                               vid,
                               time_input = None, 
                               frame_input = None, 
                               dr_grid = 1e-2, 
                               dz_grid = 1e-2, 
                               verbose = 0, 
                               inversion_method = 'lstsq', 
                               path_CAD = None, 
                               inversion_parameter = {"rcond" : 1e-3}, 
                               phi_grid = None, 
                               decimation = 1, 
                               param_fit = None
                               ):
    name_calib = get_name(path_calibration)
    name_mask = get_name(path_mask)
    name_wall = get_name(path_wall)

    if path_CAD:
        type_wall = 'CAD'
    else:
        type_wall = 'coords'
    
    name_transfert_matrix_absorb = (str(nshot) + 
        '_mask_' + name_mask + 
        '_calibration_' + name_calib+ 
        '_wall_' + type_wall + '_' + 'absorbing_surface' + 
        '_dz_' + str(int(dz_grid*1e3)) + 
        '_dr_' + str(int(dr_grid*1e3)) + 
        '_decimation_' + str(decimation) +
        'cropping' + str(param_fit)
    )
    name_transfert_matrix_refl = (str(nshot) + 
        '_mask_' + name_mask + 
        '_calibration_' + name_calib+ 
        '_wall_' + type_wall + '_' + 'tungsten' + 
        '_dz_' + str(int(dz_grid*1e3)) + 
        '_dr_' + str(int(dr_grid*1e3)) + 
        '_decimation_' + str(decimation) +
        'cropping' + str(param_fit)
    )


def readable_outputs(transfert_matrix, inversion_results, mask_noeud, mask_pixel, nb_noeuds_r, nb_noeuds_z):
    nb_images = inversion_results.shape[0]

    inversion_results_full = np.zeros((nb_images, nb_noeuds_r, nb_noeuds_z))
    images_retrofit_full = np.zeros((nb_images, mask_pixel.shape[0], mask_pixel.shape[1]))

    for i in range(nb_images):
        inversion_results_full[i, :, :] = reconstruct_2D_image(inversion_results[i, :], mask_noeud, mask_pixel, nb_noeuds_r, nb_noeuds_z)
        image_retrofit = transfert_matrix.dot(inversion_results[i, :])
        images_retrofit_full[i, :,:] = reconstruct_2D_image(image_retrofit, mask_pixel, mask_pixel,mask_pixel.shape[0], mask_pixel.shape[1])
    return inversion_results_full, images_retrofit_full


def load_results_raytracing(loading_folder, name_transfert_matrix):

    if loading_folder[-1] != '/':
        loading_folder = loading_folder+'/'
    loaded_raytracing = np.load(loading_folder + name_transfert_matrix + '.npz', allow_pickle = True)
    transfert_matrix = load_npz(loading_folder + name_transfert_matrix + '_transfert_matrix.npz')
    pixels = loaded_raytracing['pixels']
    noeuds = loaded_raytracing['noeuds']
    pixels = np.squeeze(pixels)
    noeuds = np.squeeze(noeuds)
    nb_noeuds_r = loaded_raytracing['nb_noeuds_r']
    nb_noeuds_z = loaded_raytracing['nb_noeuds_z']
    R_max_noeud = loaded_raytracing['R_max_noeud']
    R_min_noeud = loaded_raytracing['R_min_noeud']
    Z_max_noeud = loaded_raytracing['Z_max_noeud']
    Z_min_noeud = loaded_raytracing['Z_min_noeud']
    R_noeud = loaded_raytracing['R_noeud']
    Z_noeud = loaded_raytracing['Z_noeud']
    mask_noeud = loaded_raytracing['mask_noeud']
    mask_pixel = loaded_raytracing['mask_pixel']

    return (transfert_matrix,
        pixels, 
        noeuds,
        nb_noeuds_r,
        nb_noeuds_z,
        R_max_noeud,
        R_min_noeud,
        Z_max_noeud,
        Z_min_noeud,
        R_noeud,
        Z_noeud ,
        mask_noeud,
        mask_pixel,
    )

def load_inversion():
    OUT = loadmat(utility_functions.get_file('get mat file of inversion', path_root= '/Home/LF276573/Zone_Travail/Python/CHERAB/resultat_inversion/west/',full_path = 1))
    return OUT


def load_results_inversion(loading_folder, name_inversion_results):
    if loading_folder[-1] != '/':
        loading_folder = loading_folder+'/'
    loaded_results = np.load(loading_folder + name_inversion_results + '.npz', allow_pickle=True)
    inversion_results = loaded_results['inversion_results']
    mask_noeud = loaded_results['mask_noeud']
    mask_pixel = loaded_results['mask_pixel']
    frame_input = loaded_results['frame_input']
    return inversion_results, mask_pixel, mask_noeud, frame_input

def get_name_machine(machine):
    # get the name of the machine and assure it is the correct syntax in lowercase
    if machine.lower()== 'compass':
        machine = 'COMPASS'
    elif machine.lower()== 'west':
        machine = 'WEST'
    else:
        raise Exception('unrecognised machine')
    return machine

def extract_name_variables(input_str):
    PARTS =  input_str.split("/") 
    nshot = int(PARTS[0])
    parameters = PARTS[1]

    parts = parameters.split("_")  # Split the string by "_"

    i = 0
    j = 0
    regular_names = ['mask', 'calibration', 'wall', 'material', 'dz', 'dr', 'decimation', 'cropping', 'reflexion_dict']
    new_names = []
    while i < len(parts):
        name_part = ''
        if parts[i] == regular_names[j]:  # If it's a regular name, store its value
            j = j+1
            i = i+1
            while parts[i] != regular_names[j]:
                name_part = name_part+parts[i]
                i = i+1
            new_names[j-1] = name_part
    name_mask = new_names[0]
    name_calib = new_names[1]
    type_wall = new_names[2]
    name_material = new_names[3]
    dz_grid = int(new_names[4])
    dr_grid = int(new_names[5])
    decimation = int(new_names[6])
    dr_grid = int(new_names[5])
    param_fit = new_names[6]
    dict_transfert_matrix = new_names[7]




    name_parameters = ( 
        '_mask_' + name_mask + 
        '_calibration_' + name_calib+ 
        '_wall_' + type_wall + 
        '_material_' + name_material + 
        '_dz_' + str(int(dz_grid*1e3)) + 
        '_dr_' + str(int(dr_grid*1e3)) + 
        '_decimation_' + str(decimation) +
        '_cropping_' + str(param_fit) + 
        '_reflexion_dict_' + str(dict_transfert_matrix)
    )

    return name_parameters



def compare_transfert_matrix(nshot = None, path_reflexion = None, path_absorb = None, path_resultat_inversion = None, path_vid = None):
    if not path_reflexion:
        path_reflexion = utility_functions.get_file('get npz file of reflexion data', path_root= '/Home/LF276573/Documents/Python/CHERAB/transfert_matrix/west/',full_path = 1)
    path_reflexion, ext = os.path.splitext(path_reflexion)
    if not path_absorb:
        path_absorb = utility_functions.get_file('get npz file of absorb data', path_root='/Home/LF276573/Documents/Python/CHERAB/transfert_matrix/west/',full_path = 1)
    path_absorb, ext = os.path.splitext(path_absorb)
    if not path_resultat_inversion:
        path_resultat_inversion = utility_functions.get_file('get npz file of inversion results', path_root='/Home/LF276573/Documents/Python/CHERAB/resultat_inversion/west/',full_path = 1)
    path_resultat_inversion, ext = os.path.splitext(path_resultat_inversion)
    if not path_vid:
        path_vid = utility_functions.get_file('get video file', path_root = '/Home/LF276573/Documents/Python/CHERAB/videos/west/', full_path = 1)

    load_refl = np.load(path_reflexion + '.npz', allow_pickle=True)
    data_refl = {key: load_refl[key] for key in load_refl}
    data_refl['transfert_matrix'] = load_npz(path_reflexion + '_transfert_matrix.npz')
    load_absorb = np.load(path_absorb + '.npz', allow_pickle=True)
    data_absorb = {key: load_absorb[key] for key in load_absorb}
    data_absorb['transfert_matrix'] = load_npz(path_absorb + '_transfert_matrix.npz')

    inversion_results, mask_inversion, mask_noeud_inversion, frame_input = load_results_inversion(os.path.dirname(path_resultat_inversion), os.path.basename(path_resultat_inversion))
    frame_input = [frame_input[0], frame_input[1]]
    path_vid, ext = os.path.splitext(path_vid)
    if ext == '.png':
        vid, len_vid,image_dim_y,image_dim_x, fps, frame_input = get_img(path_vid, nshot)
        image = vid[0, :,:]
        image = np.flip(image, 1)
    else: #load videos
        vid, len_vid,image_dim_y,image_dim_x, fps, frame_input, name_time, t0 = get_vid(None, frame_input, path_vid = path_vid, nshot = nshot)
        image = vid[0, :,:]
        image = np.flip(image, 0)
        image = image.T
    inversion = inversion_results[0, :]

    inversion_full = reconstruct_2D_image(inversion, mask_noeud_inversion)
    inversion_refl, inversion_absorb, image_retrofit_full_absorb, image_retrofit_full_refl, diff = recreate_retrofit(image, inversion_full, data_refl, data_absorb)
    return image, image_retrofit_full_absorb, image_retrofit_full_refl, diff, inversion_full, nshot, path_absorb, path_reflexion, path_resultat_inversion, path_vid, data_refl, data_absorb

def call_function_in_environment(module_name, function_name, environment_name, args, kwargs):
    # Serialize the arguments and keyword arguments
    serialized_args = pickle.dumps(args)
    serialized_kwargs = pickle.dumps(kwargs)

    # Construct the command to call the function in the specified environment
    command = [environment_name, '-c', f'import pickle; import {module_name}; {function_name}(pickle.loads({serialized_args!r}), pickle.loads({serialized_kwargs!r}))']

    # Run the command using subprocess
    result = subprocess.run(command, capture_output=True, text=True, shell=True)

    # Check for errors
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return None

    # Return the output of the function
    return result.stdout.strip()


class separatrix_map:
    #class for easier plotting of magnetic map on WEST
    def __init__(self, magflux, time):

        idx = np.searchsorted(magflux.time, time, side="left")
        if idx == len(magflux.time):
            idx = idx-1
        elif idx == len(magflux.time)-1:
            idx = idx
        else:
            if np.abs(magflux.time[idx]-time)>np.abs(magflux.time[idx+1]-time):
                idx = idx+1
        self.time = magflux.time[idx]
        self.r = magflux.interp2D.r
        self.z = magflux.interp2D.z
        self.psi = magflux.interp2D.psi[idx, ...]
        #get psi norm
        # self.r0 = 
        # self.z0 = 
        # self.psi_norm = magflux.interp2D.psi[idx, :, :]/magflux.boundary.psi[idx]
        # self.rsep = magflux.boundary.r[idx, :] doesn't exist
        # self.zsep = magflux.boundary.z[idx, :] doesn't exist
        self.psisep = magflux.boundary.psi[idx]


def check_shot_and_video(nshot, path_vid):
    if str(nshot) not in path_vid:
        continuation = input('nshot not the same as path video, do you want to continue ? [y/n]')
        if continuation.lower() == 'y' or continuation.lower() == 'yes':
            return
        else:
            raise Exception('function aborted, nshot not consistent with path of video')
        


def check_intersection_wall_from_path(path_CAD, path_calibration):
    calcam_camera = np.load(path_calibration, allow_pickle=True)
    pixel_origins = calcam_camera['pixel_origins']
    pixel_directions = calcam_camera['pixel_directions']
    realcam = VectorCamera(pixel_origins.T, pixel_directions.T)
    Intersection = np.empty(realcam.pixels, dtype = 'object')
    world = World()
    realcam.parent = world
    from cherab.tools.observers.intersections import find_wall_intersection
    full_wall = read_CAD(path_CAD, world)
    for i in range(realcam.pixels[0]):
        for j in range(realcam.pixels[1]):
            if not np.isnan(realcam.pixel_origins[i, j][0]):
                try:
                    point, obj = find_wall_intersection(world, realcam.pixel_origins[i,j], realcam.pixel_directions[i, j])
                    Intersection[i, j] = obj.name
                except:
                    # raise Exception('pixel %02d, %02d does not intersect with wall' % (i, j))
                    print('pixel %02d, %02d does not intersect with wall' % (i, j))
    Intersection[Intersection==None] = 'None'

    return Intersection

def plot_string_matrix(matrix):
    unique_strings = np.unique(matrix)
    num_unique_strings = len(unique_strings)

    colors = plt.cm.get_cmap('Spectral', num_unique_strings)(np.linspace(0, 1, num_unique_strings))
    ind_color = {unique_strings[k]: colors[k] for k in range(num_unique_strings)}

    # Create a new matrix with colors
    color_matrix = np.zeros((matrix.shape[0], matrix.shape[1], 4))

    for i, string in enumerate(unique_strings):
        color_matrix[matrix == string] = colors[i]

    # Create the image
    fig, ax = plt.subplots()
    ax.imshow(color_matrix)
    ax.axis('off')
    plt.show(block = False)
    return color_matrix, ind_color



def check_intersection_wall(realcam, world):
    Intersection = np.empty(realcam.pixels, dtype = 'object')
    RPHIZ = np.full((realcam.pixels[0], realcam.pixels[1], 3), np.NaN)
    realcam.parent = world
    from cherab.tools.observers.intersections import find_wall_intersection
    for i in range(realcam.pixels[0]):
        for j in range(realcam.pixels[1]):
            if not np.isnan(realcam.pixel_origins[i, j][0]):
                try:
                    point, obj = find_wall_intersection(world, realcam.pixel_origins[i,j], realcam.pixel_directions[i, j])
                    Intersection[i, j] = obj.name
                    POINT = np.array([point.x, point.y, point.z])
                    pointRPHIZ = utility_functions.xyztorphiz(POINT)
                    RPHIZ[i, j, :] = pointRPHIZ
                except:
                    # raise Exception('pixel %02d, %02d does not intersect with wall' % (i, j))
                    print('pixel %02d, %02d does not intersect with wall' % (i, j))
    Intersection[Intersection==None] = 'None'

    return RPHIZ, Intersection



def optimize_boundary_grid(realcam, world, R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud):
    RPHIZ, Intersection = check_intersection_wall(realcam, world)
    R_min_noeud = min(R_min_noeud, np.nanmin(RPHIZ[:, :, 0]))
    R_max_noeud = max(R_max_noeud, np.nanmax(RPHIZ[:, :, 0]))
    Z_min_noeud = min(Z_min_noeud, np.nanmin(RPHIZ[:, :, 2]))
    Z_max_noeud = max(Z_max_noeud, np.nanmax(RPHIZ[:, :, 2]))
    phi_min = np.nanmin(RPHIZ[:, :, 1])
    phi_max = np.nanmax(RPHIZ[:, :, 1])
    
    return R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, phi_max, phi_min



def get_name_parameters_inversion(inversion_parameter):
    name_inversion_parameters = ''
    for key in inversion_parameter.keys():
        name_inversion_parameters = name_inversion_parameters + '_' + key
        if type(inversion_parameter[key]) != str:
            name_inversion_parameters = name_inversion_parameters + '_' + str(inversion_parameter[key])
        else:
            name_inversion_parameters = name_inversion_parameters + '_' + inversion_parameter[key]
    if not name_inversion_parameters:
        name_inversion_parameters = 'None'
    return name_inversion_parameters


def compare_reflection_results_parameters(parameters = None):
    if not parameters:
        name_parameters = utility_functions.get_file()

    comparison = 1
    return comparison 


def compare_synth_image(image, image_refl, image_absorb):
    std = np.load('std_noise.npy')
    diff_refl = image-image_refl
    diff_absorb = image-image_absorb
    pixel_reflections = np.abs(diff_refl)>1.2*std
    pixel_absorb = np.abs(diff_absorb)>1.2*std
    diff_refl[np.invert(pixel_reflections)] = 0
    diff_absorb[np.invert(pixel_absorb)] = 0
    utility_functions.plot_comparaison_image(np.abs(diff_refl), np.abs(diff_absorb), vmax = 50)
    return diff_refl, diff_absorb
    

def recreate_retrofit(image, inversion_full, data_refl, data_absorb):

    inversion_refl = inversion_full[data_refl['mask_noeud']]
    inversion_absorb = inversion_full[data_absorb['mask_noeud']]
    
    # image_retrofit_absorb = transfert_matrix_absorption.dot(inversion)
    # image_retrofit_full_absorb = reconstruct_2D_image(image_retrofit_absorb, mask_inversion)
    image_retrofit_refl = data_refl['transfert_matrix'].dot(inversion_refl)
    image_retrofit_full_refl = reconstruct_2D_image(image_retrofit_refl, data_refl['mask_pixel'])
    image_retrofit_absorb = data_absorb['transfert_matrix'].dot(inversion_absorb)
    image_retrofit_full_absorb = reconstruct_2D_image(image_retrofit_absorb, data_absorb['mask_pixel'])
    norm = 'linear'
    cmap = 'PiYG'
    import matplotlib.colors as mcolors
    # print(np.min(image), np.min(image_retrofit_full_absorb), np.min(image_retrofit_full_refl))
    diff = image-image_retrofit_full_refl
    if norm == 'log':
        image = np.log2(image+1e-10)   
        image_retrofit_full_absorb = np.log2(image_retrofit_full_absorb+1e-10)
        image_retrofit_full_refl = np.log2(image_retrofit_full_refl+1e-10)
        diff = np.log2(np.abs(diff)+1e-10)

    # if norm == 'log':
    #     image = np.log2(image-min(-1, np.min(image)))   
    #     image_retrofit_full_absorb = np.log2(image_retrofit_full_absorb-min(-1, np.min(image_retrofit_full_absorb)))
    #     image_retrofit_full_refl = np.log2(image_retrofit_full_refl+1)
    vmin = image.min()
    # vmin = 12
    # vmax = image.max()
    vmax = 200
    
    
    #synthetic image
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(image, cmap = cmap, vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.title('real image')

    #retro fit
    plt.subplot(2, 2,2)
    plt.imshow(image_retrofit_full_absorb, cmap = cmap, vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.title('Retro fit absorption')

    #retro fit
    plt.subplot(2, 2,3)
    plt.imshow(image_retrofit_full_refl, cmap = cmap, vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.title('Retro fit reflexion')
    #retro fit

    plt.subplot(2, 2,4)
    plt.imshow(diff, cmap = cmap)
    plt.colorbar()
    plt.title('difference retro fit refl and real image')
    plt.show(block = False)
    return inversion_refl, inversion_absorb, image_retrofit_full_absorb, image_retrofit_full_refl, diff





def compare_already_loaded_transfert_matrix(nshot, data_refl, data_absorb, path_resultat_inversion = None, path_vid = None):
    
    if not path_resultat_inversion:
        path_resultat_inversion = utility_functions.get_file('get npz file of inversion results', path_root='/Home/LF276573/Documents/Python/CHERAB/resultat_inversion/west/',full_path = 1)
    path_resultat_inversion, ext = os.path.splitext(path_resultat_inversion)
    if not path_vid:
        path_vid = utility_functions.get_file('get video file', path_root = '/Home/LF276573/Documents/Python/CHERAB/videos/west/', full_path = 1)

    

    inversion_results, mask_inversion, mask_noeud_inversion, frame_input = load_results_inversion(os.path.dirname(path_resultat_inversion), os.path.basename(path_resultat_inversion))
    frame_input = [frame_input[0], frame_input[1]]
    path_vid, ext = os.path.splitext(path_vid)
    if ext == '.png':
        vid, len_vid,image_dim_y,image_dim_x, fps, frame_input = get_img(path_vid, nshot)
        image = vid[0, :,:]
        image = np.flip(image, 1)
    else: #load videos
        vid, len_vid,image_dim_y,image_dim_x, fps, frame_input, name_time, t0 = get_vid(None, frame_input, path_vid = path_vid, nshot = nshot)
        image = vid[0, :,:]
        image = np.flip(image, 0)
        image = image.T
    inversion = inversion_results[0, :]

    inversion_full = reconstruct_2D_image(inversion, mask_noeud_inversion)
    inversion_refl, inversion_absorb, image_retrofit_full_absorb, image_retrofit_full_refl, diff = recreate_retrofit(image, inversion_full, data_refl, data_absorb)
    return image, image_retrofit_full_absorb, image_retrofit_full_refl, diff, inversion_full, nshot, path_resultat_inversion, path_vid, data_refl, data_absorb

def read_CAD(path_CAD, world, name_material = 'AbsorbingSurface', wall_material = AbsorbingSurface(), variant = 'default'):

    import json
    stl_files = [f for f in os.listdir(path_CAD) if (f.endswith('.stl') or f.endswith('.obj'))]
    path_stl = [path_CAD+stl for stl in stl_files]
    wall_materials = read_material(path_stl, name_material, wall_material)

    if not path_stl:
        stl_files = [f for f in os.listdir(path_CAD + 'large/') if (f.endswith('.stl') or f.endswith('.obj'))]
        path_stl = [path_CAD+ 'large/' + variant+ stl for stl in stl_files]
        wall_materials = read_material(path_stl, name_material, wall_material)
        full_wall = read_CALCAM_CAD(path_CAD, world, wall_materials)
        return full_wall, name_material
    else:
        try:
            with open(path_CAD + 'model.json') as f:
                dat = json.load(f) 
                f.close()
        except:
            full_wall =  [import_stl(f, parent = world, material = wall_materials[i], name = stl_files[i]) for i, f in enumerate(path_stl)]
            return full_wall,name_material
        features = dat['features'][variant].keys()
        data = dat['features'][variant]
        for stl in features:
            if data[stl]['default_enable'] == True:
                if 'mesh_up_direction' in data[stl].keys():
                    if data[stl]['mesh_up_direction'] == '+Y':
                        full_wall =  import_stl(path_CAD + stl + '.stl', scaling = data[stl]['mesh_scale'], parent = world, material = wall_material, name = stl, transform = rotate(0, -90,0)) 
                    else:
                        full_wall =  import_stl(path_CAD + stl + '.stl', scaling = data[stl]['mesh_scale'], parent = world, material = wall_material, name = stl) 
                else:
                    full_wall =  import_stl(path_CAD + stl + '.stl', scaling = data[stl]['mesh_scale'], parent = world, material = wall_material, name = stl) 

    return full_wall, name_material


def read_CALCAM_CAD(path_CAD, world, wall_materials):
    import json
    with open(path_CAD + 'model.json') as f:
        dat = json.load(f) 
        f.close()

    features = dat['features']['Default'].keys()
    data = dat['features']['Default']
    for i, obj in enumerate(features):
        if data[obj]['default_enable'] == True:
            if data[obj]['mesh_up_direction'] == '+Y':
                full_wall =  import_obj(path_CAD + 'large/' + obj + '.obj', scaling = data[obj]['mesh_scale'], parent = world, material = wall_materials[i], name = obj, transform = rotate(0, -90,0)) 
            else:
                full_wall =  import_obj(path_CAD + 'large/' + obj+ '.obj', scaling = data[obj]['mesh_scale'], parent = world, material = wall_materials[i], name = obj) 
            print(obj, wall_materials[i])
    return full_wall



def convert_npz_to_mat(file_path = None):
    if not file_path:
        file_path = utility_functions.get_file(path_root  = '/Home/LF276573/Zone_Travail/Python/CHERAB/transfert_matrix/west/', full_path=0)
    data = load_npz(file_path)
    file_name, ext = os.path.splitext(file_path)
    savemat(file_name + '.mat', {'transfert_matrix' : data})


def remove_center_from_inversion(vertex_mask, cell_vertices_r, cell_vertices_z):
    magflux = imas_west.get(61636, 'equilibrium', 0, 1)
    idx = 100
    r = magflux.interp2D.r
    z = magflux.interp2D.z
    psi = magflux.interp2D.psi[idx, ...]
    #get psi norm
    # self.r0 = 
    # self.z0 = 
    # self.psi_norm = magflux.interp2D.psi[idx, :, :]/magflux.boundary.psi[idx]
    # self.rsep = magflux.boundary.r[idx, :] doesn't exist
    # self.zsep = magflux.boundary.z[idx, :] doesn't exist
    psisep = magflux.boundary.psi[idx]
    psi0 = np.nanmax(psi)

    psi_int = 0.9*(psisep-psi0)+psi0
    # RZpoints = np.column_stack((X.ravel(), Y.ravel()))  # shape (N, 2)
    # flat_psi = psi.ravel()  # shape (N,)
    # Create contour for just this isovalue
    contour = plt.contour(r, z, psi, levels=[psi_int], colors='red')

    # Extract the contour line(s)
    paths = contour.collections[0].get_paths()

    # Optional: grab the first contour line only
    contour_points = paths[0].vertices  # shape (N, 2), columns are [x, y]

    for i in range(len(cell_vertices_r)):
            for j in range(len(cell_vertices_z)):
                r_noeud = cell_vertices_r[i]
                z_noeud = cell_vertices_z[j]
                if   is_point_in_contour(r_noeud, z_noeud, contour_points):          
                    vertex_mask[i, j]= 0

    
    return vertex_mask



# Assume contour_points is Nx2 array from a contour line (e.g., a circle-ish shape)
# contour_points = path.vertices from earlier

def is_point_in_contour(x, y, contour_points):
    from matplotlib.path import Path

    contour_path = Path(contour_points)
    return contour_path.contains_point((x, y))




def create_synth_image(path_parameters = None, path_transfert_matrix = None):
    [transfert_matrix, 
            pixels,
            noeuds, 
            nb_noeuds_r, 
            nb_noeuds_z, 
            R_max_noeud, 
            R_min_noeud, 
            Z_max_noeud, 
            Z_min_noeud, 
            R_noeud,
            Z_noeud, 
            mask_pixel, 
            mask_noeud,
            path_parameters] = load_transfert_matrix_and_parameters(path_parameters, path_transfert_matrix)
    synth_node = np.ones((transfert_matrix.shape[1], 1))
    synth_im = transfert_matrix.dot(synth_node)
    synth_im_full = reconstruct_2D_image(np.squeeze(synth_im), mask_pixel)
    utility_functions.plot_image(synth_im_full.T, title=path_parameters)
    plt.savefig(main_folder_image + 'synth_image_full_emissivity.png')
    return (transfert_matrix, 
            pixels,
            noeuds, 
            nb_noeuds_r, 
            nb_noeuds_z, 
            R_max_noeud, 
            R_min_noeud, 
            Z_max_noeud, 
            Z_min_noeud, 
            R_noeud,
            Z_noeud, 
            mask_pixel, 
            mask_noeud,
            path_parameters,
            synth_im_full)

def load_transfert_matrix_and_parameters(path_parameters = None, path_transfert_matrix = None):
    if not path_parameters:
        path_parameters =utility_functions.get_file(path_root  = '/Home/LF276573/Zone_Travail/Python/CHERAB/transfert_matrix/west/', full_path=0)
        name_parameters, ext = os.path.splitext(path_parameters)
        path_transfert_matrix = name_parameters + '_transfert_matrix.npz'
    loaded_raytracing = np.load(path_parameters, allow_pickle = True)
    transfert_matrix = load_npz(path_transfert_matrix)
    try:
        pixels = loaded_raytracing['pixels']
    except:
        test = loaded_raytracing['arr_0']
        dict_save_parameters = dict(test.tolist())
        np.savez_compressed(path_parameters, **dict_save_parameters)
        loaded_raytracing = np.load(path_parameters, allow_pickle = True)
        try:
            pixels = loaded_raytracing['pixels']
        except:
            raise(Exception('fail to resave parameters'))
    noeuds = loaded_raytracing['noeuds']
    pixels = np.squeeze(pixels)
    noeuds = np.squeeze(noeuds)
    nb_noeuds_r = loaded_raytracing['nb_noeuds_r']
    nb_noeuds_z = loaded_raytracing['nb_noeuds_z']
    R_max_noeud = loaded_raytracing['R_max_noeud']
    R_min_noeud = loaded_raytracing['R_min_noeud']
    Z_max_noeud = loaded_raytracing['Z_max_noeud']
    Z_min_noeud = loaded_raytracing['Z_min_noeud']
    R_noeud = loaded_raytracing['R_noeud']
    Z_noeud = loaded_raytracing['Z_noeud']
    mask_noeud = loaded_raytracing['mask_noeud']
    mask_pixel = loaded_raytracing['mask_pixel']
    if mask_pixel.ndim != 2:
        pdb.set_trace()
    dict_save_parameters = dict(pixels = pixels, 
                        noeuds = noeuds, 
                        nb_noeuds_r = nb_noeuds_r, 
                        nb_noeuds_z = nb_noeuds_z, 
                        R_max_noeud = R_max_noeud, 
                        R_min_noeud = R_min_noeud, 
                        Z_max_noeud = Z_max_noeud, 
                        Z_min_noeud = Z_min_noeud, 
                        R_noeud = R_noeud,
                        Z_noeud = Z_noeud, 
                        mask_pixel = mask_pixel, 
                        mask_noeud= mask_noeud)
    # path_parameters_save ,ext = os.path.splitext(path_parameters)
    # savemat(path_parameters_save + '.mat', dict_save_parameters)
    return (transfert_matrix, 
            pixels,
            noeuds, 
            nb_noeuds_r, 
            nb_noeuds_z, 
            R_max_noeud, 
            R_min_noeud, 
            Z_max_noeud, 
            Z_min_noeud, 
            R_noeud,
            Z_noeud, 
            mask_pixel, 
            mask_noeud,
            path_parameters)





# def verify_integrity(realcam, mask_pixel):
    for i in range(mask_pixel.shape[0]):
        for j in range(mask_pixel.shape[1]):
            if mask_pixel[i, j] == 1:
                if np.isnan(realcam.pixel_directions[i, j].x) or np.isnan(realcam.pixel_origins[i, j].x):
                    return False
            elif mask_pixel[i, j] == 0:
                if not np.isnan(realcam.pixel_directions[i, j].x) or not np.isnan(realcam.pixel_origins[i, j].x):
                    return False
            else:
                raise('mask data cannot be used to mask pixels of camera. Please check datatype')

    return True


def plot_transfert_matrix_and_synthetic(vmax = None):
    path_parameters =utility_functions.get_file(path_root  = '/Home/LF276573/Zone_Travail/Python/CHERAB/transfert_matrix/west/', full_path=0)
    # path_parameters =utility_functions.get_file(path_root  = '/Home/LF276573/Documents/Python/CHERAB/failure/', full_path=1)
    name_parameters, ext = os.path.splitext(path_parameters)

    path_transfert_matrix = name_parameters + '_transfert_matrix.npz'
    try:
        loaded_raytracing = np.load(path_parameters, allow_pickle = True)
        transfert_matrix = load_npz(path_transfert_matrix)
    except:
        loaded_raytracing = np.load(path_parameters, allow_pickle = True)
        folder_transfert_matrix = os.path.dirname(name_parameters)
        path_transfert_matrix = folder_transfert_matrix + '/transfert_matrix.npz'
        transfert_matrix = load_npz(path_transfert_matrix)
    try:
        pixels = loaded_raytracing['pixels']
    except:
        test = loaded_raytracing['arr_0']
        dict_save_parameters = dict(test.tolist())
        np.savez_compressed(path_parameters, **dict_save_parameters)
        loaded_raytracing = np.load(path_parameters, allow_pickle = True)
        try:
            pixels = loaded_raytracing['pixels']
        except:
            raise(Exception('fail to resave parameters'))
    noeuds = loaded_raytracing['noeuds']
    pixels = np.squeeze(pixels)
    noeuds = np.squeeze(noeuds)
    nb_noeuds_r = loaded_raytracing['nb_noeuds_r']
    nb_noeuds_z = loaded_raytracing['nb_noeuds_z']
    R_max_noeud = loaded_raytracing['R_max_noeud']
    R_min_noeud = loaded_raytracing['R_min_noeud']
    Z_max_noeud = loaded_raytracing['Z_max_noeud']
    Z_min_noeud = loaded_raytracing['Z_min_noeud']
    R_noeud = loaded_raytracing['R_noeud']
    Z_noeud = loaded_raytracing['Z_noeud']
    mask_noeud = loaded_raytracing['mask_noeud']
    mask_pixel = loaded_raytracing['mask_pixel']
    if mask_pixel.ndim != 2:
            pdb.set_trace()
    see_nodes = np.sum(transfert_matrix, 0)
    see_pix = np.sum(transfert_matrix, 1)
    see_nodes_full = reconstruct_2D_image(see_nodes, mask_noeud)
    see_pix_full = reconstruct_2D_image(see_pix, mask_pixel)

    utility_functions.plot_image(see_nodes_full, title=name_parameters, vmax=vmax)

    utility_functions.plot_image(see_pix_full, title=name_parameters, vmax=vmax)
    return transfert_matrix, see_nodes_full, see_pix_full, mask_noeud, mask_pixel



def find_material(components_dict, query_name):
    
    for full_name, material in components_dict.items():
        # Remove numerical prefix (e.g., '01_01_') and compare
        name_file = query_name.split('/')[-1]
        name_file = name_file.split('.')[0]
        if name_file == full_name:
            return material
    raise ValueError(f"No component matches the name '{query_name}'.")


def read_material(path_stl, name_material, type_materials):
    if name_material == 'absorbing_surface':
        wall_materials = [AbsorbingSurface()]*len(path_stl)
        return wall_materials
    else:
        wall_materials = []

        # with open('../ressources/components.yaml', 'rb') as f:
        #     components = yaml.safe_load(f)
        #     f.close()
        for i, f in enumerate(path_stl):
            material = find_material(components, f)
            if 'W' in material:
                wall_materials.append(type_materials)
            else:
                wall_materials.append(Lambert()) 

        return wall_materials




def recognise_material(name_material):
    if name_material== 'absorbing_surface':
        wall_material = AbsorbingSurface()
        return name_material, wall_material
    else:
        import re

       
        try:
            match = re.match(r"([a-zA-Z_]+)(\d+)$", name_material)
            if match:
                letters = match.group(1)
                numbers = match.group(2)
                numbers = smart_string_to_number(numbers)
            if letters == 'tungsten':
                wall_material = RoughTungsten(numbers)
                return name_material, wall_material
            else:
                print(letters)
                name_material= 'Lambert'
                wall_material = Lambert()
                return name_material, wall_material
                # raise(NameError('unrecognized material'))
        except:
            name_material= 'absorbing_surface'
            wall_material = AbsorbingSurface()
            return name_material, wall_material
            # raise(NameError('unrecognized material'))
            
   


def smart_string_to_number(s):
    if not s.isdigit():
        raise ValueError("Input must be a string of digits only.")
    
    if s.startswith('0'):
        return float('0.' + s.lstrip('0') or '0')  # Handles all-zero edge cases
    else:
        return float(s)
    





def read_CAD_from_calcam_module(path_CAD, world, name_material, wall_material, variant = 'Default'):
    import calcam
    CAD = calcam.CADModel(path_CAD, model_variant = variant)
    CAD.enable_only(['Limiters', 'Vessel_midplane']) #ugly fix
    features = CAD.get_enabled_features()
    print(features)

    path_stl = [CAD.features[feature].filename for feature in features] 

    wall_materials = read_material(path_stl, name_material, wall_material)
    full_wall =  [import_stl(f, parent = world, scaling = 0.001 , material = wall_materials[i], name = features[i]) for i, f in enumerate(path_stl)]
    CAD.unload()
    return full_wall, name_material


def create_synth_cam(realcam, full_wall, R_wall, Z_wall, mask):

        realcam_quick_view = VectorCamera(realcam.pixel_origins, realcam.pixel_directions)
        world2 = World()
        cylinder_inner = Cylinder(radius=min(R_wall), height=max(Z_wall) - min(Z_wall))
        cylinder_outer = Cylinder(radius=max(R_wall), height=max(Z_wall) - min(Z_wall))
        wall2 = Subtract(cylinder_outer, cylinder_inner, material=RoughTungsten(0.5),  transform=translate(0, 0, min(Z_wall)))
        rtc = RayTransferCylinder(max(R_wall), (max(Z_wall) - min(Z_wall)), 50, 100, radius_inner=min(R_wall))
        rtc.parent = world2
        rtc.transform = translate(0, 0, min(Z_wall))
        name_material = 'tungsten_05'
        name_material, wall_material = recognise_material(name_material)

        full_wall2 = read_CAD_from_calcam_module('/compass/home/fevre/WESTCOMPASS_tomography/models_and_calibrations/models/compass/compass 20879.ccm', world2, name_material, wall_material, variant = 'half model')
        rad_circle = 1000000.
        rsqr = np.linspace(-49.5, 100.5, 50) ** 2
        zsqr = np.linspace(0, 49.5, 100) ** 2
        rad = np.sqrt(rsqr[:, None] + zsqr[None, :])
        maskrad = rad < rad_circle  # a boolean array 50*50 (True inside the circle, False - outside)
        rtc.mask = maskrad[:, None, :]  # making 3D mask from 2D (RZ-plane) mask
        pipeline2 = RayTransferPipeline2D()
        realcam_quick_view.spectral_bins = rtc.bins
        realcam_quick_view.min_wavelength = 600
        realcam_quick_view.max_wavelength = realcam_quick_view.min_wavelength +1
        realcam_quick_view.frame_sampler = FullFrameSampler2D(mask)
        realcam_quick_view.parent = world2
        realcam_quick_view.pipelines = [pipeline2]
        realcam_quick_view.observe()
        # profile = rad[maskrad]
        profile = rad[maskrad]
        image = np.dot(pipeline2.matrix, profile)
        plt.figure()
        plt.imshow(image.T)
        plt.show(block = False)
        plt.savefig(main_folder_image + 'synth_image_verbose.png')


def create_synth_cam_emitter(realcam, full_wall, R_wall, Z_wall, mask, path_CAD, variant = 'Default'):
    from matplotlib.pyplot import ion, ioff
    from raysect.optical.library.spectra.colours import yellow
    from raysect.optical.observer import RGBPipeline2D, RGBAdaptiveSampler2D
    from raysect.optical.material import UniformSurfaceEmitter
    from raysect.core.workflow import MulticoreEngine
    colours = [yellow]  
    realcam_quick_view = VectorCamera(realcam.pixel_origins, realcam.pixel_directions)
    world2 = World()
    cylinder_inner = Cylinder(radius=15*min(R_wall)/32, height=1*(max(Z_wall) - min(Z_wall))/64)
    cylinder_outer = Cylinder(radius=16*max(R_wall)/32, height=1*(max(Z_wall) - min(Z_wall))/32)
    plasma = Subtract(cylinder_outer, cylinder_inner, material= UniformSurfaceEmitter(colours[0]),  transform=translate(0, 0, 0))      
    plasma.parent = world2
    # wall2 = Subtract(cylinder_outer, cylinder_inner, material=RoughTungsten(0.5),  transform=translate(0, 0, min(Z_wall)))
    # rtc = RayTransferCylinder(max(R_wall), (max(Z_wall) - min(Z_wall)), 50, 100, radius_inner=min(R_wall))
    # rtc.paSrent = world2
    # rtc.transform = translate(0, 0, min(Z_wall))
    name_material = 'tungsten05'
    name_material, wall_material = recognise_material(name_material)
    full_wall2 = read_CAD_from_calcam_module(path_CAD, world2, name_material, wall_material, variant = variant)


    rgb = RGBPipeline2D(name="sRGB")
    sampler = RGBAdaptiveSampler2D(rgb, mask = mask, ratio=10, fraction=0.2, min_samples=500, cutoff=0.05)

    realcam_quick_view.min_wavelength = 571
    realcam_quick_view.max_wavelength = realcam_quick_view.min_wavelength +2
    realcam_quick_view.frame_sampler = sampler
    realcam_quick_view.spectral_bins = 25
    realcam_quick_view.pixel_samples = 100
    realcam_quick_view.parent = world2
    realcam_quick_view.pipelines = [rgb]
    realcam_quick_view.render_engine = MulticoreEngine(4)

    # start ray tracing
    ion()
    p = 1
    while not realcam_quick_view.render_complete:
        print("Rendering pass {}...".format(p))

        realcam_quick_view.observe()
        print(realcam_quick_view.render_complete)
        p += 1
        if (p % 10) ==0:
            rgb.save(main_folder_image + 'temp_render_synth_camera_image.png')
            utility_functions.save_transposed_image(main_folder_image + 'temp_render_synth_camera_image.png', main_folder_image + 'temp_render_synth_camera_image_rotated.png')
            # plt.close('all')

    ioff()
    # rgb.display()

    utility_functions.save_transposed_image(main_folder_image + 'synth_camera_image.png')
    # plt.figure()
    # plt.imshow(image.T, origin = 'lower')
    # plt.show(block = False)
    # plt.savefig('synth_image_verbose.png')


        
def optimize_grid_from_los(realcam, world, pos_camera_RPHIZ):
    RPHIZ, Intersection = check_intersection_wall(realcam, world)
    R_min_noeud = pos_camera_RPHIZ[0] 
    R_max_noeud = pos_camera_RPHIZ[0] 
    Z_min_noeud = pos_camera_RPHIZ[2] 
    Z_max_noeud = pos_camera_RPHIZ[2] 
    for i in range(RPHIZ.shape[0]): 
        for j in range(RPHIZ.shape[1]):

            R_min, R_max, Z_min, Z_max = utility_functions.find_RZextrema_between_2_points(pos_camera_RPHIZ, RPHIZ[i, j, :])
                    
            R_min_noeud = min(R_min_noeud, R_min)
            R_max_noeud = max(R_max_noeud, R_max)
            Z_min_noeud = min(Z_min_noeud, Z_min)
            Z_max_noeud = max(Z_max_noeud, Z_max)
    phi_min = min(pos_camera_RPHIZ[1], np.nanmin(RPHIZ[:, :, 1]))
    phi_max = max(pos_camera_RPHIZ[1], np.nanmax(RPHIZ[:, :, 1]))
    
    return R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, phi_max, phi_min, RPHIZ




def load_mask(path_calibration, path_mask):
    if path_mask is None:
        try:
            import calcam
            calcam_camera = calcam.Calibration(path_calibration)
            mask_pixel = calcam_camera.subview_mask
            mask_pixel = np.invert(mask_pixel)
            name_mask = 'calibration_calcam'        
        except:

            try:
                calcam_camera = np.load(path_calibration, allow_pickle=True)

                mask_pixel = calcam_camera['mask']
                mask_pixel = np.invert(mask_pixel)
                name_mask = 'calibration_calcam'
            except:
                name_mask = get_name(path_mask)
                try:
                    fmask = loadmat(path_mask)
                    mask_pixel = fmask["mask"]
                except:
                    mask_pixel = np.load(path_mask)

    else:
        try:
            mask_pixel = calcam_camera['mask']
            mask_pixel = np.invert(mask_pixel)
            name_mask = 'calibration_calcam'
        except:
            name_mask = get_name(path_mask)
            try:
                fmask = loadmat(path_mask)
                mask_pixel = fmask["mask"]
            except:
                mask_pixel = np.load(path_mask)
    mask_pixel[np.isnan(mask_pixel)] = 0 #handle ill defined mask
    mask_pixel = np.abs(mask_pixel)# handle calcam convention, set all the non zero indices to positive for further process 
    return mask_pixel, name_mask


def load_camera(path_calibration):
    try: 
        import calcam
        calcam_camera = calcam.Calibration(load_filename = path_calibration)
        realcam = calcam_camera.get_raysect_camera(coords = 'Display')
        pixel_origins = realcam.pixel_origins
        mask = calcam_camera.subview_mask
        pixel_directions = realcam.pixel_directions
    except:
        calcam_camera = np.load(path_calibration, allow_pickle=True)
        pixel_origins = calcam_camera['pixel_origins']
        pixel_directions = calcam_camera['pixel_directions']
        realcam = VectorCamera(pixel_origins.T, pixel_directions.T)
    return realcam


def compare_voxel_map_and_pleque(plasma, FL_matrix, Z_min, phi_mem):
    ind_voxel = np.unique(plasma.voxel_map)
    ind_voxel = ind_voxel[1:]
    import random
    ind_choice = random.choice(ind_voxel)
    inverted_voxel_map = plasma.invert_voxel_map()
    inverted_voxel_map = inverted_voxel_map[ind_choice]
    material = plasma.material
    nr, nphi, nz = material.grid_shape
    dz = material.dz
    dr = material.dr
    dphi = material.dphi
    period = material.period
    rmin = material.rmin
    R_plasma = rmin + dr * np.arange(nr)
    Z_plasma = Z_min+ dz * np.arange(nz)
    PHI_plasma = dphi*np.arange(nphi)
    R1 = R_plasma[inverted_voxel_map[0]]
    PHI1 = PHI_plasma[inverted_voxel_map[1]]*np.pi/180
    Z1 = Z_plasma[inverted_voxel_map[2]]
    R2 = FL_matrix[ind_choice, 0, :]
    Z2 = FL_matrix[ind_choice, 1, :]
    PHI2 = phi_mem
    plot_compare_3D_lines(R1, PHI1, Z1, R2, PHI2, Z2, label1 = 'plasma line', label2 = 'Magnetic field line')
    return R1, PHI1, Z1, R2, PHI2, Z2


def plot_compare_3D_lines(R1, PHI1, Z1, R2, PHI2, Z2, label1 = 'Line 1 in Cylindrical Coordinates', label2 = 'Line 2 in Cylindrical Coordinates'):

    X1 = R1 * np.cos(PHI1)
    Y1 = R1 * np.sin(PHI1)
    X2 = R2 * np.cos(PHI2)
    Y2 = R2 * np.sin(PHI2)
    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X1, Y1, Z1, label= label1)
    ax.plot(X2, Y2, Z2, label= label2)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.legend()
    plt.tight_layout()
    plt.show(block = False)





def get_mask_from_wall(R_min_noeud, R_max_noeud, Z_min_noeud, Z_max_noeud, nb_noeuds_r, nb_noeuds_z, wall_limit, dict_transfert_matrix = None):

    cell_r, cell_dr = np.linspace(R_min_noeud, R_max_noeud, nb_noeuds_r, retstep=True, endpoint = False)
    cell_z, cell_dz = np.linspace(Z_min_noeud, Z_max_noeud, nb_noeuds_z, retstep=True, endpoint = False)
    cell_r_grid, cell_z_grid = np.broadcast_arrays(cell_r[:, None], cell_z[None, :])
    cell_centres = np.stack((cell_r_grid, cell_z_grid), axis=-1)  # (nx, ny, 2) array
    # Define the positions of the vertices of the voxels
    cell_vertices_r = np.linspace(cell_r[0] - 0.5 * cell_dr, cell_r[-1] + 0.5 * cell_dr, nb_noeuds_r + 1)
    cell_vertices_z = np.linspace(cell_z[0] - 0.5 * cell_dz, cell_z[-1] + 0.5 * cell_dz, nb_noeuds_z + 1)

    # Build a mask, only including cells within the wall
   
    vertex_mask = np.zeros((len(cell_vertices_r), len(cell_vertices_z)))
    

    for i in range(nb_noeuds_r):
        for j in range(nb_noeuds_z):
            r_noeud = cell_vertices_r[i]
            
            z_noeud = cell_vertices_z[j]
            pointrz = Point3D(r_noeud, 0, z_noeud)
            if   wall_limit.contains(pointrz):          
                vertex_mask[i, j]= 1

    #remove points for psi_norm<0.9
    if dict_transfert_matrix.get('crop_center'):
        vertex_mask = remove_center_from_inversion(vertex_mask, cell_vertices_r, cell_vertices_z)

    # Cell is included if at least one vertex is within the wall
    grid_mask = (vertex_mask[1:, :-1] + vertex_mask[:-1, :-1]
                + vertex_mask[1:, 1:] + vertex_mask[:-1, 1:])
    return cell_r, cell_z, grid_mask, cell_dr, cell_dz



def closest_points(grid_points, query_points):
    """
    Find the closest points in a 2D irregular grid for many queries.

    Parameters
    ----------
    grid_points : ndarray of shape (N, 2)
        Array of (x, y) grid points.
    query_points : ndarray of shape (M, 2)
        Array of query points (x, y).

    Returns
    -------
    closest_idx : ndarray of shape (M,)
        Indices of the closest grid points.
    closest_points : ndarray of shape (M, 2)
        Coordinates of the closest grid points.
    distances : ndarray of shape (M,)
        Distances to the closest points.
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(grid_points)
    dists, idxs = tree.query(query_points)
    return idxs, grid_points[idxs], dists



__all__ = ["full_inversion_toroidal"]




