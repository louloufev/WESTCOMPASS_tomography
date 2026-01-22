import numpy as np
from scipy.io import loadmat, savemat
from scipy.sparse import csr_matrix, save_npz, csc_matrix, load_npz
from matplotlib import pyplot as plt
from raysect.primitive import import_stl, import_obj, Box, import_ply
from raysect.optical import World, Node, translate, rotate, ConstantSF, Point3D, rotate_z, Vector3D
from raysect.optical.observer import PinholeCamera, FullFrameSampler2D, RGBPipeline2D, VectorCamera,  RGBAdaptiveSampler2D
from raysect.optical.material import UniformSurfaceEmitter, InhomogeneousVolumeEmitter, UniformVolumeEmitter, AbsorbingSurface, Lambert
from cherab.tools.primitives.axisymmetric_mesh import axisymmetric_mesh_from_polygon
from raysect.core.math.polygon import triangulate2d
from cherab.tools.raytransfer import RayTransferPipeline2D, RayTransferCylinder
from cherab.tools.raytransfer import RoughIron, RoughTungsten, RoughSilver
import os
from raysect.optical.library.spectra.colours import *
colours = [yellow, orange, red_orange, red, purple, blue, light_blue, cyan, green]
from raysect.primitive import Cylinder, Subtract
from tomotok.core.inversions import Bob, SparseBob, CholmodMfr, Mfr
from cherab.tools.inversions import invert_regularised_nnls
from tomotok.core.derivative import compute_aniso_dmats
from tomotok.core.geometry import RegularGrid
from scipy.interpolate import RegularGridInterpolator
import pdb
import sys
import time
import importlib
from . import utility_functions, result_inversion

import tkinter as tk
from tkinter import filedialog

from Tomography.core.inversion_module import prep_inversion, inverse_vid, inversion_and_thresolding, synth_inversion, reconstruct_2D_image, inverse_vid_from_class
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

main_folder_image = paths["main_folder_image"]


main_folder_processing = paths["main_folder_processing"]

def full_inversion_toroidal(ParamsMachine, ParamsGrid, ParamsVid):
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

            inversion_results_thresholded_full : 3D array (time, R, Z)
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
    #create folder where to save the outputs
    start_time_get_parameters = time.time()
    #import input parameters
    
    if not ParamsVid.path_vid:
        path_vid = 'None'
    #import camera calibrations
    world = World()
    real_pipeline = RayTransferPipeline2D()

    print('loading camera')
    realcam = load_camera(ParamsMachine.path_calibration)
    mask_pixel, name_mask = load_mask(ParamsMachine.path_calibration, ParamsMachine.path_mask)
    print('camera loaded')
    

#check surface
    name_material = os.path.splitext(os.path.basename(ParamsMachine.name_material))
    if ParamsMachine.path_CAD:
        type_wall = 'CAD'
        name_CAD = os.path.splitext(os.path.basename(ParamsMachine.path_CAD))[0]
        if not os.path.exists(ParamsMachine.path_CAD):
            # if os.path.dirname(path_CAD) != main_folder_CAD:
            path_CAD = main_folder_CAD + ParamsMachine.path_CAD

    else:
        type_wall = 'coords'
        name_CAD = 'coords'


    if not ParamsGrid.phi_grid:
        n_polar = 1
    else:
        n_polar = ParamsGrid.n_polar


#loading wall coordinates
    if ParamsMachine.path_wall:
        try:
            fwall = loadmat(ParamsMachine.path_wall)
            RZwall = fwall['RZwall']

        except:
            RZwall = np.load(ParamsMachine.path_wall, allow_pickle=True)
        #check that the last element is the neighbor of the first one and not the same one
        if(RZwall[0]==RZwall[-1]).all():
            RZwall = RZwall[:-1]
            print('erased last element of wall')
        #check that the wall coordinates are stocked in a counter clockwise position. If not, reverse it
        Trigo_orientation, signed_area = utility_functions.polygon_orientation(RZwall[:, 0], RZwall[:, 1])
        if Trigo_orientation:
            RZwall = RZwall[::-1]
            print('wall reversed')

        R_wall = RZwall[:, 0]
        Z_wall = RZwall[:, 1]
    else:
        RZwall = None

    try:

        Inversion_results = result_inversion.Inversion_results({'root_folder' : main_folder_processing}, ParamsMachine = ParamsMachine, ParamsGrid = ParamsGrid, ParamsVid=ParamsVid)
        Inversion_results = Inversion_results.load()

        return Inversion_results
    except:
        print('no previous results found')
        
    # if ext == '.png':
    #     vid, len_vid,image_dim_y,image_dim_x, fps, frame_input = get_img(ParamsVid)
    #     t0 = 0
    #     tstart = 0
    #     tinv = 0
    ####
    # else: #load videos
    print('loading videos')
    vid, len_vid,image_dim_y,image_dim_x, fps, frame_input, name_time, t_start, t0, t_inv = utility_functions.get_vid(ParamsVid)
    Inversion_results = result_inversion.Inversion_results({'root_folder' : main_folder_processing}, ParamsMachine = ParamsMachine, ParamsGrid = ParamsGrid, ParamsVid=ParamsVid)
    Inversion_results.t_inv = t_inv
    print('video loaded')
    #load mask, check size
    # utility_functions.save_array_as_img(vid, main_folder_image + 'image_vid_mid.png')
    # utility_functions.save_array_as_gif(vid, gif_path=main_folder_image + 'quickcheck_cam.gif', num_frames=100, cmap='gray')
    mask_pixel = mask_pixel.T
    mask_pixel = np.ascontiguousarray(mask_pixel)
    vid = np.swapaxes(vid, 1,2)
    # utility_functions.save_array_as_img(vid, main_folder_image + 'image_vid_mid_rotated.png')
    if ParamsMachine.machine == 'WEST':
        vid = np.swapaxes(vid, 1,2)
        vid = np.flip(vid, 1)
        # if image_dim_y == mask_pixel.shape[0] and ParamsMachine.param_fit==None:
        #     vid = np.swapaxes(vid, 1,2)
    # utility_functions.save_array_as_gif(vid, gif_path=main_folder_image + 'quickcheck_cam_after_rotation.gif', num_frames=100, cmap='viridis')
    pdb.set_trace()
    realcam, mask_pixel, vid = fit_size_all(realcam, mask_pixel, vid, ParamsMachine.param_fit)
    # utility_functions.save_array_as_gif(vid, gif_path=main_folder_image + 'quickcheck_cam_after_rezizing.gif', num_frames=100, cmap='viridis')


    if ParamsMachine.decimation is not None:
        realcam, mask_pixel, vid = reduce_camera_precision(realcam, mask_pixel, vid, decimation = ParamsMachine.decimation)
        # utility_functions.save_array_as_gif(vid, gif_path=main_folder_image + 'quickcheck_cam_after_decimation.gif', num_frames=100, cmap='viridis')
   
    realcam.frame_sampler=FullFrameSampler2D(mask_pixel)
    realcam.pipelines=[real_pipeline]
    realcam.parent=world
    realcam.pixel_samples = 100
    realcam.min_wavelength = 640
    realcam.max_wavelength = realcam.min_wavelength +1
    realcam.render_engine.processes = 32
    
    try:
        Transfert_Matrix = result_inversion.Transfert_Matrix({'root_folder' : main_folder_processing}, ParamsMachine = ParamsMachine, ParamsGrid = ParamsGrid)
        Transfert_Matrix = Transfert_Matrix.load()
        print('found already calculated transfert matrix')
        #skips the loading of the walls, go straight to inversion
    except:

        #load wall models
        # check how the wall is described; either CAD or coord
        
        if ParamsMachine.path_CAD is not None:
            full_wall = load_walls(ParamsMachine, world)
            
        else: 
            full_wall = axisymmetric_mesh_from_polygon(RZwall)
            full_wall.material = wall_material
            full_wall.parent = world
        #calculate transfert matrix
        Transfert_Matrix = result_inversion.Transfert_Matrix({'root_folder' : main_folder_processing}, ParamsMachine = ParamsMachine, ParamsGrid = ParamsGrid)
        Transfert_Matrix.mask_pixel = mask_pixel
        Transfert_Matrix = get_transfert_matrix(Transfert_Matrix, realcam,  world, ParamsMachine, ParamsGrid, ParamsVid, Inversion_results, RZwall)
        print('transfert matrix calculated')
    end_time_get_parameters = time.time()-start_time_get_parameters

    start_time_get_equilibrium = time.time()


    # if derivative_matrix:
    #     derivative_matrix = [[matrix[noeuds, :][:, noeuds] for matrix in sublist] for sublist in derivative_matrix]
    end_time_get_equilibrium = time.time()-start_time_get_equilibrium
    



    start_time_get_inversion = time.time()

    if ParamsVid.inversion_method == 'Cholmod' or ParamsVid.inversion_method == 'Mfr_Cherab':
        
        inversion_results, inversion_results_normed, inversion_results_thresholded, inversion_results_thresholded_normed, images_retrofit, mask_noeud, transfert_matrix = call_module2_function("inverse_vid", 
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
        Inversion_results.vid = vid
        Inversion_results= inverse_vid_from_class(Transfert_Matrix,Inversion_results, ParamsMachine, ParamsGrid, ParamsVid)
        #save results
    start_time_get_save = time.time()
    
       
    end_time_get_inversion = time.time()-start_time_get_inversion


    # full_wall = axisymmetric_mesh_from_polygon(RZwall)

    print('time for input = ', end_time_get_parameters)
    print('time for equilibrium = ', end_time_get_equilibrium)
    print('time for inversion = ', end_time_get_inversion)
    try:
        Inversion_results.save()
    except:
        print('failed to save inversion results')
        pdb.set_trace()
    return Inversion_results



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

def get_transfert_matrix(Transfert_Matrix, realcam, world, ParamsMachine, ParamsGrid, ParamsVid, Inversion_results, RZwall = None):
    """
    return transfert_matrix for west"
    """
    #get wall coordinates to get the cherab object
    if ParamsGrid.symetry == 'magnetic': #overwrites wall with the one saved in pleque
        t_pleque = Inversion_results.t_inv[len(Inversion_results.t_inv)//2] #choose middle time of the inversion for time of equilibrium
        t_pleque = t_pleque*1000 # converting in ms for pleque
        import pleque.io.compass as plq
        print("magnetic symmetry")
        revision_mag = ParamsGrid.revision
        revision_mag = revision_mag or 1
        variant_mag =  ParamsGrid.variant_mag
        variant_mag = variant_mag or ''
        eq = plq.cdb(ParamsVid.nshot, t_pleque, revision = revision_mag, variant = variant_mag)
        RZwall = np.array([eq.first_wall.R,eq.first_wall.Z]).T
        RZwall =RZwall[:-1, :] 
        RZwall = RZwall[::-1]
    start = time.time()

    wall_limit = axisymmetric_mesh_from_polygon(RZwall)
    R_wall = RZwall[:, 0]
    Z_wall = RZwall[:, 1]


    visible_pix = np.where(Transfert_Matrix.mask_pixel) 
    pos_camera = realcam.pixel_origins[visible_pix[0][0]  , visible_pix[1][0]]
    pos_camera = np.array([pos_camera.x,pos_camera.y, pos_camera.z] )
    pos_camera_RPHIZ = utility_functions.xyztorphiz(pos_camera)
    if pos_camera_RPHIZ[1]<0: #setting [-pi, pi] interval into [0, 2pi] interval
        pos_camera_RPHIZ[1] = pos_camera_RPHIZ[1] + 2 *np.pi
    # R_max_noeud = pos_camera_RPHIZ[0] 
    # R_min_noeud = pos_camera_RPHIZ[0] 
    # Z_max_noeud = pos_camera_RPHIZ[2] 
    # Z_min_noeud = pos_camera_RPHIZ[2] 
    # R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, phi_max, phi_min = optimize_boundary_grid(realcam, world, R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud)
    if ParamsMachine.machine == 'WEST':
        R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud =[3.129871200000000, 1.834345700000000,0.798600000000000,-0.789011660000000]
    else:
        if ParamsGrid.phi_grid == 'auto':
            R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, phi_max, phi_min, RPHIZ = optimize_grid_from_los(realcam, world, pos_camera_RPHIZ)
            print(R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, phi_max, phi_min)
            phi_grid = (phi_max+phi_min)/2*180/np.pi
            print(phi_grid)

        else:
            phi_grid = ParamsGrid.phi_grid
        R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, dud1,dud2,  RPHIZ = optimize_grid_from_camera(realcam, phi_grid, RZwall,pos_camera_RPHIZ)
        print(R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, phi_max, phi_min)
    if ParamsGrid.extra_steps:
        R_max_noeud = R_max_noeud+ParamsGrid.extra_steps*ParamsGrid.dr_grid
        R_min_noeud = R_min_noeud-ParamsGrid.extra_steps*ParamsGrid.dr_grid
        Z_max_noeud = Z_max_noeud+ParamsGrid.extra_steps*ParamsGrid.dz_grid
        Z_min_noeud = Z_min_noeud-ParamsGrid.extra_steps*ParamsGrid.dz_grid
     # if verbose:
    #     fig = utility_functions.plot_cylindrical_coordinates(RPHIZ)
    #     fig = utility_functions.plot_line_from_cylindrical(pos_camera_RPHIZ, RPHIZ[0,0,:], fig, color = 'blue', label = 'point [0,0]')
    #     fig = utility_functions.plot_line_from_cylindrical(pos_camera_RPHIZ, RPHIZ[-1,0,:], fig, color = 'red', label = 'point [-1,0]')
    #     fig = utility_functions.plot_line_from_cylindrical(pos_camera_RPHIZ, RPHIZ[0,-1,:], fig, color = 'green', label = 'point [0,-1]')
    #     fig = utility_functions.plot_line_from_cylindrical(pos_camera_RPHIZ, RPHIZ[-1,-1,:], fig, color = 'yellow', label = 'point [-1,-1]')
    #     plt.savefig(main_folder_image + 'images line of sight and wall')
    extent_RZ =[R_min_noeud, R_max_noeud, Z_min_noeud, Z_max_noeud] 
    nb_noeuds_r = int((R_max_noeud-R_min_noeud)/ParamsGrid.dr_grid)
    nb_noeuds_z = int((Z_max_noeud-Z_min_noeud)/ParamsGrid.dz_grid)
    cell_r, cell_z, grid_mask, cell_dr, cell_dz = get_mask_from_wall(R_min_noeud, R_max_noeud, Z_min_noeud, Z_max_noeud, nb_noeuds_r, nb_noeuds_z, wall_limit, ParamsGrid.crop_center, ParamsGrid)
    # The RayTransferCylinder object is fully 3D, but for simplicity we're only
    # working in 2D as this case is axisymmetric. It is easy enough to pass 3D
    # views of our 2D data into the RayTransferCylinder object: we just ues a
    # numpy.newaxis (or equivalently, None) for the toroidal dimension.
    grid_mask = grid_mask[:, np.newaxis, :]
    if ParamsGrid.symetry =='magnetic':
        n_polar = ParamsGrid.n_polar
    else:
        n_polar = 1
    RZ_mask_grid = np.copy(grid_mask)
    grid_mask = np.tile(grid_mask, (1, n_polar, 1))
    num_points_rz = nb_noeuds_r*nb_noeuds_z
    # R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, phi_min, phi_max = optimize_grid(R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, realcam, phi_grid, RZwall)
    
    #recalculate the extremities of the grid; the grid starts at (r_min, z_min) and its last point is (r_max-dr_grid, z_max-dz_grid)

   
    if ParamsGrid.symetry =='magnetic':
                
        start = time.time()
        
        FL_MATRIX, dPhirad, phi_min, phi_mem = FL_lookup(eq, phi_grid, cell_r, cell_z, phi_min, phi_max, 0.5e-3, 0.2)
        end = time.time()

        elapsed = end - start
        print(f"Magnetic field lines calculation : {elapsed:.3f} seconds")

        
        cell_r_precision, cell_z_precision, grid_mask_precision, cell_dr_precision, cell_dz_precision = get_mask_from_wall(R_min_noeud, R_max_noeud, Z_min_noeud, Z_max_noeud, nb_noeuds_r*ParamsGrid.grid_precision_multiplier, nb_noeuds_z*ParamsGrid.grid_precision_multiplier, wall_limit, ParamsGrid.crop_center, ParamsGrid)
        grid_mask_precision = grid_mask_precision[:, np.newaxis, :]
        grid_mask_precision = np.tile(grid_mask_precision, (1, n_polar, 1))
        phi_tour = np.linspace(0, 360, n_polar, endpoint = False)
        phi_vision = (phi_tour>phi_min*180/np.pi) & (phi_tour<phi_max*180/np.pi) #only keep points in the grid into the esteemed line of sights
        grid_mask_precision[:, np.invert(phi_vision), :] = True
        phi_tour = phi_tour[phi_vision]
        plasma = RayTransferCylinder(radius_outer = R_max_noeud, 
                                    height= Z_max_noeud-Z_min_noeud, 
                                    n_radius = nb_noeuds_r*ParamsGrid.grid_precision_multiplier, 
                                    n_height = nb_noeuds_z*ParamsGrid.grid_precision_multiplier, 
                                    radius_inner = R_min_noeud,  
                                    transform=translate(0., 0., Z_min_noeud), 
                                    n_polar = n_polar, 
                                    mask = grid_mask_precision,
                                    period = 360)
        

        # plasma = RayTransferCylinder(R_max_noeud, nb_noeuds_z*dz_grid, nb_noeuds_r*grid_precision_multiplier, nb_noeuds_z*grid_precision_multiplier, radius_inner = R_min_noeud,  parent = world, transform=translate(0., 0., Z_min_noeud), n_polar = n_polar, period = 360, voxel_map = plasma.voxel_map)
        # plasma.voxel_map[~grid_mask] = -1 
        # pdb.set_trace()
        # voxel_map = create_voxel_map_from_equilibrium(FL_MATRIX, plasma, cell_r_precision, cell_z_precision, grid_mask_precision, cell_dr_precision, cell_dz_precision, phi_mem, dPhirad, wall_limit, dict_transfert_matrix)
        voxel_map = create_voxel_map_from_equilibrium_query(FL_MATRIX, plasma, cell_r_precision, cell_z_precision, grid_mask_precision, cell_dr_precision, cell_dz_precision, phi_mem, dPhirad, wall_limit)
    
        plasma2 = RayTransferCylinder(
            radius_outer=R_max_noeud,
            radius_inner=R_min_noeud,
            height=nb_noeuds_z*ParamsGrid.dz_grid,
            n_radius=nb_noeuds_r*ParamsGrid.grid_precision_multiplier, 
            n_height=nb_noeuds_z*ParamsGrid.grid_precision_multiplier,  
            n_polar=n_polar,
            mask = grid_mask_precision,
            voxel_map = voxel_map,
            period = 360,
            parent = world,
            transform=translate(0, 0, Z_min_noeud)
        )
        

    elif ParamsGrid.symetry == 'toroidal':
        phi_tour = n_polar
        plasma2 = RayTransferCylinder(radius_outer=cell_r[-1],
                                        radius_inner=cell_r[0],
                                        height=cell_z[-1] - cell_z[0],
                                        n_radius=nb_noeuds_r, n_height=nb_noeuds_z, 
                                        mask=grid_mask, n_polar=n_polar, 
                                        parent = world, transform=translate(0, 0, cell_z[0]))
    
    else:
        raise(NameError('unrecognized symetry, write toroidal or magnetic'))
    # if verbose:
    #     plt.figure()
    #     plt.imshow(np.sum(plasma.voxel_map, 1).T, extent= extent_RZ, origin = 'lower' )
    #     plt.show(block = False)
    #     plt.savefig(main_folder_image + '2D_voxel_map.png')
    
    #calculate inversion matrix
    print(plasma2.bins)
    print(num_points_rz)
    realcam.spectral_bins = plasma2.bins #set the grid to a size (NR, NZ) plus 1 extra node for elements of the grid too far from calculated field lines
    if realcam.spectral_bins >realcam.pixels[0]*realcam.pixels[1]:
        raise Exception("more nodes than pixels, inversion is impossible. Lower dr_grid or dz_grid")
    if realcam.spectral_bins >10000:
        print("careful, huge number of nodes")
    # if verbose:
    #     compare_voxel_map_and_pleque(plasma2, FL_MATRIX, Z_min_noeud, phi_mem)

    # if verbose:
    #     create_synth_cam_emitter(realcam, full_wall, R_wall, Z_wall, mask, path_CAD, variant = variant)
    
    
    end = time.time()

    elapsed = end - start
    print(f"time setup camera : {elapsed:.3f} seconds")

    realcam.observe()
    pipelines = realcam.pipelines[0]
    print('shape full transfert matrix = ' + str(pipelines.matrix.shape))
    flattened_matr = pipelines.matrix.reshape(pipelines.matrix.shape[0] * pipelines.matrix.shape[1], pipelines.matrix.shape[2])
    
    if flattened_matr.shape[1] > num_points_rz:
        #some elements of the grid don't see the field lines. Checking if they are out of the field of view of the camera
        invisible_nodes = np.sum(flattened_matr, 0)[-1]
        if invisible_nodes>0:
            print('nodes not seen, choose bigger grid limits, or wall limits differ between CAD model and magnetic equilibrium')
            # pdb.set_trace()
        flattened_matr = flattened_matr[:, :-1]
    print('flattened_matr shape', flattened_matr.shape)


    pixels,  = np.where(np.sum(flattened_matr, 1)) #sum over nodes
    noeuds,  = np.where(np.sum(flattened_matr, 0)) #sum over pixels
    #save results

    mask_pixel = np.zeros(flattened_matr.shape[0], dtype = bool)
    mask_pixel[pixels] = True
    mask_pixel = mask_pixel.reshape(pipelines.matrix.shape[0:2])

    # x, y, z = np.where(RZ_mask_grid)
    # x = x[noeuds]
    # y = y[noeuds]
    # # z = z[noeuds]
    if ParamsGrid.symetry == 'magnetic':
        mask_noeud = np.zeros_like(RZ_mask_grid, dtype = bool)
        rows_noeud, indphi, cols_noeud = np.unravel_index(noeuds, mask_noeud.shape)
        mask_noeud[rows_noeud,indphi, cols_noeud] = True
    elif ParamsGrid.symetry == 'toroidal':
        true_nodes = np.flatnonzero(RZ_mask_grid)
        mask_noeud = np.zeros_like(RZ_mask_grid, dtype = bool)
        mask_noeud.ravel()[true_nodes[noeuds]] = True
    print('shape voxel_map ', plasma2.voxel_map.shape)
    print('shape mask_noeud ', mask_noeud.shape)
    
    transfert_matrix = flattened_matr[pixels,:][:, noeuds]

    nb_visible_noeuds = len(np.unique(noeuds))
    nb_vision_pixel = len(np.unique(pixels))
    print('visible node = ' + str(nb_visible_noeuds) + 'out of ' + str(nb_noeuds_r*nb_noeuds_z))
    print('vision pixels = ' + str(nb_vision_pixel) + 'out of ' + str(pipelines.matrix.shape[0] * pipelines.matrix.shape[1]))

    transfert_matrix = csr_matrix(transfert_matrix)
    print(transfert_matrix.shape)
    pixels = np.squeeze(pixels)
    noeuds = np.squeeze(noeuds)
    
    print('shape reduced transfert matrix = ' + str(transfert_matrix.shape))
    # plt.figure()
    # plt.imshow(np.squeeze(mask_noeud).T, extent= extent_RZ, origin = 'lower' )
    # plt.savefig(main_folder_image + '2D_map_nodes.png')
    # if verbose:
    #     plt.show(block = False)
    Transfert_Matrix.transfert_matrix = transfert_matrix
    Transfert_Matrix.mask_noeud = mask_noeud
    Transfert_Matrix.mask_pixel = mask_pixel
    Transfert_Matrix.noeuds = noeuds
    Transfert_Matrix.pixels = pixels
    Transfert_Matrix.R_noeud = cell_r
    Transfert_Matrix.Z_noeud = cell_z
    Transfert_Matrix.pixels = pixels
    try:
        Transfert_Matrix.save()
    except:
        print('failed to save transfert matrix')
        pdb.set_trace()
        
    # try: 
    #     save_npz(path_transfert_matrix, transfert_matrix)
    # except:
    #     print('save of transfert matrix failed')
    #     pdb.set_trace()
    # try:
    #     dict_save_parameters = dict(pixels = pixels, 
    #                         noeuds = noeuds, 
    #                         nb_noeuds_r = nb_noeuds_r, 
    #                         nb_noeuds_z = nb_noeuds_z, 
    #                         R_max_noeud = R_max_noeud, 
    #                         R_min_noeud = R_min_noeud, 
    #                         Z_max_noeud = Z_max_noeud, 
    #                         Z_min_noeud = Z_min_noeud, 
    #                         R_noeud = cell_r,
    #                         Z_noeud = cell_z, 
    #                         mask_pixel = mask_pixel, 
    #                         mask_noeud= mask_noeud,
    #                         phi_tour = phi_tour)
    #     np.savez_compressed(path_parameters, **dict_save_parameters)
    #     path_parameters_save ,ext = os.path.splitext(path_parameters)
    #     savemat(path_parameters_save + '.mat', dict_save_parameters)
    # except:
    #     print('save of parameters failed')
    #     pdb.set_trace()
    
    return Transfert_Matrix
                  

                                  
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

    # Apraw the rotation to the coordinates
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
                phi_start = phi_eval[0]
               
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
    dPhirad = np.mean(np.diff(phi_mem))
    ind_PHI = int(np.ceil((phi_grid-phi_min)/dPhirad))
    # phi_mem[ind_PHI-1] = phi_grid
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
            raise ValueError("integration on toroidal angle incorrectly borned")
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
    


def reduce_camera_precision(camera, mask, vid, decimation =1):
    decimation = decimation or 1
    mask = downsample_with_avg(mask, decimation)
    # mask = mask[::decimation, ::decimation]
    vid_downsize = np.zeros((vid.shape[0], mask.shape[0], mask.shape[1]))
    for i in range(vid.shape[0]):
        vid_downsize[i, :, :] = downsample_with_avg(vid[i, :, :] , decimation)


    pixel_directions = downsample_with_avg(camera.pixel_directions, decimation)
    # pixel_directions = pixel_directions[::decimation, ::decimation]
    pixel_origins = camera.pixel_origins[::decimation, ::decimation]
    pixel_origins[np.invert(mask)] = Point3D(np.NaN, np.NaN, np.NaN)

    camera = VectorCamera(pixel_origins, pixel_directions)

    return camera, mask, vid_downsize


def downsample_with_avg(matrix, block_size=4):
    # Get the shape of the matrix
    rows, cols = matrix.shape

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
                if matrix.dtype == bool:
                    if block.all() == True:
                        result[i//block_size, j//block_size] = True
                    else:
                        result[i//block_size, j//block_size] = False

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
                                 inv_image_thresholded, 
                                 inv_image_thresholded_normed, 
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
    inv_image_thresholded_full = reconstruct_2D_image(inv_image_thresholded, mask_noeud, nb_noeuds_r, nb_noeuds_z)
    inv_image_thresholded_normed_full = reconstruct_2D_image(inv_image_thresholded_normed, mask_noeud, nb_noeuds_r, nb_noeuds_z)
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
    plt.imshow(inv_image_thresholded_full.T, extent = extent, origin = 'lower')
    plt.colorbar()
    plt.title('inversed image thresholded, c_c = '+ str(c_c))
    plt.plot(R_wall, Z_wall, 'r')
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    #normed inversion
    plt.subplot(2,4, 8)
    plt.imshow(inv_image_thresholded_normed_full.T, extent = extent, origin = 'upper')
    plt.colorbar()
    plt.title('inversed image thresholded and normalized, c_c = '+ str(c_c))
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






# def read_CAD(path_CAD, world, name_material = 'AbsorbingSurface', wall_material = AbsorbingSurface(), variant = 'default'):

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


# def read_CALCAM_CAD(path_CAD, world, wall_materials):
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


def remove_center_from_inversion(vertex_mask, cell_vertices_r, cell_vertices_z, ParamsGrid):
    nshot = ParamsGrid.nshot
    

    try:
        magflux = utility_functions.call_module_function_in_environment('west_functions','get_equilibrium', 'python-3.11', nshot)

        r = magflux.interp2D.r
        z = magflux.interp2D.z
        idx = magflux.interp2D.psi.shape[0]//2
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
    except:
        magflux = loadmat('/Home/LF285735/Documents/MATLAB/shot 61357 sep 7_7s.mat', simplify_cells = True)
        magflux = magflux['equi']
        r = magflux['interp2D']['r']
        z = magflux['interp2D']['z']
        idx = magflux['interp2D']['psi'].shape[0]//2
        psi = magflux['interp2D']['psi'][idx, :, :]
        psisep = magflux['boundary']['psi'][idx]
        psi0 = np.nanmax(psi)
        psi_int = 0.9*(psisep-psi0)+psi0
    # Create contour for just this isovalue
    contour = plt.contour(r, z, psi, levels=[psi_int], colors='red')
    plt.close()
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
    features = CAD.get_enabled_features()
    components_list = set(components.keys())
    enabled_features = list(components_list & set(features))
    print(features)

    CAD.enable_only(enabled_features)

    path_stl = [CAD.features[feature].filename for feature in enabled_features] 

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

    #setting [-pi, pi] intervall back into [0, 2pi] intervall
    RPHIZ[:, :, 1] = (RPHIZ[:, :, 1] +2*np.pi) % (2*np.pi)
    
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
            mask_pixel = np.ascontiguousarray(mask_pixel)
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
    mask_pixel.dtype = bool
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





def get_mask_from_wall(R_min_noeud, R_max_noeud, Z_min_noeud, Z_max_noeud, nb_noeuds_r, nb_noeuds_z, wall_limit, crop_center, ParamsGrid):

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
    if crop_center:
        vertex_mask = remove_center_from_inversion(vertex_mask, cell_vertices_r, cell_vertices_z, ParamsGrid)

    # Cell is included if at least one vertex is within the wall
    grid_mask = (vertex_mask[1:, :-1] + vertex_mask[:-1, :-1]
                + vertex_mask[1:, 1:] + vertex_mask[:-1, 1:])
    grid_mask=grid_mask>0
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
    return idxs, grid_points, dists







def create_voxel_map_from_equilibrium(FL_MATRIX, plasma, cell_r_precision, cell_z_precision, grid_mask_precision, cell_dr_precision, cell_dz_precision, phi_mem, dPhirad, wall_limit, dict_transfert_matrix):

    # turning angle back into degrees

    nb_noeuds_r = len(cell_r_precision)
    nb_noeuds_z = len(cell_z_precision)

    phi_min = phi_mem[0]*180/np.pi
    phi_max = phi_mem[-1]*180/np.pi
    # seuil = np.sqrt((plasma.material.dr*grid_precision_multiplier)**2+(plasma.material.dz*grid_precision_multiplier)**2)*3
    seuil = np.sqrt((np.mean(np.diff(cell_r_precision)))**2+(np.mean(np.diff(cell_z_precision)))**2)
    n_polar = plasma.material.grid_shape[1]
    voxel_map = -1*np.ones_like(plasma.voxel_map) #setting all nodes to blind 
    for nphi in range(n_polar):
        if nphi*plasma.material.dphi>phi_max or nphi*plasma.material.dphi<phi_min:
            voxel_map[:, nphi, :] = -1
            # print('phi out of range')
        else:
            ind_phi_closest = np.round((nphi*plasma.material.dphi-phi_min)/(dPhirad*180/np.pi)).astype('int')
            #print('phi in range')
            for i in range(nb_noeuds_r):
                for j in range(nb_noeuds_z):
                    noeud_r = cell_r_precision[0] + plasma.material.dr*i
                    noeud_z = cell_z_precision[0] + plasma.material.dz*j

                    if noeud_r != cell_r_precision[i] or noeud_z != cell_z_precision[j]:
                        pdb.set_trace()

                        raise(ValueError('careful, grid ill defined'))
                    pointrz = Point3D(noeud_r, 0, noeud_z)
                    if grid_mask_precision[i,j]:
                        dist = np.sqrt((noeud_r-FL_MATRIX[:, 0, ind_phi_closest])**2+(noeud_z-FL_MATRIX[:, 1, ind_phi_closest])**2)
                        argmin = np.nanargmin(dist)
                        minlos = dist[argmin]

                        if minlos < seuil:
                            voxel_map[i, nphi, j] = argmin
                        else:
                            #set the element of the grid to a virtual node (not related to a position R, Z of the 2D map) for debugging
                            voxel_map[i, nphi, j] = nb_noeuds_z*nb_noeuds_r #careful of indexing, last real point of voxel map is num_points_rz-1
                            # pdb.set_trace()
                            # raise(ValueError('element of plasma too far from calculated magnetic lines'))
                            # plasma.voxel_map[i, nphi, j] = -1
                    else:
                        voxel_map[i, nphi, j] = -1
    print(np.max(voxel_map))
    return voxel_map



def create_voxel_map_from_equilibrium_query(FL_MATRIX, plasma, cell_r_precision, cell_z_precision, grid_mask_precision, cell_dr_precision, cell_dz_precision, phi_mem, dPhirad, wall_limit):

    # turning angle back into degrees

    nb_noeuds_r = len(cell_r_precision)
    nb_noeuds_z = len(cell_z_precision)
    R, Z = np.meshgrid(cell_r_precision, cell_z_precision, indexing = 'ij')

    phi_min = phi_mem[0]*180/np.pi
    phi_max = phi_mem[-1]*180/np.pi
    # seuil = np.sqrt((plasma.material.dr*grid_precision_multiplier)**2+(plasma.material.dz*grid_precision_multiplier)**2)*3
    seuil = np.sqrt((np.mean(np.diff(cell_r_precision)))**2+(np.mean(np.diff(cell_z_precision)))**2)
    n_polar = plasma.material.grid_shape[1]
    voxel_map = -1*np.ones_like(plasma.voxel_map) #setting all nodes to blind 
    query_points = np.column_stack((R.flatten(), Z.flatten()))

    for nphi in range(n_polar):
        if nphi*plasma.material.dphi>phi_max or nphi*plasma.material.dphi<phi_min:
            voxel_map[:, nphi, :] = -1
            # print('phi out of range')
        else:
        
            ind_phi_closest = np.round((nphi*plasma.material.dphi-phi_min)/(dPhirad*180/np.pi)).astype('int')
            #print('phi in range')
            grid_points = FL_MATRIX[:, :, ind_phi_closest] 

            idxs, _,  dists = closest_points(grid_points, query_points)
            voxel_map[:, nphi, :] = idxs.reshape(nb_noeuds_r, nb_noeuds_z)

    voxel_map[np.invert(grid_mask_precision)] = -1
    print(np.max(voxel_map))
    return voxel_map



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



def optimize_grid_from_camera(realcam, phi_grid, RZwall, pos_camera_RPHIZ):
    from matplotlib.pyplot import ion, ioff
    world_intersection = World()
    realcam_intersection = VectorCamera(realcam.pixel_origins, realcam.pixel_directions, parent = world_intersection, frame_sampler = FullFrameSampler2D(realcam.frame_sampler.mask))
    x_lower = np.nanmin(RZwall[:, 0])
    y_lower = 0
    z_lower = np.nanmin(RZwall[:, 1])
    lower_point = Point3D(x_lower, y_lower, z_lower)

    x_upper = np.nanmax(RZwall[:, 0])
    y_upper = 0.01
    z_upper = np.nanmax(RZwall[:, 1])
    upper_point = Point3D(x_upper, y_upper, z_upper)

    fake_wall = Box(lower_point, upper_point, material=AbsorbingSurface(), parent=world_intersection, transform = rotate_z(phi_grid))
    RPHIZ, Intersection = check_intersection_wall(realcam_intersection, world_intersection)
    R_min_noeud = pos_camera_RPHIZ[0] 
    R_max_noeud = pos_camera_RPHIZ[0] 
    Z_min_noeud = pos_camera_RPHIZ[2] 
    Z_max_noeud = pos_camera_RPHIZ[2] 


    # for i in range(RPHIZ.shape[0]): 
    #     for j in range(RPHIZ.shape[1]):

    #         R_min, R_max, Z_min, Z_max = utility_functions.find_RZextrema_between_2_points(pos_camera_RPHIZ, RPHIZ[i, j, :])
    #         R_min_noeud = min(R_min_noeud, R_min)
    #         R_max_noeud = max(R_max_noeud, R_max)
    #         Z_min_noeud = min(Z_min_noeud, Z_min)
    #         Z_max_noeud = max(Z_max_noeud, Z_max)

    R_max_noeud = np.nanmax(RPHIZ[:, :, 0])
    R_min_noeud = np.nanmin(RPHIZ[:, :, 0])
    Z_max_noeud = np.nanmax(RPHIZ[:, :, 2])
    Z_min_noeud = np.nanmin(RPHIZ[:, :, 2])
    
    
    #setting [-pi, pi] intervall back into [0, 2pi] intervall
    RPHIZ[:, :, 1] = (RPHIZ[:, :, 1] +2*np.pi) % (2*np.pi)
    
    phi_min = min(pos_camera_RPHIZ[1], np.nanmin(RPHIZ[:, :, 1]))
    phi_max = max(pos_camera_RPHIZ[1], np.nanmax(RPHIZ[:, :, 1]))

    R_max_noeud = min(R_max_noeud, np.nanmax(RZwall[:, 0]))
    R_min_noeud = max(R_min_noeud, np.nanmin(RZwall[:, 0]))
    Z_max_noeud = min(Z_max_noeud, np.nanmax(RZwall[:, 1]))
    Z_min_noeud = max(Z_min_noeud, np.nanmin(RZwall[:, 1]))
        
    world_rgb =World()
    wall_limit = axisymmetric_mesh_from_polygon(RZwall)
    wall_limit.parent = world_rgb
    wall_limit.material = UniformVolumeEmitter(colours[1])
    fake_wall = Box(lower_point, upper_point, material=UniformVolumeEmitter(colours[0]), parent=world_rgb, transform = rotate_z(phi_grid))

    rgb = RGBPipeline2D()
    sampler = RGBAdaptiveSampler2D(rgb, min_samples=100, fraction=0.2, cutoff=0.01)
    # camera = VectorCamera(realcam.pixel_origins, realcam.pixel_directions, parent=world_rgb, pipelines=[rgb], frame_sampler=sampler)

    camera = PinholeCamera((128, 256), parent=world_rgb, pipelines=[rgb], frame_sampler=sampler, transform = translate(0, 0, -4))
    camera.spectral_rays = 1
    camera.spectral_bins = 15
    camera.pixel_samples = 250

    ion()
    name = 'transform test'
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    render_pass = 1
    camera.observe()
    # rgb.save("{}_{}_pass_{}.png".format(name, timestamp, render_pass))
    # while not camera.render_complete:

    #     print("Rendering pass {}...".format(render_pass))
    #     camera.observe()
    #     rgb.save("{}_{}_pass_{}.png".format(name, timestamp, render_pass))
    #     print()

    #     render_pass += 1

    ioff()
    rgb.display()

    return R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, phi_max, phi_min, RPHIZ



def load_components(namefile, features):
    import yaml
    instances = []
    namefile = namefile if namefile.endswith(".yaml") else namefile + ".yaml"
    with open(namefile, "r") as f:
        config = yaml.safe_load(f)

    components_cfg = config["components"]
    for name in features:
        comp_cfg = components_cfg.get(name)
        if comp_cfg is None:
            raise ValueError(f"Component '{name}' not found in YAML.")

        cls_name = comp_cfg["class"]
        params = comp_cfg.get("params", {})
        
        # Assume class is already imported
        cls = globals()[cls_name]
        instance = cls(**params)
        instances.append(instance) 
    
    return instances



def read_CAD_from_raw(ParamsMachine, world):
    #fetch enabled wall components in main components file in Tomography/ressources
    components_list = set(components.keys())

    #load names of all components stored in the raw files
    features_files = os.listdir(ParamsMachine.path_CAD)
    features = [os.path.splitext(features)[0] for features in features_files] #remove raw extension from name 
    
    #select only features listed in the main components files
    enabled_features = list(set(features) & components_list)
    ext_map = {os.path.splitext(f)[0]: os.path.splitext(f)[1] for f in features_files}

    extensions = [ext_map[name] for name in enabled_features]
    path_enabled_features = [ParamsMachine.path_CAD + features + extensions[i] for i, features in enumerate(enabled_features)] #full path for enabled components

    #set up Node for better hierarchy of wall components (relevant for raytracing computing memory)
    wall_group = Node(parent=world)

    #get extension type for file
    if ParamsMachine.name_material == "absorbing_surface":
        for i, f in enumerate(path_enabled_features):
            if extensions[i] == '.ply':
                full_wall =  import_ply(f, parent = wall_group , material = AbsorbingSurface(), name = enabled_features[i]) 
            elif extensions[i] == '.stl':
                full_wall =  import_stl(f, parent = wall_group , material = AbsorbingSurface(), name = enabled_features[i]) 
            elif extensions[i] == '.npy':
                RZ = np.load(f)
                full_wall = axisymmetric_mesh_from_polygon(RZ)
                full_wall.parent = wall_group
                full_wall.material = AbsorbingSurface()
                full_wall.name = enabled_features[i]
            else:
                raise(NameError('unrecognised 3D files extension'))
    else:
        # ParamsMachine.name_material should be the path to a components file containing each type of the enabled components
        wall_materials = load_components(ParamsMachine.name_material, enabled_features)
        for i, f in enumerate(path_enabled_features):
            if extensions[i] == '.ply':
                full_wall =  import_ply(f, parent = wall_group , material = wall_materials[i], name = enabled_features[i]) 
            elif extensions[i] == '.stl':
                full_wall =  import_stl(f, parent = wall_group , material = wall_materials[i], name = enabled_features[i]) 
            elif extensions[i] == '.npy':
                    RZ = np.load(f)
                    full_wall = axisymmetric_mesh_from_polygon(RZ)
                    full_wall.parent = wall_group
                    full_wall.material = wall_materials[i]
            else:
                raise(NameError('unrecognised 3D files extension'))
    return full_wall

def read_CAD_from_components(ParamsMachine, world):
    import calcam
    CAD = calcam.CADModel(ParamsMachine.path_CAD, model_variant = ParamsMachine.variant_CAD)
    features = CAD.get_enabled_features()

    #read the file Tomography/ressources/components.yaml, listing each elements of the CAD model that need to be enabled.
    components_list = set(components.keys())
    enabled_features = list(components_list & set(features))
    print(features)
    CAD.enable_only(enabled_features)
    path_stl = [CAD.features[feature].filename for feature in enabled_features] 

    #create node to load all meshes for more efficient loading
    wall_group = Node(parent=world)

    #load each enabled features, depending on the type of materials
    if ParamsMachine.name_material == "absorbing_surface":
        # meshes in the CAD model on West are oriented (up direction) in the +Y direction, they need to be rotated.
        if ParamsMachine.machine == 'WEST':
            full_wall =  [import_stl(f, parent = wall_group, scaling = 0.001 , material = AbsorbingSurface(), name = features[i], transform = rotate(0, -90,0)) for i, f in enumerate(path_stl)]
        # meshes in the CAD model on Compass are oriented (up direction) in the +Z direction, no need for rotation
        elif ParamsMachine.machine == 'COMPASS':
            full_wall =  [import_stl(f, parent = wall_group, scaling = 0.001 , material = AbsorbingSurface(), name = features[i]) for i, f in enumerate(path_stl)]
        else:
            raise(NameError('unrecognized machine'))
    else:
        # Look into the file (path in ParamsMachine.name_material) to assign material type to each enabled features
        wall_materials = load_components(ParamsMachine.name_material, enabled_features)

        # meshes in the CAD model on West are oriented (up direction) in the +Y direction, they need to be rotated.
        if ParamsMachine.machine == 'WEST':
            full_wall =  [import_stl(f, parent = wall_group, scaling = 0.001 , material = wall_materials[i], name = features[i],transform = rotate(0, -90,0)) for i, f in enumerate(path_stl)]
        # meshes in the CAD model on Compass are oriented (up direction) in the +Z direction, no need for rotation
        elif ParamsMachine.machine == 'COMPASS':
            full_wall =  [import_stl(f, parent = wall_group, scaling = 0.001 , material = wall_materials[i], name = features[i]) for i, f in enumerate(path_stl)]
        else:
            raise(NameError('unrecognized machine'))
    CAD.unload()
    return full_wall




def geometry_matrix_spectro(ParamsMachine, ParamsGrid):
    from scipy.io import loadmat
    LOS = loadmat('/Home/NF216031/MATLAB_NF/WEST_functions/DVIS/LOS_position_name.mat', struct_as_record=False, squeeze_me=True)
    LOS =utility_functions.matstruct_to_dict(LOS['dat'])

    ind_LODIV = [i for i, x in enumerate(LOS.name) if 'LODIV' in x]
    name = LOS.name[ind_LODIV]
    R1 = LOS.R1[ind_LODIV]
    Z1 = LOS.Z1[ind_LODIV]
    PHI1 = LOS.PHI1[ind_LODIV]
    R2 = LOS.R2[ind_LODIV]
    Z2 = LOS.Z2[ind_LODIV]
    PHI2 = LOS.PHI2[ind_LODIV]
    offset_angles = np.mean(PHI1)
    PHI1 = PHI1-offset_angles+200 #place the LOS inside the cut part of the tokamak
    PHI2 = PHI2-offset_angles+200 #apply same shift to endpoints
    deg2rad = np.pi/180
    X1 = R1*np.cos(PHI1*deg2rad)
    Y1 = R1*np.sin(PHI1*deg2rad)
    X2 = R2*np.cos(PHI2*deg2rad)
    Y2 = R2*np.sin(PHI2*deg2rad)
    x = X2-X1
    y = Y2-Y1
    z = Z2-Z1

    LOS =LOS
    p = np.vstack((x, y, z)).T

    # Direction vectors

    # Compute magnitudes
    mag = np.linalg.norm(p, axis=1, keepdims=True)

    # Avoid divide-by-zero
    mag[mag == 0] = np.nan

    # Normalize
    v_norm = p / mag
    pixel_origins = np.array([Point3D(X1[i], Y1[i], Z1[i]) for i in range(X1.shape[0])])

    pixel_directions = np.array([Vector3D(v_norm[v, 0], v_norm[v, 1], v_norm[v, 2]) for v in range(v_norm.shape[0])])
    world = World()
    # wall_CAD = read_CAD_from_components(ParamsMachine, world)

    if ParamsMachine.path_wall:
        try:
            fwall = loadmat(ParamsMachine.path_wall)
            RZwall = fwall['RZwall']

        except:
            RZwall = np.load(ParamsMachine.path_wall, allow_pickle=True)
        #check that the last element is the neighbor of the first one and not the same one
        if(RZwall[0]==RZwall[-1]).all():
            RZwall = RZwall[:-1]
            print('erased last element of wall')
        #check that the wall coordinates are stocked in a counter clockwise position. If not, reverse it
        Trigo_orientation, signed_area = utility_functions.polygon_orientation(RZwall[:, 0], RZwall[:, 1])
        if Trigo_orientation:
            RZwall = RZwall[::-1]
            print('wall reversed')

        R_wall = RZwall[:, 0]
        Z_wall = RZwall[:, 1]
    else:
        RZwall = None

    R_wall = RZwall[:, 0]
    Z_wall = RZwall[:, 1]

    wall_limit = axisymmetric_mesh_from_polygon(RZwall)
    flag_wall_limit = 0
    if flag_wall_limit:
        name_wall_material = 'tungsten'
        wall_limit.parent = world
        if name_wall_material == 'absorbing surface':
            wall_limit.material = AbsorbingSurface()
        else:
            wall_limit.material = RoughTungsten(0.5)
    else:
        full_wall = load_walls(ParamsMachine, world)

    
    if ParamsMachine.machine == 'WEST':
        R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud =[3.129871200000000, 1.834345700000000,0.798600000000000,-0.789011660000000]
    
       
    if ParamsGrid.extra_steps:
        R_max_noeud = R_max_noeud+ParamsGrid.extra_steps*ParamsGrid.dr_grid
        R_min_noeud = R_min_noeud-ParamsGrid.extra_steps*ParamsGrid.dr_grid
        Z_max_noeud = Z_max_noeud+ParamsGrid.extra_steps*ParamsGrid.dz_grid
        Z_min_noeud = Z_min_noeud-ParamsGrid.extra_steps*ParamsGrid.dz_grid
    extent_RZ =[R_min_noeud, R_max_noeud, Z_min_noeud, Z_max_noeud] 
    nb_noeuds_r = int((R_max_noeud-R_min_noeud)/ParamsGrid.dr_grid)
    nb_noeuds_z = int((Z_max_noeud-Z_min_noeud)/ParamsGrid.dz_grid)
    cell_r, cell_z, grid_mask, cell_dr, cell_dz = get_mask_from_wall(R_min_noeud, R_max_noeud, Z_min_noeud, Z_max_noeud, nb_noeuds_r, nb_noeuds_z, wall_limit, ParamsGrid.crop_center, ParamsGrid)
    # The RayTransferCylinder object is fully 3D, but for simplicity we're only
    # working in 2D as this case is axisymmetric. It is easy enough to pass 3D
    # views of our 2D data into the RayTransferCylinder object: we just ues a
    # numpy.newaxis (or equivalently, None) for the toroidal dimension.
    grid_mask = grid_mask[:, np.newaxis, :]
    
    n_polar = 1
    RZ_mask_grid = np.copy(grid_mask)
    grid_mask = np.tile(grid_mask, (1, n_polar, 1))
    num_points_rz = nb_noeuds_r*nb_noeuds_z
    step = 1e-4
    plasma2 = RayTransferCylinder(radius_outer=cell_r[-1],
                                        radius_inner=cell_r[0],
                                        height=cell_z[-1] - cell_z[0],
                                        n_radius=nb_noeuds_r, n_height=nb_noeuds_z, 
                                        step=step, n_polar=n_polar, 
                                        parent = world, transform=translate(0, 0, cell_z[0]))
 
    real_pipeline = RayTransferPipeline2D()
    
    camera = VectorCamera(pixel_origins[np.newaxis, :], pixel_directions[np.newaxis, :], parent = world)
    pixel_samples = 100
    # TRANSFORM = translate()
    camera.frame_sampler=FullFrameSampler2D()
    camera.pipelines=[real_pipeline]
    camera.pixel_samples = pixel_samples
    camera.min_wavelength = 640
    camera.max_wavelength = camera.min_wavelength +1
    camera.render_engine.processes = 32
    camera.spectral_bins = plasma2.bins
    camera.observe()
    pipelines = camera.pipelines[0]
    print('shape full transfert matrix = ' + str(pipelines.matrix.shape))
    flattened_matr = pipelines.matrix.reshape(pipelines.matrix.shape[0] * pipelines.matrix.shape[1], pipelines.matrix.shape[2])
    
    if flattened_matr.shape[1] > num_points_rz:
        #some elements of the grid don't see the field lines. Checking if they are out of the field of view of the camera
        invisible_nodes = np.sum(flattened_matr, 0)[-1]
        if invisible_nodes>0:
            print('nodes not seen, choose bigger grid limits, or wall limits differ between CAD model and magnetic equilibrium')
            # pdb.set_trace()
        flattened_matr = flattened_matr[:, :-1]
    print('flattened_matr shape', flattened_matr.shape)


    pixels,  = np.where(np.sum(flattened_matr, 1)) #sum over nodes
    noeuds,  = np.where(np.sum(flattened_matr, 0)) #sum over pixels
    #save results

    mask_pixel = np.zeros(flattened_matr.shape[0], dtype = bool)
    mask_pixel[pixels] = True
    mask_pixel = mask_pixel.reshape(pipelines.matrix.shape[0:2])

    mask_noeud = np.zeros_like(RZ_mask_grid, dtype = bool)
    rows_noeud, indphi, cols_noeud = np.unravel_index(noeuds, mask_noeud.shape)
    mask_noeud[rows_noeud,indphi, cols_noeud] = True
    print('shape voxel_map ', plasma2.voxel_map.shape)
    print('shape mask_noeud ', mask_noeud.shape)
    
    transfert_matrix = flattened_matr[pixels,:][:, noeuds]

    nb_visible_noeuds = len(np.unique(noeuds))
    nb_vision_pixel = len(np.unique(pixels))
    print('visible node = ' + str(nb_visible_noeuds) + 'out of ' + str(nb_noeuds_r*nb_noeuds_z))
    print('vision pixels = ' + str(nb_vision_pixel) + 'out of ' + str(pipelines.matrix.shape[0] * pipelines.matrix.shape[1]))

    transfert_matrix = csr_matrix(transfert_matrix)
    print(transfert_matrix.shape)
    pixels = np.squeeze(pixels)
    noeuds = np.squeeze(noeuds)


    name_folder = 'mat_saves/'
    name_material = ParamsMachine.name_material.split('/')[-1]
    if os.path.splitext(ParamsMachine.path_CAD)[1]:
        name_CAD = os.path.splitext(ParamsMachine.path_CAD)[1]
    else:
        name_CAD = ParamsMachine.path_CAD.split('/')[-2]
    if flag_wall_limit:
        name_CAD = '2D_wall_mesh'
        name_material = name_wall_material
    dict_spectro = dict(R1 = R1, R2= R2, Z1=Z1, Z2= Z2, grid_mask = grid_mask, extent_RZ = extent_RZ, R_wall = R_wall,  Z_wall= Z_wall, transfert_matrix = transfert_matrix, mask_pixel= mask_pixel, mask_noeud= mask_noeud,  pipelines= pipelines, cell_r = cell_r, cell_z= cell_z)
    savemat(name_folder + name_CAD+name_material+'dr'+str(ParamsGrid.dr_grid)+'step'+str(step)+'pixel_samples'+ str(pixel_samples)+'spectro_los.mat', dict_spectro)
    utility_functions.plot_image(np.squeeze(mask_noeud).T, extent = extent_RZ, origin = 'lower')
    plt.savefig(name_folder + name_CAD+name_material+'dr'+str(ParamsGrid.dr_grid)+'step'+str(step)+'pixel_samples'+ str(pixel_samples)+'view spectro los.png')
    plt.close()
    return R1, R2, Z1, Z2, grid_mask, extent_RZ, R_wall, Z_wall, transfert_matrix, mask_pixel, mask_noeud, pipelines, cell_r, cell_z


def load_walls(ParamsMachine, world):
    filename, ext = os.path.splitext(ParamsMachine.path_CAD)
    if ext == '':
        full_wall = read_CAD_from_raw(ParamsMachine, world)
    elif ext == '.ccm': 
        full_wall = read_CAD_from_components(ParamsMachine, world)
    else:
        raise(NameError('cannot read 3D files, extension {ext} unrecognised' ))
        
    return full_wall
                

def test_wall_cherab(ParamsMachine, ParamsGrid):
    from scipy.io import loadmat
    LOS = loadmat('/Home/NF216031/MATLAB_NF/WEST_functions/DVIS/LOS_position_name.mat', struct_as_record=False, squeeze_me=True)
    LOS =utility_functions.matstruct_to_dict(LOS['dat'])

    ind_LODIV = [i for i, x in enumerate(LOS.name) if 'LODIV' in x]
    name = LOS.name[ind_LODIV]
    R1 = LOS.R1[ind_LODIV]
    Z1 = LOS.Z1[ind_LODIV]
    PHI1 = 200*np.pi/180
    X1 = R1*np.cos(PHI1)
    Y1 = R1*np.sin(PHI1)
    
   

    if ParamsMachine.path_wall:
        try:
            fwall = loadmat(ParamsMachine.path_wall)
            RZwall = fwall['RZwall']

        except:
            RZwall = np.load(ParamsMachine.path_wall, allow_pickle=True)
        #check that the last element is the neighbor of the first one and not the same one
        if(RZwall[0]==RZwall[-1]).all():
            RZwall = RZwall[:-1]
            print('erased last element of wall')
        #check that the wall coordinates are stocked in a counter clockwise position. If not, reverse it
        Trigo_orientation, signed_area = utility_functions.polygon_orientation(RZwall[:, 0], RZwall[:, 1])
        if Trigo_orientation:
            RZwall = RZwall[::-1]
            print('wall reversed')

        R_wall = RZwall[:, 0]
        Z_wall = RZwall[:, 1]
    else:
        RZwall = None

    R_wall = RZwall[:, 0]
    Z_wall = RZwall[:, 1]

    #simplify starting los
    n = len(R_wall)
    X1 = X1[0]*np.ones(n)
    Y1 = Y1[0]*np.ones(n)
    Z1 = Z1[0]*np.ones(n)
    X_wall = R_wall*np.cos(PHI1)
    Y_wall = R_wall*np.sin(PHI1)
    x = X_wall-X1
    y = Y_wall-Y1
    z = Z_wall-Z1
    p = np.vstack((x, y, z)).T

    # Direction vectors

    # Compute magnitudes
    mag = np.linalg.norm(p, axis=1, keepdims=True)

    # Avoid divide-by-zero
    mag[mag == 0] = np.nan

    # Normalize
    v_norm = p / mag
    
    los = np.array([500])
    los = np.arange(len(R_wall)) #array of the LOS I want to plot. Here plot every LOS
    n_los = len(los)
    pixel_origins = np.array([Point3D(X1[v], Y1[v], Z1[v]) for v in los]) 
    pixel_directions =  np.array([Vector3D(v_norm[v, 0], v_norm[v, 1], v_norm[v, 2]) for v in los])
    world = World()

    # wall_CAD = read_CAD_from_components(ParamsMachine, world)


    wall_limit = axisymmetric_mesh_from_polygon(RZwall)
    flag_wall_limit = 0
    if flag_wall_limit:
        name_wall_material = 'tungsten'
        wall_limit.parent = world
        if name_wall_material == 'absorbing surface':
            wall_limit.material = AbsorbingSurface()
        else:
            wall_limit.material = RoughTungsten(0.5)
    else:
        full_wall = load_walls(ParamsMachine, world)

    if ParamsMachine.machine == 'WEST':
        R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud =[3.129871200000000, 1.834345700000000,0.798600000000000,-0.789011660000000]
    
    if ParamsGrid.extra_steps:
        R_max_noeud = R_max_noeud+ParamsGrid.extra_steps*ParamsGrid.dr_grid
        R_min_noeud = R_min_noeud-ParamsGrid.extra_steps*ParamsGrid.dr_grid
        Z_max_noeud = Z_max_noeud+ParamsGrid.extra_steps*ParamsGrid.dz_grid
        Z_min_noeud = Z_min_noeud-ParamsGrid.extra_steps*ParamsGrid.dz_grid
    
    extent_RZ =[R_min_noeud, R_max_noeud, Z_min_noeud, Z_max_noeud] 
    nb_noeuds_r = int((R_max_noeud-R_min_noeud)/ParamsGrid.dr_grid)
    nb_noeuds_z = int((Z_max_noeud-Z_min_noeud)/ParamsGrid.dz_grid)
    cell_r, cell_z, grid_mask, cell_dr, cell_dz = get_mask_from_wall(R_min_noeud, R_max_noeud, Z_min_noeud, Z_max_noeud, nb_noeuds_r, nb_noeuds_z, wall_limit, ParamsGrid.crop_center, ParamsGrid)
    # The RayTransferCylinder object is fully 3D, but for simplicity we're only
    # working in 2D as this case is axisymmetric. It is easy enough to pass 3D
    # views of our 2D data into the RayTransferCylinder object: we just ues a
    # numpy.newaxis (or equivalently, None) for the toroidal dimension.
    grid_mask = grid_mask[:, np.newaxis, :]
    
    n_polar = 1
    RZ_mask_grid = np.copy(grid_mask)
    grid_mask = np.tile(grid_mask, (1, n_polar, 1))
    num_points_rz = nb_noeuds_r*nb_noeuds_z
    step = 1e-4
    plasma2 = RayTransferCylinder(radius_outer=cell_r[-1],
                                        radius_inner=cell_r[0],
                                        height=cell_z[-1] - cell_z[0],
                                        n_radius=nb_noeuds_r, n_height=nb_noeuds_z, 
                                        n_polar=n_polar, 
                                        step = step,
                                        parent = world, transform=translate(0, 0, cell_z[0]))
 
    real_pipeline = RayTransferPipeline2D()
    
    camera = VectorCamera(pixel_origins[np.newaxis, :], pixel_directions[np.newaxis, :], parent = world)
    pixel_samples = 1
    # TRANSFORM = translate()
    camera.frame_sampler=FullFrameSampler2D()
    camera.pipelines=[real_pipeline]
    camera.pixel_samples = pixel_samples
    camera.min_wavelength = 640
    camera.max_wavelength = camera.min_wavelength +1
    camera.render_engine.processes = 32
    camera.spectral_bins = plasma2.bins
    camera.observe()
    pipelines = camera.pipelines[0]
    print('shape full transfert matrix = ' + str(pipelines.matrix.shape))
    flattened_matr = pipelines.matrix.reshape(pipelines.matrix.shape[0] * pipelines.matrix.shape[1], pipelines.matrix.shape[2])
    
    if flattened_matr.shape[1] > num_points_rz:
        #some elements of the grid don't see the field lines. Checking if they are out of the field of view of the camera
        invisible_nodes = np.sum(flattened_matr, 0)[-1]
        if invisible_nodes>0:
            print('nodes not seen, choose bigger grid limits, or wall limits differ between CAD model and magnetic equilibrium')
            # pdb.set_trace()
        flattened_matr = flattened_matr[:, :-1]
    print('flattened_matr shape', flattened_matr.shape)


    pixels,  = np.where(np.sum(flattened_matr, 1)) #sum over nodes
    noeuds,  = np.where(np.sum(flattened_matr, 0)) #sum over pixels
    #save results

    mask_pixel = np.zeros(flattened_matr.shape[0], dtype = bool)
    mask_pixel[pixels] = True
    mask_pixel = mask_pixel.reshape(pipelines.matrix.shape[0:2])

    mask_noeud = np.zeros_like(RZ_mask_grid, dtype = bool)
    rows_noeud, indphi, cols_noeud = np.unravel_index(noeuds, mask_noeud.shape)
    mask_noeud[rows_noeud,indphi, cols_noeud] = True
    # true_nodes = np.flatnonzero(RZ_mask_grid)
    # mask_noeud = np.zeros_like(RZ_mask_grid, dtype = bool)
    # mask_noeud.ravel()[true_nodes[noeuds]] = True
    print('shape voxel_map ', plasma2.voxel_map.shape)
    print('shape mask_noeud ', mask_noeud.shape)
    
    transfert_matrix = flattened_matr[pixels,:][:, noeuds]

    nb_visible_noeuds = len(np.unique(noeuds))
    nb_vision_pixel = len(np.unique(pixels))
    print('visible node = ' + str(nb_visible_noeuds) + 'out of ' + str(nb_noeuds_r*nb_noeuds_z))
    print('vision pixels = ' + str(nb_vision_pixel) + 'out of ' + str(pipelines.matrix.shape[0] * pipelines.matrix.shape[1]))

    transfert_matrix = csr_matrix(transfert_matrix)
    print(transfert_matrix.shape)
    pixels = np.squeeze(pixels)
    noeuds = np.squeeze(noeuds)

    name_folder = 'mat_saves/'
    name_material = ParamsMachine.name_material.split('/')[-1]
    if os.path.splitext(ParamsMachine.path_CAD)[1]:
        name_CAD = os.path.splitext(ParamsMachine.path_CAD)[1]
    else:
        name_CAD = ParamsMachine.path_CAD.split('/')[-2]
    if flag_wall_limit:
        name_CAD = '2D_wall_mesh'
        name_material = name_wall_material
    #plot and save los
    utility_functions.plot_image(np.squeeze(mask_noeud).T, extent = extent_RZ, origin = 'lower')
    plt.savefig(name_folder + name_CAD+name_material+'dr'+str(ParamsGrid.dr_grid)+'step'+str(step)+'nlos' + str(n_los)+'rays'+str(pixel_samples)+'view los.png')
    plt.close()
    dict_spectro = dict(R1 = R1, Z1=Z1, grid_mask = grid_mask, extent_RZ = extent_RZ, R_wall = R_wall,  Z_wall= Z_wall, transfert_matrix = transfert_matrix, mask_pixel= mask_pixel, mask_noeud= mask_noeud,  pipelines= pipelines, cell_r = cell_r, cell_z= cell_z)
    savemat(name_folder + name_CAD+name_material+'dr'+str(ParamsGrid.dr_grid)+'step'+str(step)+ str(n_los)+'rays'+str(pixel_samples)+'cherab_los.mat', dict_spectro)
    return R1, Z1, grid_mask, extent_RZ, R_wall, Z_wall, transfert_matrix, mask_pixel, mask_noeud, pipelines, cell_r, cell_z

