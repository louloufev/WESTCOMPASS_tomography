'''
Structure of main :
inits : how to setup environment
machine_COMPASS : holds results

Structure of Tomography:
core : 
    fonction_tomo_test : hosts main function for raytracing
    result_inversion : hosts classes for saving and manipulating results
    inversion_module : hosts function for different inversion methods
    utility_functions : hosts qol functions

ressources : 
    hosts components of the 3D model that need to be kept (as well as its material type)





Fonction full_inversion_toroidal in fonction_tomo_test :
main function for calculation of geometry matrix (transfert matrix) and inversion
As a rule, the videos (real, reconstructed) are saved rotated, but the rotated result is in the conventional format for videos (vertical axis from top to bottom)
plt.imshow(vid.T)
Inverted results are also rotated, but the rotation has its z axis in growing order. 
plt.imshow(inversion.T, origin = 'lower')

The result class handles these rotations when plotting results or creating video files


Inputs :
ParamsMachine : main parameters of the tokamak (wall, 3d model, material type, camera calibrations)
ParamsGrid : parameters for the 2D grid of the inversion (scale, symmetry hypothesis, etc...)
ParamsVid : parameters for the treatment of the video (video length, filters)

Outputs :
Inversion_results : Special class handling the results. 
Save automatically the Geometry matrix, the inversion matrix and the results of the inversion.


The function looks if the results have been already calculated for these parameters.
If not, loads the video and apply filtering : utility_functions.get_vid

Try to look if the Geometry matrix has been already calculated for these parameters
If not, loads the camera, the 3D model (read_CAD_from_components) and calculate the Geometry matrix (get_transfert_matrix)

Finally, runs the inversion (inverse_vid_from_class)


The result class : Inversion_results
3 functions for recalculating data with new parameters. Any of these functions modify the result for the function after it, make sure to call all the following functions if modifying a parameter.
Inversion_results.redo_video() to retreat camera video (ex : modifying gaussian fiter : Inversion_results.ParamsVid.dict_vid['sigma'] = 4 )
Inversion_results.redo_inversion_results() #recalculate inversion (ex : modifying minimum value in geometry matrix : Inversion_results.ParamsVid.inversion_parameter['min_visibility_node'] = 0)
Inversion_results.denoising()  # redo inversion thresholding for noise reduction (ex : modifying denoising level : Inversion_results.ParamsVid.c = 4)


exemple files :
exemple_compass : how to input parameters and launch an inversion
vid_processing : how to treat results once calculated
'''