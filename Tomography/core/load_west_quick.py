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
importlib.reload(result_inversion)
importlib.reload(utility_functions)



Inversion_results = result_inversion.Inversion_results()
Inversion_results = Inversion_results.from_file()
###### path parameters to look for calibrations, 3D models, etc..

# mask for the camera
# path_mask =  '/Home/LF285735/Zone_Travail/Python/CHERAB/masks/west/60990/custom_frame_421_1.npy'
