import calcam
from scipy.sparse import load_npz, save_npz
import numpy as np


def load_calibration(name_calibration):
    path_calibration = name_calibration + '.ccc' 
    calcam_camera = calcam.Calibration(path_calibration)
    realcam = calcam_camera.get_raysect_camera()
    
    np.savez_compressed(name_calibration, 
                        pixel_origins = realcam.pixel_origins, 
                        mask = calcam_camera.subview_mask,
                        pixel_directions = realcam.pixel_directions)


def get_name(path):
    path = path.split('/')
    name = path[len(path)-1]
    name_shortened = name.split('.')

    name_shortened = name_shortened[:-1]


    return name_shortened
