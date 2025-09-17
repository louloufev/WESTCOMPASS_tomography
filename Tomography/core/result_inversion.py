import fonction_tomo
import utility_functions
import inversion_module
from scipy.sparse import csr_matrix, save_npz, csc_matrix, load_npz, isspmatrix
from scipy.io import loadmat
import numpy as np


class raytrace_result:

    def __init__(self, path_transfert_matrix, path_parameters, path_inverse = None):
        self.path_transfert_matrix = path_transfert_matrix
        self.path_parameters = path_parameters
        self.path_inverse = path_inverse
        self.transfert_matrix = load_npz(path_transfert_matrix)
        parameters = loadmat(path_parameters)
        self.inversion_method = parameters['inversion_method']
        self.inversion_parameter = parameters['inversion_parameter']
        self.name_material = parameters['name_material']
        self.symetry = parameters['symetry']
        self.machine = parameters['machine']
        self.path_CAD = parameters['path_CAD']
        self.path_calibration = parameters['path_calibration']
        self.variant = parameters['variant']
        self.n_polar = parameters['n_polar']
        self.dr_grid = parameters['dr_grid']
        self.dz_grid = parameters['dz_grid']
        self.path_mask = parameters['path_mask']
        self.decimation = parameters['decimation']
        self.param_fit = parameters['param_fit']
        self.test = parameters['test']
        self.test = parameters['test']
        self.test = parameters['test']
        self.test = parameters['test']





class result_inversion:

    def __init__(self, path_save, raytrace_result):
        self.path_save = path_save
        self.raytrace_result = raytrace_result

        resultat = np.load(path_save, allow_pickle = True)
        self.inversion_parameter = resultat['inversion_parameter']
        self.inversion_parameter = resultat['inversion_parameter']
        self.inversion_parameter = resultat['inversion_parameter']
        self.inversion_parameter = resultat['inversion_parameter']
        self.inversion_parameter = resultat['inversion_parameter']
        



        self.raytrace_result.update(self.inversion_parameter)

