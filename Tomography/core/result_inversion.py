from . import utility_functions
from . import inversion_module
import importlib
from scipy.sparse import csr_matrix, save_npz, csc_matrix, load_npz, isspmatrix
from scipy.io import loadmat
from scipy.sparse import csr_matrix, issparse
import numpy as np
import h5py
import os
import numpy.typing as npt
import pdb
from dataclasses import dataclass, asdict, field, is_dataclass, fields
import hashlib
import matplotlib.pyplot as plt


def get_name(string):
    if string:
        return os.path.splitext(os.path.basename(string))[0]
    else:
        return ''

@dataclass
class Params:
    root_folder : str = None
    

    @property
    def filename(self):
        return self.to_filename()

    def format_value(self, key: str, value, full:bool = False) -> str:
        """Format each field depending on its type."""
        if isinstance(value, float):
            return f"{key}-{value:.3g}"        # 3 sig. digits
        elif isinstance(value, int):
            return f"{key}-{value}"
        elif isinstance(value, str):
            if full:
                return f"{key}-{value}" if value else ""
            else:
                return f"{key}-{os.path.splitext(os.path.basename(value))[0]}" if value else ""
        elif isinstance(value, dict):
            if not value:
                return ""    # skip if empty dict
            name_dict_parameters = "_".join(f"{k}_{v}" for k, v in value.items())
            return f"{key}-{name_dict_parameters}"   
        elif isinstance(value, np.ndarray):
            return f"{key}-{value[0]}-{value[-1]}"
        elif isinstance(value, list):
            return f"{key}-{value[0]}-{value[-1]}"
    
        elif value is None:
            return ""                          # skip if None
        else:
            return f"{key}-{str(value)}"

    def to_filename(self, prefix="", ext="", sep="_", exclude=("root_folder", "class_name"), full = True):
        parts = []
        for k, v in asdict(self).items():
            if k not in exclude:
                formatted = self.format_value(k, v, full)
                if formatted:
                    parts.append(formatted)

        return sep.join([prefix] + parts) + f"{ext}"
    

@dataclass
class ParamsVid(Params):
    
    inversion_method : str = None
    nshot : int = None
    path_vid : str = None
    dict_denoising : dict = field(default_factory=dict)
    time_input : np.array = None
    frame_input : np.array = None
    inversion_parameter : dict = field(default_factory=dict)
    class_name : str = 'ParamsVid'
    def to_filename(self, prefix="", ext="", sep="_", exclude=None):
        if exclude is None:
            exclude = ("root_folder", "class_name")  
            # 👆 add subclass-specific exclusions
        return super().to_filename(prefix, ext, sep, exclude)
@dataclass
class ParamsGrid(Params):
    
    dr_grid : float = None
    dz_grid : float = None
    symetry : str = 'toroidal'
    variant_mag : str = None
    revision : int = None
    phi_grid : float = None
    grid_precision_multiplier : float = None
    n_polar : int = None
    crop_center : bool = None
    class_name : str = 'ParamsGrid'
    def __post_init__(self):
        if not self.symetry:
            pass
        elif self.symetry.lower() == 'toroidal':
            self.symetry = 'toroidal'
        elif self.symetry.lower() == 'magnetic':
            self.symetry = 'magnetic'
        else:
            raise(ValueError(f"{self.symetry} is not a supported emissivity hypothesis"))
        
    def to_filename(self, prefix="", ext="", sep="_", exclude=None):
        if exclude is None:
            exclude = ("root_folder", "class_name")  
            # 👆 add subclass-specific exclusions
        return super().to_filename(prefix, ext, sep, exclude)


@dataclass
class ParamsMachine(Params):
    
    machine : str = None
    path_calibration : str= None
    path_wall : str = None
    path_CAD : str = None
    variant_CAD : str = None
    path_mask : str = None
    name_material : str = 'absorbing_surface'
    param_fit : str = None,
    decimation : int = 1

    class_name : str = 'ParamsMachine'

    def __post_init__(self):
        if not self.machine:
            pass
        elif self.machine.lower() == 'compass':
            self.machine = 'COMPASS'
        elif self.machine.lower() == 'west':
            self.machine = 'WEST'
        else:
            raise(ValueError('Unrecognized machine, type either compass or west'))
    def to_filename(self, prefix="", ext="", sep="_", exclude=None, full = False):
        if exclude is None:
            exclude = ("root_folder", "class_name", "machine", "path_calibration", "path_CAD" ,"variant_CAD")  
            # 👆 add subclass-specific exclusions

        return f"machine_{self.machine}/path_calibration_{get_name(self.path_calibration)}/path_CAD_{get_name(self.path_CAD)}_variant_CAD{self.variant_CAD}/" + super().to_filename(prefix, ext, sep, exclude, full)


class TomographyResults:
    required_keys = []
    def __init__(self, data: dict = None, **kwargs):
        """
        Initialize from dictionary or keyword arguments.
        """
        combined_data = {}
        if data:
            combined_data.update(data)
        combined_data.update(kwargs)
         
        # Set attributes
        for key, value in combined_data.items():
            setattr(self, key, value)

        if (data is not None) and (kwargs is not None):
            missing_keys = [key for key in self.required_keys if key not in combined_data]
            if missing_keys:
                raise ValueError(
                    f"Missing required keys for {self.__class__.__name__}: {missing_keys}"
                )
        try:
            if not os.path.exists(self.root_folder):
                self.root_folder = os.getcwd()
        except:
            self.root_folder = os.getcwd()

    

    def update(self, data: dict = None, **kwargs):
        """
        Update attributes of the instance.
        Can pass a dictionary or keyword arguments.
        """
        combined_data = {}
        if data:
            combined_data.update(data)
        combined_data.update(kwargs)

        for key, value in combined_data.items():
            setattr(self, key, value)

    def to_dict(self) -> dict:
        """Convert object back to dictionary."""
        return self.__dict__


    def save(self):
        self.save_h5(self.filename)


    def load(self):
        
        return self.load_h5(self.filename)

    def save_h5(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        filename = filename if filename.endswith(".h5") else filename + ".h5"
        with h5py.File(filename, "w") as f:
            for key, value in self.__dict__.items():

                # 🔹 Case 1: dataclass instance
                if is_dataclass(value):
                    grp = f.create_group(key)
                    grp.attrs["_class"] = value.__class__.__name__
                    grp.attrs["_module"] = value.__class__.__module__

                    for field in fields(value):
                        v = getattr(value, field.name)
                        self._save_value(grp, field.name, v)

                else:
                    # 🔹 Save generic value
                    self._save_value(f, key, value)

        print(f"Saved to {filename}")

    def _save_value(self, parent, key, value):
        """Helper: save different value types into HDF5."""
        if isinstance(value, np.ndarray):
            parent.create_dataset(key, data=value)
        elif isinstance(value, list):
            parent.create_dataset(key, data=np.array(value))
        elif issparse(value):
            grp = parent.create_group(key)
            grp.attrs["_sparse"] = "csr_matrix"
            grp.create_dataset("data", data=value.data)
            grp.create_dataset("indices", data=value.indices)
            grp.create_dataset("indptr", data=value.indptr)
            grp.attrs["shape"] = value.shape

        elif isinstance(value, dict):  # 🔹 handle dictionaries
            grp = parent.create_group(key)
            grp.attrs["_type"] = "dict"
            for subkey, subval in value.items():
                self._save_value(grp, subkey, subval)

        elif isinstance(value, (int, float, str)):
            parent.attrs[key] = value

        elif value is None:
            return

        else:
            parent.attrs[key] = str(value)

    @classmethod
    def load_h5(cls, filename):
        filename = filename if filename.endswith(".h5") else filename + ".h5"
        if not os.path.isfile(filename):
            raise FileNotFoundError(f"{filename} does not exist.")

        def _load_value(item):
            """Helper: load datasets, dicts, dataclasses, sparse, etc."""
            if isinstance(item, h5py.Dataset):
                return np.array(item)

            elif isinstance(item, h5py.Group):
                # Sparse matrix
                if "_sparse" in item.attrs and item.attrs["_sparse"] == "csr_matrix":
                    data = np.array(item["data"])
                    indices = np.array(item["indices"])
                    indptr = np.array(item["indptr"])
                    shape = tuple(item.attrs["shape"])
                    return csr_matrix((data, indices, indptr), shape=shape)

                # Dictionary
                if "_type" in item.attrs and item.attrs["_type"] == "dict":
                    d = {}
                    for subkey, subitem in item.items():
                        d[subkey] = _load_value(subitem)
                    for subkey, val in item.attrs.items():
                        if subkey != "_type":
                            d[subkey] = val
                    return d

                # Dataclass
                if "_class" in item.attrs and "_module" in item.attrs:
                    class_name = item.attrs["_class"]
                    module_name = item.attrs["_module"]

                    try:
                        module = importlib.import_module(module_name)
                        cls_type = getattr(module, class_name)
                    except Exception as e:
                        print(f"⚠️ Could not import {module_name}.{class_name}, using dict: {e}")
                        cls_type = None

                    values = {}
                    for subkey, subitem in item.items():
                        values[subkey] = _load_value(subitem)
                    for subkey, val in item.attrs.items():
                        if subkey not in ["_class", "_module"]:
                            values[subkey] = val

                    return cls_type(**values) if cls_type else values

                # Fallback group → dict
                d = {}
                for subkey, subitem in item.items():
                    d[subkey] = _load_value(subitem)
                for subkey, val in item.attrs.items():
                    d[subkey] = val
                return d

            else:
                return item

        data_dict = {}
        with h5py.File(filename, "r") as f:
            for key, item in f.items():
                data_dict[key] = _load_value(item)
            for key, value in f.attrs.items():
                data_dict[key] = value
        return cls(data_dict)


    @classmethod
    def from_file(cls, filename: str = None):
        """
        Alternative constructor:
        - If filename is given, load directly.
        - If no filename is given, open a file chooser
          (GUI if available, otherwise terminal).
        """
        if filename is None:
            try:
                filename = utility_functions.get_file(full_path=1)
                if not filename:
                    print("No file selected.")
                    return None
            except Exception:
                # Terminal fallback with folder selection
                print("GUI not available. Terminal mode.")
                folder = input("Enter folder path (or leave empty for current folder): ").strip()
                if not folder:
                    folder = "."
                if not os.path.isdir(folder):
                    print(f"Folder '{folder}' does not exist. Using current directory.")
                    folder = "."

                # List all .npz files in the folder
                files = [f for f in os.listdir(folder) if f.endswith((".npz", ".mat"))]
                if not files:
                    print(f"No .npz or .mat files found in folder '{folder}'.")
                    return None

                # Let user pick a file
                print("Select a file:")
                for i, f in enumerate(files):
                    print(f"[{i}] {f}")
                idx = int(input("Enter number: "))
                filename = os.path.join(folder, files[idx])

        
        return cls.load_h5(filename)

def __str__(self):
    lines = [f"{self.__class__.__name__} object with fields:"]
    for key, value in self.__dict__.items():
        if isinstance(value, np.ndarray):
            desc = f"ndarray, shape={value.shape}, dtype={value.dtype}"
        else:
            desc = repr(value)
        lines.append(f"  {key}: {desc}")
    return "\n".join(lines)

class Inversion_results(TomographyResults):
    # required_keys = ["Params_vid", "t_start", "inversion_method", "time_input", "frame_input", "vid", "inversion_results_full"]
    required_keys = []
    @property
    def filename(self):
        return (self.root_folder + '/' + self.ParamsMachine.filename + '/' + self.ParamsGrid.filename + '/' + self.ParamsVid.filename)

    def __init__(self, data: dict = None,ParamsMachine = ParamsMachine(), ParamsGrid = ParamsGrid(), ParamsVid= ParamsVid()):
        self.root_folder = None
        self.ParamsMachine = ParamsMachine
        self.ParamsGrid = ParamsGrid
        self.ParamsVid = ParamsVid
        self._Transfert_Matrix = None
        self._prep_inversion = False
        super().__init__(data)
        # if "ParamsMachine" in data:
        #     self.ParamsMachine = data["ParamsMachine"]
        # else:
        #     self.ParamsMachine = ParamsMachine
        # if "ParamsGrid" in data:
        #     self.ParamsGrid = data["ParamsGrid"]
        # else:
        #     self.ParamsGrid = ParamsGrid
        # if "ParamsVid" in data:
        #     self.ParamsVid = data["ParamsVid"]
        # else:
        #     self.ParamsVid = ParamsVid

    def load_transfert_matrix(self):

        transfert_matrix = Transfert_Matrix(ParamsMachine = self.ParamsMachine, ParamsGrid = self.ParamsGrid)
        try:
            self._Transfert_Matrix = transfert_matrix.load()
        except:
            print(f"No transfert matrix found at {self.root_folder}/{self.ParamsMachine.filename}/{self.ParamsGrid.filename}")
            # raise(ValueError(f"No transfert matrix found at {self.root_folder} / {self.ParamsMachine.filename}/ {self.ParamsGrid.filename}"))
        return self._Transfert_Matrix
    @property
    def Transfert_Matrix(self):
        if self._Transfert_Matrix is None:
            self._Transfert_Matrix = self.load_transfert_matrix()
            print('successfully loaded Transfert Matrix')
        return self._Transfert_Matrix

    def prep_inversion(self):
        if self.ParamsVid.inversion_parameter:
            self.transfert_matrix, self.pixels, self.noeuds, self.mask_pixel, self.mask_noeud = inversion_module.prep_inversion(self.Transfert_Matrix.transfert_matrix, self.Transfert_Matrix.mask_pixel, self.Transfert_Matrix.mask_noeud, self.Transfert_Matrix.pixels, self.Transfert_Matrix.noeuds, self.ParamsVid.inversion_parameter, self.Transfert_Matrix.R_noeud, self.Transfert_Matrix.Z_noeud)
            print(f"successfully prepared transfert matrix, appling inversion parameter {self.ParamsVid.inversion_parameter}")

        self.inversion_parameter = self.ParamsVid.inversion_parameter
        self._prep_inversion = True


    def reload_transfert_matrix(self):

        self._Transfert_Matrix = self.load_transfert_matrix()

    @property
    def inversion_results_full(self):
        if self._prep_inversion:
            if self.inversion_parameter != self.ParamsVid.inversion_parameter:
                self.prep_inversion()
        else:
            self.prep_inversion()
        
        
        return utility_functions.reconstruct_2D_image_all_slices(self.inversion_results, self.mask_noeud)
        

    @property
    def images_retrofit_full(self):
        if self._prep_inversion:
            if self.inversion_parameter != self.ParamsVid.inversion_parameter:
                self.prep_inversion()
        else:
            self.prep_inversion()
        
        return utility_functions.reconstruct_2D_image_all_slices(self.images_retrofit, self.mask_pixel)
    @property
    def inversion_results_full_thresolded(self):
        if self._prep_inversion:
            if self.inversion_parameter != self.ParamsVid.inversion_parameter:
                self.prep_inversion()
                self.denoising()
        else:
            self.prep_inversion()
            self.denoising()
       
        return  utility_functions.reconstruct_2D_image_all_slices(self.inversion_results_thresholded, self.mask_noeud)


    def inversion_results_full_normalized_thresolded(self):
        if self._prep_inversion:
            if self.inversion_parameter != self.ParamsVid.inversion_parameter:
                self.prep_inversion()
                self.denoising()
        else:
            self.prep_inversion()
            self.denoising()
       
        return  utility_functions.reconstruct_2D_image_all_slices(self.inversion_results_thresholded*self.norms().T, self.mask_noeud)


    def inversion_results_full_normalized(self):
        if self._prep_inversion:
            if self.inversion_parameter != self.ParamsVid.inversion_parameter:
                self.prep_inversion()
                self.denoising()
        else:
            self.prep_inversion()
            self.denoising()
       
        return  utility_functions.reconstruct_2D_image_all_slices(self.inversion_results*self.norms().T, self.mask_noeud)


    def denoising(self):
        if self._prep_inversion:
            if self.inversion_parameter != self.ParamsVid.inversion_parameter:
                self.prep_inversion()
        else:
            self.prep_inversion()
        if (self.ParamsVid.inversion_method == 'SparseBob') or (self.ParamsVid.inversion_method == 'Bob'):
            self.inversion_results_thresholded = inversion_module.denoising(self)
        else:
            raise(ValueError('This method has no denoising method implemented'))
    
    @property
    def path_inverse_matrix(self):
        return self.Transfert_Matrix.filename + '/' + self.ParamsVid.filename +  'inverse_matrix'

    def norms(self):
        from tomotok.core.inversions import Bob, SparseBob, CholmodMfr, Mfr
        if (self.ParamsVid.inversion_method == 'SparseBob') or (self.ParamsVid.inversion_method == 'Bob'):
            
            if (self.ParamsVid.inversion_method == 'SparseBob'):
                inversion = SparseBob()
                inversion.load_decomposition(self.path_inverse_matrix)
            elif (self.ParamsVid.inversion_method == 'Bob'):
                inversion = Bob()
                inversion.load_decomposition(self.path_inverse_matrix)

            return inversion.norms
        else:
            raise(ValueError('This method has no norms method implemented'))
    def plot_results(self, frame = 0):
        if (self.ParamsVid.inversion_method == 'SparseBob') or (self.ParamsVid.inversion_method == 'Bob'):
            self.plot_bob(frame)
        else:
            self.plot_simple(frame)

    def plot_bob(self, frame, vmin = None, vmax = None):
        figure_results =plt.figure()
        vmax = vmax or np.max(self.inversion_results_full)
        vmin = vmin or np.min(self.inversion_results_full)
        extent = [self.Transfert_Matrix.R_noeud[0], self.Transfert_Matrix.R_noeud[-1], self.Transfert_Matrix.Z_noeud[0], self.Transfert_Matrix.Z_noeud[-1]]
   
        #synthetic image
        plt.subplot(2,2,1)
        plt.imshow(self.inversion_results_full[frame, :, :].T, extent = extent, origin='lower', vmin = vmin, vmax = vmax)
        plt.colorbar()
        plt.title('inversion results')

        #retro fit
        plt.subplot(2,2,2)
        plt.imshow(self.inversion_results_full_thresolded[frame, :, :].T, extent = extent, origin='lower')
        plt.colorbar()
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.title('inversion results thresholded')

        #inversion
        plt.subplot(2,2,3)
        plt.imshow(self.inversion_results_full_normalized()[frame, :, :].T, extent = extent, origin='lower')
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.colorbar()
        plt.title('inversion results normalized')

        plt.subplot(2,2,4)
        plt.imshow(self.inversion_results_full_normalized_thresolded()[frame, :, :].T, extent = extent, origin='lower')
        plt.xlabel('R [m]')
        plt.ylabel('Z [m]')
        plt.title('inversion results normalized and thresholded') 
        plt.show(block = False)

    def plot_simple(self, frame, vmin = None, vmax = None):
        figure_results =plt.figure()
        #synthetic image
        vmax = vmax or np.max(self.inversion_results_full)
        vmin = vmin or np.min(self.inversion_results_full)
        extent = [self.Transfert_Matrix.R_noeud[0], self.Transfert_Matrix.R_noeud[-1], self.Transfert_Matrix.Z_noeud[0], self.Transfert_Matrix.Z_noeud[-1]]
        plt.subplot(1, 2,1)
        plt.imshow(self.inversion_results_full[frame, :, :].T, origin='lower', extent = extent, vmin = vmin, vmax = vmax)
        plt.colorbar()
        plt.title('inversion results normalized and thresolded')

        #retro fit
        plt.subplot(1,2,2)
        plt.imshow(self.images_retrofit_full[frame, :, :].T)
        plt.colorbar()
        plt.title('Retro fit')
        plt.show(block = False)
    def create_video(self, filename = None):
        filename = filename or self.filename + 'vid.mp4'
        array = np.swapaxes(self.inversion_results_full, 1,2)
   
        utility_functions.save_array_as_video(array, filename, fps=20)

class Transfert_Matrix(TomographyResults):
    required_keys = []
                    #  , "mask_noeud", "mask_pixel", "transfert_matrix", "noeuds", "pixels", "RZ_wall", "R_noeud", "Z_noeud"]

    def __init__(self, data: dict = None, ParamsMachine = ParamsMachine(), ParamsGrid = ParamsGrid()):
        self.root_folder = None
        self.ParamsMachine = ParamsMachine
        self.ParamsGrid = ParamsGrid
        super().__init__(data)
        # if "ParamsMachine" in data:
        #     self.ParamsMachine = data["ParamsMachine"]
        # else:
        #     self.ParamsMachine = ParamsMachine
        # if "ParamsGrid" in data:
        #     self.ParamsGrid = data["ParamsGrid"]
        # else:
        #     self.ParamsGrid = ParamsGrid
    @property
    def filename(self):
        return (self.root_folder + '/' + self.ParamsMachine.filename + '/' + self.ParamsGrid.filename)

