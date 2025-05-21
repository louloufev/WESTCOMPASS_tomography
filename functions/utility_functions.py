import importlib
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os


def plot_wall(image_raw = None, xlabel = None, ylabel = None, percentile_inf = None, percentile_sup = None, extent = None, origin = 'upper', vmax = None, title = '', cmap = 'viridis'):
    path_wall = '/Home/LF276573/Zone_Travail/Python/CHERAB/models_and_calibration/models/west/WEST_wall.npy'
    RZwall = np.load(path_wall)
    R_wall = RZwall[:, 0]
    Z_wall = RZwall[:, 1]
    extent = [np.min(R_wall), np.max(R_wall), np.min(Z_wall), np.max(Z_wall)]
    fig, ax = plt.subplots(1,1)
    if image_raw:
        image = np.copy(image_raw).astype(float)
        plt.imshow(image, extent=extent, origin = origin, vmax=vmax, cmap = cmap)
        plt.title(title)
        plt.colorbar()
    plt.plot(R_wall, Z_wall, 'r')
    fig.tight_layout()
    plt.show(block = False)

def plot_masked_image(image, mask, alpha = 0.5, percentile_inf = None, percentile_sup = None):
    # Superpose 2 images, giving some degree of transparency to mask (0<alpha<1)
    fig, ax = plt.subplots(1,1)

    percentile_inf = percentile_inf or 0
    percentile_sup = percentile_sup or 100
    Q1 = np.percentile(image, percentile_inf)
    Q3 = np.percentile(image, percentile_sup)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    image_clean = np.copy(image)
    # image_clean[np.invert((image>lower_bound) & (image<upper_bound))] = np.NaN
    
    ax.imshow(image_clean)
    ax.imshow(mask, alpha=alpha)
    plt.show(block = False)
    return fig


def plot_image(image_raw, xlabel = None, ylabel = None, percentile_inf = None, percentile_sup = None, extent = None, origin = 'upper', vmax = None, title = '', cmap = 'viridis'):
    # Superpose 2 images, giving some degree of transparency to mask (0<alpha<1)
    fig, ax = plt.subplots(1,1)
    
    image = np.copy(image_raw).astype(float)
    if percentile_inf or percentile_sup:
        percentile_inf = percentile_inf or 0
        percentile_sup = percentile_sup or 100
        Q1 = np.percentile(image, percentile_inf)
        Q3 = np.percentile(image, percentile_sup)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        image[np.invert((image>lower_bound) & (image<upper_bound))] = np.NaN
        print(lower_bound)
    if not vmax:
        vmax = np.max(image)
    plt.imshow(image, extent=extent, origin = origin, vmax=vmax, cmap = cmap)
    plt.title(title)
    plt.colorbar()
    plt.show(block = False)
    return fig, ax


import tkinter as tk
from tkinter import filedialog

def get_file(title = "Select file", path_root = None, full_path = 0):
    root = tk.Tk()
    root.withdraw()
    file_path =  filedialog.askopenfilename(title=title, initialdir=path_root, filetypes=[("All files", "*.*"), ("Images", "*.png"), ("CSV files", "*.csv"), ("python files", "*.npy, *.npz")])
    if not full_path:
        file_path = os.path.relpath(file_path, path_root)

    
    return file_path

def compute_sparsity(matrix):
    total_elements = matrix.shape[0] * matrix.shape[1]
    sparsity = 1 - (matrix.nnz / total_elements)
    return sparsity


def plot_comparaison_image(image1, image2, percentile_inf = None, percentile_sup = None, origin = 'upper', extent = None, vmax = None):
    # Superpose 2 images, giving some degree of transparency to mask (0<alpha<1)
    fig, ax = plt.subplots(1,3)

    percentile_inf = percentile_inf or 0
    percentile_sup = percentile_sup or 100
    Q1 = np.percentile(image1, percentile_inf)
    Q3 = np.percentile(image1, percentile_sup)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    image1_clean = np.copy(image1)
    # image1_clean[np.invert((image1>lower_bound) & (image1<upper_bound))] = np.NaN
    
    Q1 = np.percentile(image2, percentile_inf)
    Q3 = np.percentile(image2, percentile_sup)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    image2_clean = np.copy(image2)
    # image2_clean[np.invert((image2>lower_bound) & (image2<upper_bound))] = np.NaN
    
    if not vmax:
        vmax = np.max(image1_clean)
    plt.subplot(1,3,1)
    plt.imshow(image1_clean, origin = origin, extent = extent, vmax = vmax)
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(image2_clean, origin = origin, extent = extent, vmax = vmax)
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.imshow(image1_clean-image2_clean, origin = origin, extent = extent, interpolation='nearest')
    plt.colorbar()

    plt.show(block = False)
    return fig, ax

def xyztorphiz(Vec):
    if Vec.ndim>1:
        RPHIZ = np.full(Vec.shape[0], 3)
      
        for i in range(Vec.shape[0]):
            RPHIZ[i, 0] = np.sqrt(Vec[i, 0]**2+Vec[i, 1]**2)
            RPHIZ[i, 1] =np.arctan2(Vec[i, 1], Vec[i, 0])
            RPHIZ[i, 2] = Vec[i, 2]

       
    else:
        
        r = np.sqrt(Vec[0]**2+Vec[1]**2)
        phi = np.arctan2(Vec[1], Vec[0] )
        z = Vec[2]
        RPHIZ = [r, phi, z]
    return RPHIZ


def smooth_line_image(image = None, window_size = 3):
    if not image:
        image_path = get_file(full_path=1)
        import imageio.v3 as iio
        image = iio.imread(image_path)
        horizontal_data = np.mean(image, 1)
        smoothed_data = np.convolve(horizontal_data, np.ones(window_size)/window_size, mode='same')
        # smoothed_data = 
        # pdb.set_trace()
        smoothed_image = smoothed_data*image.T/np.mean(smoothed_data)
        smoothed_image = smoothed_image.T
    return smoothed_image, image


def convert_npz_to_mat(path_npz):
    from scipy.io import savemat
    out = np.load(path_npz, allow_pickle=True)
    dictionnary = dict()
    for key in out.keys():
        dictionnary.update({key : out[key]})
    file_path, ext = os.path.splitext(path_npz)
    

    savemat(file_path + '.mat', dictionnary)

def convert_npz_file_to_mat(path_npz = None):
    from scipy.io import savemat
    import numpy as np
    import glob
    import os
    if not path_npz:
            path_npz = get_file(title = "Select file", path_root = None, full_path = 1)
    try:
        d = np.load(path_npz, allow_pickle=True)
        path_npz, ext = os.path.splitext(path_npz)
        fm = path_npz+'.mat'
        savemat(fm, d)
    except:
        print('fail to read npz file', path_npz)
        

    # npzFiles = glob.glob("*.npz")
    # for f in npzFiles:
    #     fm = os.path.splitext(f)[0]+'.mat'
    #     d = np.load(f)
    #     savemat(fm, d)
    #     print('generated ', fm, 'from', f)


    import numpy as np
import matplotlib.pyplot as plt

def compare_images(A, B, title='', subtitles=('A', 'B', '|A - B|'), clim=None, axis='equal'):
    """
    Display two 2D arrays and their absolute difference side by side.
    
    Parameters:
        A, B         : 2D numpy arrays to compare.
        title        : Main title for the figure.
        subtitles    : Tuple of 3 subtitles for A, B, and |A - B|.
        clim         : Tuple (vmin, vmax) for color scale. If None, uses automatic scaling.
        axis         : Axis style, e.g., 'equal', 'auto', 'off'.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    images = [A, B, np.abs(A - B)]
    
    for i, ax in enumerate(axes):
        im = ax.imshow(images[i], cmap='viridis', vmin=clim[0] if clim else None, vmax=clim[1] if clim else None)
        ax.set_title(subtitles[i])
        ax.axis(axis)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title:
        fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Leave space for suptitle
    plt.show()



def average_along_first_row(matrix, block_size = 10):
    T = matrix.shape[0]

    T_trunc = T - (T % block_size)

    matrix_trunc = matrix[:T_trunc]  # Truncate to a multiple of 10
    matrix_avg = matrix_trunc.reshape(-1, block_size, matrix.shape[1], matrix.shape[2]).mean(axis=1)
    # Result shape: (T_trunc//10, M, N)
    return matrix_avg


def add_variable_to_npz(file_path=None, new_var_name=None, new_var_value=None, output_path=None, path_root = None):
    """
    Load a .npz file, add or replace a variable, and save it.
    
    Parameters:
        file_path (str): Path to the existing .npz file.
        new_var_name (str): Name of the variable to add or replace.
        new_var_value (any): Numpy-compatible value to add.
        output_path (str, optional): If None, overwrite the original file.
        path_root (str, optional): QOL option, path to the folder where to look for the file
    """
    if file_path is None:
        file_path =  filedialog.askopenfilename(title='choose file', initialdir=path_root, filetypes=[("All files", "*.*"), ("Images", "*.png"), ("CSV files", "*.csv"), ("python files", "*.npy, *.npz")])
       
    # Load existing data
    data = dict(np.load(file_path, allow_pickle=True))
    
    if new_var_name is None:
        new_var_name = input('give name of variable to add/modify')
    if new_var_value is None:
        new_var_value = input('give variable')
    # Add or update the new variable
    data[new_var_name] = new_var_value
    
    # Save to the same file or to a new one
    if output_path is None:
        output_path = file_path
    
    np.savez(output_path, **data)
    print(f"Saved updated .npz to {output_path}")



def apply_to_files(folder_path, extension, function_to_apply):
    """
    Apply a function to all files with a given extension in a folder and its subfolders.

    Parameters:
        folder_path (str): Root directory to search.
        extension (str): File extension to look for (e.g. '.npz', '.txt').
        function_to_apply (callable): Function that takes the full path of a file as input.

    Returns:
        results (list): List of outputs from function_to_apply for each file.
    """
    results = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(extension):
                full_path = os.path.join(root, file)
                try:
                    result = function_to_apply(full_path)
                    results.append(result)
                except Exception as e:
                    print(f"⚠️ Failed to process {full_path}: {e}")
                    continue
    return results