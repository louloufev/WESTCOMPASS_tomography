import importlib
import numpy as np
import matplotlib.pyplot as plt
import pdb
import os
import imageio
from .inversion_module import reconstruct_2D_image
from typing import Optional, Tuple

def plot_wall(image_raw = None, xlabel = None, ylabel = None, percentile_inf = None, percentile_sup = None, extent = None, origin = 'upper', vmax = None, vmin = None, title = '', cmap = 'viridis'):
    path_wall = '/Home/LF276573/Zone_Travail/Python/CHERAB/models_and_calibration/models/west/WEST_wall.npy'
    RZwall = np.load(path_wall)
    R_wall = RZwall[:, 0]
    Z_wall = RZwall[:, 1]
    extent = [np.min(R_wall), np.max(R_wall), np.min(Z_wall), np.max(Z_wall)]
    fig, ax = plt.subplots(1,1)
    if image_raw:
        image = np.copy(image_raw).astype(float)
        plt.imshow(image, extent=extent, origin = origin, vmax=vmax, vmin = vmin,  cmap = cmap)
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


def plot_image(image_raw, xlabel = None, ylabel = None, percentile_inf = None, percentile_sup = None, extent = None, origin = 'upper', vmax = None, vmin = None, title = '', cmap = 'viridis'):
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
        vmax = np.nanmax(image)


    if not vmin :
        vmin = np.nanmin(image)
    plt.imshow(image, extent=extent, origin = origin, vmax=vmax, vmin = vmin, cmap = cmap)
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
    # print(f"Saved updated .npz to "{output_path})



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


def find_RZextrema_between_2_points(point1, point2):

    # Example points
    R1, phi1, Z1 = point1
    R2, phi2, Z2 = point2

    # Convert to Cartesian coordinates
    x1, y1 = R1 * np.cos(phi1), R1 * np.sin(phi1)
    x2, y2 = R2 * np.cos(phi2), R2 * np.sin(phi2)

    # Number of interpolation points
    N = 1000
    t = np.linspace(0, 1, N)

    # Linear interpolation in Cartesian coordinates
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    z = Z1 + t * (Z2 - Z1)

    # Convert back to cylindrical R
    R = np.sqrt(x**2 + y**2)

    # Find extrema
    R_min, R_max = R.min(), R.max()
    Z_min, Z_max = z.min(), z.max()

    return R_min, R_max, Z_min, Z_max




def save_array_as_gif(array_3d, gif_path='output.gif', num_frames=100, cmap='Greys'):
    """
    Saves a 3D NumPy array (time, x, y) as a GIF with num_frames.

    Parameters:
        array_3d : np.ndarray
            3D array with shape (time, x, y)
        gif_path : str
            File path for output GIF
        num_frames : int
            Number of frames to include in the GIF (evenly sampled)
        cmap : str
            Matplotlib colormap for visualization
    """
    from tempfile import TemporaryDirectory

    assert array_3d.ndim == 3, "Input must be a 3D array (time, x, y)"

    time_dim = array_3d.shape[0]
    frame_indices = np.linspace(0, time_dim - 1, num_frames, dtype=int)

    with TemporaryDirectory() as tmpdir:
        filenames = []

        for i, idx in enumerate(frame_indices):
            fig, ax = plt.subplots()
            ax.axis('off')
            im = ax.imshow(array_3d[idx], cmap=cmap)
            fig.tight_layout(pad=0)

            frame_path = os.path.join(tmpdir, f'frame_{i:03d}.png')
            plt.savefig(frame_path, bbox_inches='tight', pad_inches=0)
            plt.close(fig)

            filenames.append(frame_path)

        # Create GIF
        images = [imageio.imread(fname) for fname in filenames]
        imageio.mimsave(gif_path, images, duration=0.3)  # duration per frame in seconds

    print(f"GIF saved to {gif_path}")





def save_array_as_img(array_3d, img_path='output.png', cmap='Greys'):
    """
    Saves a 3D NumPy array (time, x, y) as a GIF with num_frames.

    Parameters:
        array_3d : np.ndarray
            3D array with shape (time, x, y)
        gif_path : str
            File path for output GIF
        num_frames : int
            Number of frames to include in the GIF (evenly sampled)
        cmap : str
            Matplotlib colormap for visualization
    """

    assert array_3d.ndim == 3, "Input must be a 3D array (time, x, y)"

    time_dim = array_3d.shape[0]//2

    fig, ax = plt.subplots()
    ax.axis('off')
    im = ax.imshow(array_3d[time_dim], cmap=cmap)
    fig.tight_layout(pad=0)

    plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


    print(f"Image saved to {img_path}")


def plot_line_from_cylindrical(point1, point2, fig = None, color = 'blue', label = 'Line between points'):
    R1, phi1, Z1 = point1
    R2, phi2, Z2 = point2
    # Convert to Cartesian
    x1, y1, z1 = R1 * np.cos(phi1), R1 * np.sin(phi1), Z1
    x2, y2, z2 = R2 * np.cos(phi2), R2 * np.sin(phi2), Z2

    # Interpolate points along the line
    N = 100
    t = np.linspace(0, 1, N)
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    z = z1 + t * (z2 - z1)

    # Plotting the line
    if not fig:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.axes[0]
    ax.plot(x, y, z, label=label, color=color)
    ax.scatter([x1, x2], [y1, y2], [z1, z2], color='red')  # endpoints

    # Labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show(block = False)
    return fig


def plot_cylindrical_coordinates(contour_data, fig = None):

    # --- Step 1: Extract R, φ, Z
    R = contour_data[..., 0]
    phi = contour_data[..., 1]
    Z = contour_data[..., 2]

    # --- Step 2: Convert to Cartesian
    X = R * np.cos(phi)
    Y = R * np.sin(phi)

    # --- Step 3: Plot with matplotlib
    if not fig:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.axes[0]

    # Optional: if it's a surface-like contour
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.7)

    # Or if it's just a wire/line contour:
    # ax.plot_wireframe(X, Y, Z, color='blue')

    # Axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Contour in Cartesian Coordinates')
    plt.tight_layout()
    plt.show(block = False)
    return fig

def save_rotated_image(filename):
    from PIL import Image

    # Load the image
    image = Image.open(filename)

    # Rotate 90 degrees counter-clockwise (use -90 for clockwise)
    rotated = image.rotate(90, expand=True)
    filenamebase ,ext = os.path.splitext(filename)
    # Save the rotated image
    rotated.save(filenamebase + '_rotated' + ext)


def save_transposed_image(filename, file_output = None):
    from PIL import Image

    # Load the image
    image = Image.open(filename)

    
    # Transpose: swap X and Y (like matrix transpose)
    transposed = image.transpose(Image.TRANSPOSE)
    filenamebase ,ext = os.path.splitext(filename)
    # Save the image
    if not file_output:
        transposed.save(filenamebase + '_correct_orientation' + ext)
    else:
        transposed.save(file_output + ext)


def create_quick_synth_image(transfert_matrix, mask_noeud, mask_pixel):
    import random
    import fonction_tomo
    visible_nodes, = np.where(mask_noeud.flatten())

    ind_node_1D = random.choice(range(len(visible_nodes)))
    ind_node_2D = visible_nodes[ind_node_1D]

    synth_node_1D = np.zeros(len(visible_nodes))
    synth_node_1D[ind_node_1D] = 1
    synth_node_2D = fonction_tomo.reconstruct_2D_image(synth_node_1D, mask_noeud)
    synth_image_1D = transfert_matrix.dot(synth_node_1D)
    synth_image_2D = fonction_tomo.reconstruct_2D_image(synth_image_1D, mask_pixel)

    return synth_node_2D, synth_image_2D


def plot_comparison_synth_inversion(transfert_matrix, mask_noeud, mask_pixel, extent_RZ):


    synth_node_2D, synth_image_2D = create_quick_synth_image(transfert_matrix, mask_noeud, mask_pixel)

    fig = plt.figure()
    ax = fig.add_subplot(121)
    ax.imshow(synth_image_2D.T)
    ax = fig.add_subplot(122)
    ax.imshow(synth_node_2D.T, origin = 'lower', extent = extent_RZ )
    plt.show(block = False)
    return synth_node_2D, synth_image_2D





def plot_comparison_synth_inversion_noise(transfert_matrix, mask_noeud, mask_pixel, extent_RZ, noise = 0, inversion_method = 'lstsq'):


    synth_node_2D, synth_node_2D_noise_inversed, synth_image_2D, synth_image_2D_noise = create_quick_synth_image_noise(transfert_matrix, mask_noeud, mask_pixel, noise, inversion_method)

    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.imshow(synth_image_2D.T)
    ax = fig.add_subplot(222)
    ax.imshow(synth_node_2D.T, origin = 'lower', extent = extent_RZ )
    ax = fig.add_subplot(223)
    ax.imshow(synth_image_2D_noise.T)
    ax = fig.add_subplot(224)
    ax.imshow(synth_node_2D_noise_inversed.T, origin = 'lower', extent = extent_RZ )
    
    
    plt.show(block = False)
    return synth_node_2D, synth_image_2D



def create_quick_synth_image_noise(transfert_matrix, mask_noeud, mask_pixel, noise, inversion_method):
    import random
    import fonction_tomo
    visible_nodes, = np.where(mask_noeud.flatten())

    ind_node_1D = random.choice(range(len(visible_nodes)))
    ind_node_2D = visible_nodes[ind_node_1D]

    synth_node_1D = np.zeros(len(visible_nodes))
    synth_node_1D[ind_node_1D] = 1
    synth_node_2D = fonction_tomo.reconstruct_2D_image(synth_node_1D, mask_noeud)

    synth_image_1D = transfert_matrix.dot(synth_node_1D)
    synth_image_2D = fonction_tomo.reconstruct_2D_image(synth_image_1D, mask_pixel)
    synth_image_2D_noise = synth_image_2D + noise*np.random.normal(loc = 0, scale = synth_image_2D.max(), size = synth_image_2D.shape)

    synth_image_2D_noise_reduced = synth_image_2D_noise[mask_pixel]
    from inversion_module import inversion_and_thresolding
    inv_images = inversion_and_thresolding(synth_image_2D_noise_reduced[np.newaxis, :], transfert_matrix, inversion_method, mask = mask_noeud)[0]
    synth_node_1D_noise_inversed = np.squeeze(inv_images)
    synth_node_2D_noise_inversed = fonction_tomo.reconstruct_2D_image(synth_node_1D_noise_inversed, mask_noeud)
    return synth_node_2D, synth_node_2D_noise_inversed, synth_image_2D, synth_image_2D_noise



def gaussian_blur_video(video, sigma=1):
    from scipy.ndimage import gaussian_filter
    return np.array([gaussian_filter(frame, sigma=sigma) for frame in video])




def create_quick_synth_image_noise_nr_nz(nr, nz, transfert_matrix, mask_noeud, mask_pixel, noise, inversion_method):
    import random
    import fonction_tomo
    # visible_nodes, = np.where(mask_noeud.flatten())

    # ind_node_1D = random.choice(range(len(visible_nodes)))
    # ind_node_2D = visible_nodes[ind_node_1D]

    # synth_node_1D = np.zeros(len(visible_nodes))
    # synth_node_1D[ind_node_1D] = 1
    # synth_node_2D = fonction_tomo.reconstruct_2D_image(synth_node_1D, mask_noeud)
    synth_node_2D = np.zeros(mask_noeud.shape)
    synth_node_2D[nr, nz] = 1
    synth_node_1D = synth_node_2D[mask_noeud]
    synth_image_1D = transfert_matrix.dot(synth_node_1D)
    synth_image_2D = fonction_tomo.reconstruct_2D_image(synth_image_1D, mask_pixel)
    synth_image_2D_noise = synth_image_2D + noise*np.random.normal(loc = 0, scale = synth_image_2D.max(), size = synth_image_2D.shape)

    synth_image_2D_noise_reduced = synth_image_2D_noise[mask_pixel]
    from inversion_module import inversion_and_thresolding
    inv_images = inversion_and_thresolding(synth_image_2D_noise_reduced[np.newaxis, :], transfert_matrix, inversion_method, mask = mask_noeud)[0]
    synth_node_1D_noise_inversed = np.squeeze(inv_images)
    synth_node_2D_noise_inversed = fonction_tomo.reconstruct_2D_image(synth_node_1D_noise_inversed, mask_noeud)
    return synth_node_2D, synth_node_2D_noise_inversed, synth_image_2D, synth_image_2D_noise




def plot_comparison_synth_inversion_noise_nr_nz(nr, nz, transfert_matrix, mask_noeud, mask_pixel, extent_RZ, noise = 0, inversion_method = 'lstsq'):


    synth_node_2D, synth_node_2D_noise_inversed, synth_image_2D, synth_image_2D_noise = create_quick_synth_image_noise_nr_nz(nr, nz,transfert_matrix, mask_noeud, mask_pixel, noise, inversion_method)

    fig = plt.figure()
    ax = fig.add_subplot(221)
    ax.imshow(synth_image_2D.T)
    ax = fig.add_subplot(222)
    ax.imshow(synth_node_2D.T, origin = 'lower', extent = extent_RZ )
    ax = fig.add_subplot(223)
    ax.imshow(synth_image_2D_noise.T)
    ax = fig.add_subplot(224)
    ax.imshow(synth_node_2D_noise_inversed.T, origin = 'lower', extent = extent_RZ )
    
    
    plt.show(block = False)
    return synth_node_2D, synth_image_2D


__all__ = ["plot_image"]


import numpy as np

def arrays_to_side_by_side_video(arr1, arr2, filename="output.mp4", fps=20, normalize=True):
    import cv2

    """
    Create a side-by-side video from two 3D arrays (time, height, width).
    Adds a time overlay to each frame.

    Parameters
    ----------
    arr1 : np.ndarray
        First array of shape (time, h1, w1).
    arr2 : np.ndarray
        Second array of shape (time, h2, w2).
    filename : str
        Output video filename (e.g., "output.mp4").
    fps : int
        Frames per second.
    normalize : bool
        If True, scale arrays to 0-255 for display.
    """
    import matplotlib.cm as cm  
    
    # Check time dimension
    if arr1.shape[0] != arr2.shape[0]:
        raise ValueError("Both arrays must have the same time dimension")

    T = arr1.shape[0]
    viridis = cm.get_cmap("viridis")

    def to_colored(frame):
        # Normalize to [0,1]
        if normalize:
            frame = (frame - frame.min()) / (frame.ptp() + 1e-9)
        frame = np.clip(frame, 0, 1)

        # Apply colormap (returns RGBA in [0,1])
        colored = viridis(frame)[..., :3]  # drop alpha channel

        # Convert to uint8 BGR for OpenCV
        return (colored * 255).astype(np.uint8)[..., ::-1]

    frames = []
    for t in range(T):
        f1 = to_colored(arr1[t])
        f2 = to_colored(arr2[t])

        # Resize second frame to match first frame height
        h1, w1 = f1.shape[:2]
        h2, w2 = f2.shape[:2]
        if h1 != h2:
            f2 = cv2.resize(f2, (int(w2 * h1 / h2), h1))

        # Concatenate horizontally
        combined = np.hstack((f1, f2))

        # Overlay time (seconds based on fps)
        elapsed_time = t / fps
        text = f"t = {elapsed_time:.2f} s"
        cv2.putText(
            combined, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )

        frames.append(combined)

    # Video writer
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (w, h))

    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Video saved to {filename}")



def filename_contains_folder(filename: str, folder: str) -> bool:
    """
    Check if the filename already contains the folder name.
    
    Args:
        filename (str): The full or relative path to a file.
        folder (str): The folder name (not necessarily full path).
    
    Returns:
        bool: True if folder is part of filename path, False otherwise.
    """
    # Normalize paths
    filename_path = os.path.normpath(filename)
    folder_name = os.path.basename(os.path.normpath(folder))
    
    # Split filename into parts and check
    return folder_name in filename_path.split(os.sep)


def reconstruct_2D_image_all_slices(x_2d, mask):
    return np.array([reconstruct_2D_image(arr, mask) for arr in x_2d])


def save_array_as_video(array, filename, fps=20):
    import cv2

    """
    Save a 3D or 4D numpy array as a video file.

    Parameters
    ----------
    array : np.ndarray
        Video frames. 
        - If shape is (T, H, W): grayscale frames
        - If shape is (T, H, W, 3): color frames (RGB)
    filename : str
        Output video filename (e.g., 'output.mp4' or 'output.avi')
    fps : int
        Frames per second
    """
    # Check dimensions
    if array.ndim not in (3, 4):
        raise ValueError("Array must be 3D (T,H,W) or 4D (T,H,W,C)")

    T = array.shape[0]
    H, W = array.shape[1:3]

    # Choose color or grayscale
    if array.ndim == 3:
        is_color = False
    else:
        if array.shape[3] == 3:
            is_color = True
        else:
            raise ValueError("Last dimension must be 3 for RGB color frames")

    # Normalize if needed
    if array.dtype != np.uint8:
        array = (255 * (array - array.min()) / (array.max() - array.min())).astype(np.uint8)

    # OpenCV wants BGR, not RGB
    if is_color:
        array = array[..., ::-1]

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
    out = cv2.VideoWriter(filename, fourcc, fps, (W, H), isColor=is_color)

    # Write frames
    for i in range(T):
        frame = array[i]
        if not is_color:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # convert grayscale to BGR
        out.write(frame)

    out.release()
    print(f"Video saved as {filename}")






def _to_uint8(frames: np.ndarray, vmin: Optional[float]=None, vmax: Optional[float]=None) -> np.ndarray:
    """
    Normalize a float or int array to uint8 in range 0..255.
    frames: (T, H, W) or (H, W, T) already transformed before call.
    """
    # ensure float
    f = frames.astype(np.float32)
    if vmin is None:
        vmin = float(np.nanmin(f))
    if vmax is None:
        vmax = float(np.nanmax(f))
    if vmax == vmin:
        # constant image -> mid-gray
        out = np.clip(np.round((f - vmin) * 0 + 127), 0, 255).astype(np.uint8)
        return np.repeat(out[np.newaxis, ...], f.shape[0], axis=0) if f.ndim == 3 else out
    # linear scale
    scaled = (f - vmin) / (vmax - vmin)
    scaled = np.clip(scaled, 0.0, 1.0)
    out = (scaled * 255.0).round().astype(np.uint8)
    return out

def array3d_to_video(arr: np.ndarray,
                     out_path: str,
                     fps: int = 25,
                     vmin: Optional[float] = None,
                     vmax: Optional[float] = None,
                     percentile_sup : Optional[float] = None,
                     percentile_inf : Optional[float] = None,
                     codec: str = "libx264",
                     quality: int = 8
                    ) -> str:
    """
    Save a 3D numpy array to a video file without using cv2.
    
    Parameters
    ----------
    arr : np.ndarray
        3D array representing frames. Accepted shapes:
        - (T, H, W) : T frames of HxW grayscale images (preferred)
        - (H, W, T) : T frames as last dimension (will be transposed)
    out_path : str
        Output filename (e.g., "out.mp4", "out.mkv"). Extension determines container.
    fps : int
        Frames per second.
    vmin, vmax : optional floats
        Min and max used for normalization. If None, computed from array.
    codec : str
        Codec string passed to imageio/ffmpeg (e.g. "libx264").
    quality : int
        Quality parameter (used by imageio when available; 0-10-ish, implementation-defined).
    
    Returns
    -------
    str
        Path to the written video file.
    
    Notes
    -----
    - Uses `imageio` if installed; falls back to `matplotlib.animation.FFMpegWriter` if not.
    - Requires ffmpeg on PATH for the matplotlib fallback; imageio may also require ffmpeg depending on backend.
    - Input dtype can be float or int; the function scales values to uint8 0-255.
    """
    # Validate & rearrange shape to (T, H, W)
    if arr.ndim != 3:
        raise ValueError("Input array must be 3D (T, H, W) or (H, W, T).")
    a = arr
    
    T, H, W = a.shape
    
    a = clip_to_percentiles(a, low = percentile_inf, high = percentile_sup)
        
    # Convert to uint8 frames: shape (T, H, W)
    frames_u8 = _to_uint8(a, vmin=vmin, vmax=vmax)

    # If imageio is available, prefer it
    try:
        import imageio.v2 as iio
        # imageio expects (H,W) or (H,W,3). For grayscale it's fine.
        ext = os.path.splitext(out_path)[-1].lower()
        writer_kwargs = {}
        # imageio's ffmpeg writer accepts 'quality' param in some versions; we pass it regardless.
        try:
            # new imageio v3 uses imageio.v3.get_writer

            writer = iio.get_writer(out_path, format='FFMPEG', mode='I', fps=fps,
                       codec='libx264',
                       )
            for frame in frames_u8:
                writer.append_data(frame)
            writer.close()
        except:
            pass
    except Exception:
        # If imageio not installed, try matplotlib's FFMpegWriter (requires ffmpeg in PATH)
        try:
            import matplotlib.pyplot as plt
            from matplotlib.animation import FFMpegWriter
            fig, ax = plt.subplots()
            im = ax.imshow(frames_u8[0], cmap="gray", vmin=0, vmax=255)
            ax.set_axis_off()
            metadata = dict(title=os.path.basename(out_path), artist="array3d_to_video")
            # bitrate ~ quality * 1000 heuristic; user can change if needed
            bitrate = max(2000, quality * 1000)
            writer = FFMpegWriter(fps=fps, metadata=metadata, bitrate=bitrate)
            with writer.saving(fig, out_path, dpi=100):
                for frame in frames_u8:
                    im.set_array(frame)
                    writer.grab_frame()
            plt.close(fig)
            return out_path
        except Exception as e:
            raise RuntimeError(
                "Failed to write video: imageio not available and matplotlib/ffmpeg fallback failed.\n"
                "Install imageio (`pip install imageio imageio-ffmpeg`) or ensure ffmpeg is on PATH for matplotlib fallback.\n"
                f"Original error: {e}"
            )
        


def prep_image(array, vmin = None, vmax = None, percentile_inf = None, percentile_sup = None):
    vmin = vmin or np.min(array)
    vmax = vmax or np.max(array)
    percentile_inf = percentile_inf or 0
    percentile_sup = percentile_sup or 1





def get_vid(ParamsVid):
    path_vid = ParamsVid.path_vid
    time_input = ParamsVid.time_input
    frame_input = ParamsVid.frame_input
    nshot = ParamsVid.nshot
    dict_vid = ParamsVid.dict_vid
    time_input = ParamsVid.time_input
    from scipy import ndimage
    if path_vid:
        
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
            add_variable_to_npz(path_vid + '.npz', 't_start', t_start)
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

            if frame_input is None:
                out = RIS.get_info(nshot, RIS_number)
                frame_input =[0, out.daq_parameters.Images]
            if time_input is not None:
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
 

    sigma = dict_vid.get('sigma')
    sigma = sigma or 0
    if sigma:
        images = gaussian_blur_video(images, sigma=sigma)

    median = dict_vid.get('median')
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

    if 'reduce_frames' in  dict_vid.keys():
        reduce_frames = dict_vid['reduce_frames']
        images = average_along_first_row(images,reduce_frames)
        fps = fps/reduce_frames
    t_inv = t0+np.arange(images.shape[0])/fps
    return images, images.shape[0], image_dim_y, image_dim_x, fps, frame_input, name_time, t_start, t0, t_inv 
        

def clip_to_percentiles(data, low=10, high=90):
    """
    Clips values in 'data' to lie within the given percentile range.
    
    Parameters:
        data (array-like): Input data (list, tuple, or NumPy array)
        low (float): Lower percentile (default 10)
        high (float): Upper percentile (default 90)
        
    Returns:
        np.ndarray: Array with clipped values
    """
    data = np.asarray(data)
    lower = np.percentile(data, low)
    upper = np.percentile(data, high)
    return np.clip(data, lower, upper)