from scipy.sparse import load_npz, isspmatrix, csc_matrix, csr_matrix, save_npz

from scipy.interpolate import RegularGridInterpolator
import sys
import pickle
import pdb
import numpy as np
import scipy
from cherab.tools.inversions.opencl.sart_opencl import SartOpencl

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


def synth_inversion(transfert_matrix, mask_pixel, mask_noeud,  pixels, noeuds, nb_noeuds_r, nb_noeuds_z,R_noeud, Z_noeud, R_wall, Z_wall, inversion_method, derivative_matrix, noise = 0, num_structures = 4, size_struct = 4, inversion_parameter = {"rcond": 1e-3}, c_c = 3):
    import numpy as np
    from scipy.io import loadmat
    from scipy.sparse import load_npz, csr_matrix, eye, issparse
    import random
    # transfert_matrix = load_npz(transfert_matrix + '.npz')
    image_full = np.zeros(mask_pixel.shape)
    node_full = np.zeros((nb_noeuds_r, nb_noeuds_z))

    # for i in range(0,num_structures):
    #         loop = 1
    #         while loop:
    #             node = random.randint(0, len(noeuds)-1)

    #             r, z = np.argwhere(np.squeeze(mask_noeud))[node, :]
    #             if i ==0:
    #                 struct= np.random.normal(loc = 1, size = (size_struct, size_struct))
    #             else:
    #                 struct= np.random.normal(loc = np.random.rand(), size = (size_struct, size_struct))
    #             try:
    #                 node_full[r-size_struct//2:r+size_struct//2, z-size_struct//2:z+size_struct//2] = struct
    #                 loop = 0
    #             except:
    #                 print('try again')
    
    node_full[:] = 0
    r = nb_noeuds_r//2
    z = nb_noeuds_z//2
    for theta in range(60):
        theta_ang = theta*6
        rc = int(r+8*np.cos(theta_ang/180*np.pi)) 
        zc = int(z+8*np.sin(theta_ang/180*np.pi))
        node_full[rc, zc] = np.cos(theta_ang/180*np.pi)*np.sin(theta_ang/180*np.pi)+np.cos(theta_ang/180*np.pi) + 4
    
    node_masked =  node_full[np.squeeze(mask_noeud)]             
    extent = (R_noeud[0], R_noeud[-1], Z_noeud[0], Z_noeud[-1])
    image = transfert_matrix.dot(node_masked) 
    image_noise = image + noise*np.random.normal(loc = 0, scale = image.max(), size = image.shape)
    image_full = reconstruct_2D_image(image, mask_pixel, mask_pixel.shape[0],  mask_pixel.shape[1])
    image_full_noise = reconstruct_2D_image(image_noise, mask_pixel, mask_pixel.shape[0], mask_pixel.shape[1])
    images_noise = image_noise[np.newaxis, :]

    inv_image, inv_normed, inv_image_thresolded, inv_image_thresolded_normed, image_retrofit, mask, transfert_matrix = inversion_and_thresolding(images_noise, transfert_matrix, inversion_method, c_c = c_c, inversion_parameter = inversion_parameter, derivative_matrix = derivative_matrix, mask = mask_noeud)
    inv_image = np.squeeze(inv_image)
    inv_normed = np.squeeze(inv_normed)
    inv_image_thresolded = np.squeeze(inv_image_thresolded)
    inv_image_thresolded_normed = np.squeeze(inv_image_thresolded_normed)
    image_retrofit = np.squeeze(image_retrofit)
    image_full_noise = np.squeeze(image_full_noise)
    image_full = np.squeeze(image_full)
    return node_full, inv_image, inv_normed, inv_image_thresolded, inv_image_thresolded_normed, image_retrofit, image_full_noise, image_full
    



def get_derivative_matrix(inversion_method, R_noeud = None, Z_noeud = None, magflux = None, mask = None, dr = 1.0, dz= 1.0):
    if inversion_method == 'Mfr_Cherab':
        from cherab.inversion.derivative import derivative_matrix
        dmat_r = derivative_matrix(mask.shape, dr, axis=0, scheme="forward", mask=mask)
        dmat_z = derivative_matrix(mask.shape, dz, axis=1, scheme="forward", mask=mask)

        dmat_pair = [(dmat_r, dmat_r), (dmat_z, dmat_z)]
        return dmat_pair
    elif inversion_method == 'Cholmod':
        imid = len(magflux.time)//2
        interp = RegularGridInterpolator((magflux.interp2D.r[:, 0], magflux.interp2D.z[0, :]), magflux.interp2D.psi[imid, :, :], bounds_error = False)
        RZ_grid = RegularGrid(len(R_noeud), len(Z_noeud),[min(R_noeud), max(R_noeud)],[min(Z_noeud), max(Z_noeud)])
        #RZ are swapped because meshgrid indexing is different than the rest
        # RZ_grid = RegularGrid(len(Z_noeud), len(R_noeud),[min(Z_noeud), max(Z_noeud)],[min(R_noeud), max(R_noeud)])
        maggrid = interp((RZ_grid.center_mesh[1].T, RZ_grid.center_mesh[0].T))
        derivative_matrix = compute_aniso_dmats(RZ_grid, maggrid)
        return derivative_matrix
    else:
        return None

def inversion_and_thresolding(images, transfert_matrix, inversion_method, c_c = 3, inversion_parameter = {"rcond" : 1e-3}, derivative_matrix = None, mask = None, R_noeud = None, Z_noeud = None):
    from cherab.tools.inversions import invert_regularised_nnls
    # images = images[:, None] #add a dimension for the computation to procede correctly
    # transfert_matrix = transfert_matrix.todense()
    # transfert_matrix[:, np.sum(transfert_matrix, 0)<1e-3] = 0
    # transfert_matrix = csr_matrix(transfert_matrix)
    mask_copy = np.squeeze(mask.copy())


    if isspmatrix(transfert_matrix):
        transfert_matrix = transfert_matrix.todense() 
    if inversion_method == 'Cholmod':

        inv_images = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_normed = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_images_thresolded = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_images_thresolded_normed = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        images_retrofit = np.zeros_like(images)

        inversion = CholmodMfr()
        for i in range(images.shape[0]):
            image = images[i, :]
            inv_image, dict_inv = inversion.invert(image, transfert_matrix, derivative_matrix)
            inv_images[i, :] = inv_image
            inv_normed[i, :] = inv_image
            inv_images_thresolded[i, :] = inv_image
            inv_images_thresolded_normed[i, :] = inv_image
            images_retrofit[i, :] = transfert_matrix.dot(inv_image)  

    elif inversion_method == 'lstsq':
        from tomotok.core.inversions import Bob, SparseBob, CholmodMfr, Mfr
        from tomotok.core.derivative import compute_aniso_dmats
        from tomotok.core.geometry import RegularGrid
        inversion = SparseBob()
        simple_base = csr_matrix(np.eye( transfert_matrix.shape[1] ))
        transfert_matrix = csr_matrix(transfert_matrix)
        inversion.decompose(transfert_matrix, simple_base)
        # inv_images = inversion(images.T)
        inversion.normalise()
        # inv_normed = np.divide(inv_images, inversion.norms)
        # inv_images_thresolded  = inversion.thresholding(images, c_c)
        # inv_images_thresolded_normed = np.divide(inv_images_thresolded,inversion.norms)

        inv_images = inversion(images.T) #put the image in the #pixels, times dimension order
        inv_normed = np.divide(inv_images, inversion.norms)
        inv_images_thresolded = np.zeros((transfert_matrix.shape[1], images.shape[0]))
        inv_images_thresolded_normed = np.zeros((transfert_matrix.shape[1], images.shape[0]))
        images_retrofit = np.zeros_like(images)
        for i in range(images.shape[0]):
        #     image = images[i, :]
        #     mfr = Mfr_Cherab(transfert_matrix, derivative_matrix, data = image)
        #     inv_image, norms = mfr.solve()
        #     inv_images[i, :] = inv_image
        #     inv_normed[i, :] = inv_image
            image = images.T[:, i]
            inv_images_thresolded[:, i] = np.squeeze(inversion.thresholding(image[:, np.newaxis], c = c_c))
        #     inv_images_thresolded_normed[i, :] = inv_image
            inv_image = inv_images[:, i] 
            images_retrofit[i, :] = transfert_matrix.dot(inv_image)
        #put back the inversion in the times, nodes order
        inv_images = inv_images.T 
        inv_normed = inv_normed.T
        inv_images_thresolded = inv_images_thresolded.T
        inv_images_thresolded_normed = inv_images_thresolded_normed.T
    elif inversion_method == 'Mfr':
        inversion = Mfr()
        simple_base = csr_matrix(np.eye( transfert_matrix.shape[1] ))
        inversion.decompose(transfert_matrix, simple_base, solver_kw= inversion_parameter)
        inv_images = inversion(images, simple_base)

        images_retrofit = transfert_matrix.dot(inv_image.T)
    elif inversion_method == 'nnls':
        try:
            alpha = inversion_parameter["alpha"]
        except:
            alpha = 0.01

        inv_images = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_normed = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_images_thresolded = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_images_thresolded_normed = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        images_retrofit = np.zeros_like(images)
        for i in range(images.shape[0]):
            image = images[i, :]
            inv_image, norms = invert_regularised_nnls(transfert_matrix.toarray(), image, alpha=alpha, tikhonov_matrix=None)
            inv_images[i, :] = inv_image
            inv_normed[i, :] = inv_image
            inv_images_thresolded[i, :] = inv_image
            inv_images_thresolded_normed[i, :] = inv_image
            images_retrofit[i, :] = transfert_matrix.dot(inv_image)  

    elif inversion_method == 'Mfr_Cherab':
        from cherab.inversion import Mfr as Mfr_Cherab

        derivative_matrix = get_derivative_matrix(inversion_method, mask = mask_copy)
        inv_images = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_normed = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_images_thresolded = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_images_thresolded_normed = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        images_retrofit = np.zeros_like(images)
        for i in range(images.shape[0]):
            image = images[i, :]
            mfr = Mfr_Cherab(transfert_matrix, derivative_matrix, data = image)
            inv_image, norms = mfr.solve()
            inv_images[i, :] = inv_image
            inv_normed[i, :] = inv_image
            inv_images_thresolded[i, :] = inv_image
            inv_images_thresolded_normed[i, :] = inv_image
            images_retrofit[i, :] = transfert_matrix.dot(inv_image)


    elif inversion_method == 'SART':
        import pyopencl as cl
        platform = cl.get_platforms()[0]
        device = platform.get_devices()[0]

        invert_sart = SartOpencl(transfert_matrix, device = device)

        inv_images = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_normed = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_images_thresolded = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_images_thresolded_normed = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        images_retrofit = np.zeros_like(images)
        for i in range(images.shape[0]):
            image = images[i, :]
        #     if i>0:
        #         inv_image, residual = invert_sart(image, initial_guess=inv_images[i-1, :])
        #     else:
        #         inv_image, residual = invert_sart(image)
            inv_image, residual = invert_sart(np.squeeze(image))
            inv_images[i, :] = inv_image
            inv_normed[i, :] = inv_image
            inv_images_thresolded[i, :] = inv_image
            inv_images_thresolded_normed[i, :] = inv_image
            images_retrofit[i, :] = transfert_matrix.dot(inv_image)  
                                          
    else:
        raise Exception('unrecognised inversion method')
    # raise RuntimeError(f"Failed to deserialize input: {inv_image}")

    return inv_images, inv_normed, inv_images_thresolded, inv_images_thresolded_normed, images_retrofit, mask, transfert_matrix
        


def reconstruct_2D_image(image, mask, dim_r = None, dim_z = None):
    if dim_r and len(mask.shape)==2:
        if (dim_r, dim_z) != mask.shape:
            raise Exception("careful, dimensions are wrong in reconstruction of full image")
    image_reconstructed = np.zeros(mask.shape)
    if type(image) == np.matrix:
        image = np.squeeze(np.array(image))
    image_reconstructed[mask] = image
    if image_reconstructed.ndim ==3:
        image_reconstructed = np.sum(image_reconstructed, 1)

    return image_reconstructed
                                   
def plot_results_inversion(inv_image, inv_normed, inv_image_thresolded, inv_image_thresolded_normed, transfert_matrix, image, mask_pixel, mask_noeud, pixels, noeuds, R_wall, Z_wall, nb_noeuds_r, nb_noeuds_z, R_noeud, Z_noeud, c_c = 3):
    extent = (R_noeud[0], R_noeud[-1], Z_noeud[0], Z_noeud[-1])
    image_retrofit = transfert_matrix.dot(inv_image)
    inv_image_full = reconstruct_2D_image(inv_image, mask_noeud, nb_noeuds_r, nb_noeuds_z)
    inv_normed_full = reconstruct_2D_image(inv_normed, mask_noeud, nb_noeuds_r, nb_noeuds_z)    
    inv_image_thresolded_full = reconstruct_2D_image(inv_image_thresolded, mask_noeud, nb_noeuds_r, nb_noeuds_z)
    inv_image_thresolded_normed_full = reconstruct_2D_image(inv_image_thresolded_normed, mask_noeud, nb_noeuds_r, nb_noeuds_z)
    image_retrofit_full = reconstruct_2D_image(image_retrofit, mask_pixel, mask_pixel.shape[0], mask_pixel.shape[1])
    figure_results =plt.figure()
    #synthetic image
    plt.subplot(2,3,1)
    plt.imshow(image)
    plt.colorbar()
    plt.title('image')

    #retro fit
    plt.subplot(2,3,2)
    plt.imshow(image_retrofit_full)
    plt.colorbar()
    plt.title('Retro fit')

    #inversion
    plt.subplot(2,3,3)
    plt.imshow(inv_image_full.T, extent = extent, origin = 'lower')
    plt.colorbar()
    plt.plot(R_wall, Z_wall, 'r')
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.title('inversed image')
    
    #normed inversion
    plt.subplot(2,3,4)
    plt.imshow(inv_normed_full.T, extent = extent, origin = 'upper')
    plt.colorbar()
    plt.plot(R_wall, Z_wall, 'r')
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.title('inversed normalized image')
    # thresolded inversion
    plt.subplot(2,3,5)
    plt.imshow(inv_image_thresolded_full, extent = extent, origin = 'lower')
    plt.colorbar()
    plt.plot(R_wall, Z_wall, 'r')
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    # thresolded and normalized inversion
    plt.subplot(2,3,6)
    plt.imshow(inv_image_thresolded_normed_full, extent = extent)
    plt.colorbar()
    plt.title('inversed image thresolded and normalized, c_c = '+ str(c_c))
    plt.plot(R_wall, Z_wall, 'r')
    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.tight_layout()
    return figure_results

def inverse_vid(transfert_matrix, mask_pixel,mask_noeud, vid, R_noeud, Z_noeud, inversion_method, inversion_parameter, derivative_matrix = None):
    #initisalisation
    # images = np.reshape(vid, (vid.shape[0], vid.shape[1]*vid.shape[2]))
    # images = images[:, mask_pixel]
    images = vid[:, mask_pixel]
    inversion_results, inversion_results_normed, inversion_results_thresolded,inversion_results_thresolded_normed, images_retrofit, mask_noeud, transfert_matrix = inversion_and_thresolding(images, 
                                                                                                                                                               transfert_matrix, 
                                                                                                                                                               inversion_method, 
                                                                                                                                                               c_c = 3, 
                                                                                                                                                               inversion_parameter =inversion_parameter, 
                                                                                                                                                               derivative_matrix = derivative_matrix,
                                                                                                                                                               mask = mask_noeud,
                                                                                                                                                               R_noeud = R_noeud, 
                                                                                                                                                               Z_noeud = Z_noeud)

	
    return inversion_results, inversion_results_normed, inversion_results_thresolded,inversion_results_thresolded_normed, images_retrofit, mask_noeud, transfert_matrix



def f(x, y, arrayprint):
    print(arrayprint)
    return x**2 + y

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


def test_function(sparse_matrix, array):
    return {"sparse_sum": sparse_matrix.sum(), "array_sum": array.sum()}



if __name__ == "__main__":
    # try:
    #     kdhfz
    # except Exception as e:
    #     raise RuntimeError(f"Failed to deserialize input: {sys.executable}")
    
    # # Debug: Check if input data is received
    f = open("demofile2.txt", "a")
    f.write("New try")
    f.write(sys.version)
    f.write(sys.executable)
    f.close()
    raw_input = sys.stdin.buffer.read()
    if not raw_input:
        raise RuntimeError("No input received by subprocess.")
    print("Raw Input Received.")
    # Deserialize input data
    try:
        input_data = pickle.loads(raw_input)
    except Exception as e:
        raise RuntimeError(f"Failed to deserialize input: {e}")

    func_name = input_data["func_name"]
    args = [deserialize_data(arg) for arg in input_data["args"]]
    # sys.stdout.buffer.write(pickle.dumps(args))

    if func_name in globals() and callable(globals()[func_name]):
        result = globals()[func_name](*args)
        sys.stdout.buffer.write(pickle.dumps(result))
    else:
        sys.stdout.buffer.write(pickle.dumps({"error": f"Function {func_name} not found"}))



def get_name(path):
    path = path.split('/')
    name = path[len(path)-1]
    name_shortened = name.split('.')

    name_shortened = name_shortened[0]


    return name_shortened




                        
def plot_results_inversion_simplified(inv_image, transfert_matrix, image, mask_pixel, mask_noeud, pixels, noeuds, R_wall, Z_wall, nb_noeuds_r, nb_noeuds_z, R_noeud, Z_noeud, c_c = 3, cmap = 'viridis', norm = 'linear', magflux = None):
    extent = (R_noeud[0], R_noeud[-1], Z_noeud[0], Z_noeud[-1])
    image_retrofit = transfert_matrix.dot(inv_image)
    inv_image_full = reconstruct_2D_image(inv_image, mask_noeud, nb_noeuds_r, nb_noeuds_z)
    image_retrofit_full = reconstruct_2D_image(image_retrofit, mask_pixel, mask_pixel.shape[0], mask_pixel.shape[1])
    import matplotlib.colors as mcolors
    if norm == 'log':
        image = np.log2(image+1)   
        image_retrofit_full = np.log2(image_retrofit_full+1)
        inv_image_full = np.log2(inv_image_full+1)
    vmin = image.min()
    vmax = image.max()
    

    fig = plt.figure(figsize=(12, 7))
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.6, hspace=0.6)
    
    #real image
    axi = fig.add_subplot(2, 2,1)
    plt.imshow(image.T, cmap = cmap, vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.title('image')

    #retro fit
    axi = fig.add_subplot(2, 2,2)
    plt.imshow(image_retrofit_full.T, cmap = cmap, vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.title('Retro fit')

    #inversion
    axi = fig.add_subplot(2, 2,3)
    plt.imshow(inv_image_full.T, extent = extent, origin = 'lower', cmap = cmap, vmin = -1000, vmax = 1000)
    plt.colorbar()
    plt.plot(R_wall, Z_wall, 'r')
    # if magflux:
    #     nbr_levels = 30
    #     levels_req = np.linspace(np.nanmin(magflux.psi), \
    #                         np.nanmax(magflux.psi), \
    #                         nbr_levels)
    #     axi.contour(magflux.r, magflux.z, \
    #                 np.squeeze(magflux.psi), \
    #                 levels=levels_req, colors='blue', linestyles='-', \
    #                 linewidths=0.5)
    #     # Separatrix plot
    #     axi.contour(magflux.r, magflux.z, \
    #                 np.squeeze(magflux.psi), \
    #                 levels=(magflux.psisep,), linestyles='-', colors='tab:red')

    plt.xlabel('R [m]')
    plt.ylabel('Z [m]')
    plt.title('inversed image')
    plt.show(block = False)
    #error retro fit
    axi = fig.add_subplot(2, 2,4)
    plt.imshow(np.abs(image_retrofit_full-image)/image, cmap = cmap)
    plt.colorbar()
    plt.title('Error Retro fit')
    return fig

