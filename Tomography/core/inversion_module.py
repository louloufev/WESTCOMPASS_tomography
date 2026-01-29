from scipy.sparse import load_npz, isspmatrix, csc_matrix, csr_matrix, save_npz
import os
from scipy.interpolate import RegularGridInterpolator
import sys
import pickle
import pdb
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from Tomography.core import utility_functions

def synth_inversion(transfert_matrix, mask_pixel, mask_noeud,  pixels, noeuds, nb_noeuds_r, nb_noeuds_z,R_noeud, Z_noeud, R_wall, Z_wall, inversion_method, derivative_matrix, noise = 0, num_structures = 4, size_struct = 4, inversion_parameter = {"rcond": 1e-3}, c = 3):
    """
    Function that generate a synthetic image from a given geometry matrix. Can generate a random pattern of emissivity or random noise
    (currently broken)

    Inputs :
    transfert_matrix
    mask_pixel
    mask_noeud
    pixels
    noeuds
    nb_noeuds_r
    nb_noeuds_z
    R_noeud
    Z_noeud
    R_wall
    Z_wall
    inversion_method
    derivative_matrix
    noise = 0
    num_structures = 4
    size_struct = 4
    inversion_parameter = {"rcond": 1e-3}
    
    Outputs :
    node_full, inv_image, inv_normed, inv_image_thresholded, inv_image_thresholded_normed, image_retrofit, image_full_noise, image_full
    
    
    
    
    """
    
    
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

    inv_image, inv_normed, inv_image_thresholded, inv_image_thresholded_normed, image_retrofit, transfert_matrix = inversion_and_thresolding(images_noise, transfert_matrix, inversion_method, folder_inverse_matrix, dict_vid, inversion_parameter)
    inv_image = np.squeeze(inv_image)
    inv_normed = np.squeeze(inv_normed)
    inv_image_thresholded = np.squeeze(inv_image_thresholded)
    inv_image_thresholded_normed = np.squeeze(inv_image_thresholded_normed)
    image_retrofit = np.squeeze(image_retrofit)
    image_full_noise = np.squeeze(image_full_noise)
    image_full = np.squeeze(image_full)
    return node_full, inv_image, inv_normed, inv_image_thresholded, inv_image_thresholded_normed, image_retrofit, image_full_noise, image_full
    



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

def inversion_and_thresolding(images, transfert_matrix, inversion_method, folder_inverse_matrix, dict_vid= {}, inversion_parameter = {"rcond" : -1}, c = None):
    """
    Inputs :
    images : 2D array
    video data, in the #times, #pixels order (pixels grid has been previously flattened)
    transfert_matrix : csr_matrix
    geometry matrix, in the #pixels, #nodes order
    inversion_method : string
    name of the inversion method used
    folder_inverse_matrix : string of a folder
    folder where to save or load the inverse matrix
    inversion_parameter : dictionnary, optionnal
    dictionnary to add optionnal parameters for inversion
    dict_vid : dictionnary, optionnal
    dictionnary to add instructions on operations annex to the inversion (filtering, denoising, etc...)

    
    Outputs :

    inv_images : 2D array
    video inversed, in the #times, #nodes order 
    inv_normed : 2D array
    video inversed and normalized, in the #times, #nodes order 
    inv_images_thresholded :2D array
    video inversed and denoized, in the #times, #nodes order  
    inv_images_thresholded_normed : 2D array
    video inversed and normalized and denoized, in the #times, #nodes order 
    images_retrofit: 2D array
    video reconstructed from nodes profile, in the #times, #pixels order 
    transfert_matrix : csr_matrix 
    geometry matrix, in the #pixels, #nodes order
    """
    
    
    
    
    from cherab.tools.inversions import invert_regularised_nnls
    # images = images[:, None] #add a dimension for the computation to procede correctly
    # transfert_matrix = transfert_matrix.todense()
    # transfert_matrix[:, np.sum(transfert_matrix, 0)<1e-3] = 0
    # transfert_matrix = csr_matrix(transfert_matrix)


    if isspmatrix(transfert_matrix):
        transfert_matrix = transfert_matrix.todense() 
    if inversion_method == 'Cholmod':

        inv_images = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_normed = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_images_thresholded = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_images_thresholded_normed = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        images_retrofit = np.zeros_like(images)

        inversion_class = CholmodMfr()
        for i in range(images.shape[0]):
            image = images[i, :]
            inv_image, dict_inv = inversion_class.invert(image, transfert_matrix, derivative_matrix)
            inv_images[i, :] = inv_image
            inv_normed[i, :] = inv_image
            inv_images_thresholded[i, :] = inv_image
            inv_images_thresholded_normed[i, :] = inv_image
            images_retrofit[i, :] = transfert_matrix.dot(inv_image)  

    elif inversion_method == 'Bob':
        from tomotok.core.inversions import Bob, SparseBob, CholmodMfr, Mfr
        from tomotok.core.derivative import compute_aniso_dmats
        from tomotok.core.geometry import RegularGrid
        
        c = c or 0



        inversion_class = Bob()
        simple_base = csr_matrix(np.eye( transfert_matrix.shape[1] ))
        transfert_matrix = csr_matrix(transfert_matrix)
        rcond = inversion_parameter.get('rcond')
        solver_dict = {'rcond' : rcond}
        

        path_inverse_matrix = folder_inverse_matrix
        path_norm_matrix = folder_inverse_matrix + 'norm'

        try:
            inversion_class.load_decomposition(path_inverse_matrix)
            print('successfully loaded inverse matrix')
        except:
            inversion_class.decompose(transfert_matrix, simple_base, solver_kw = solver_dict)
            inversion_class._normalise_wo_mat()
        os.makedirs(os.path.dirname(path_inverse_matrix), exist_ok = True)
        inversion_class.save_decomposition(path_inverse_matrix)

        inv_images = inversion_class(images.T) #put the image in the #pixels, times dimension order
        inv_normed = np.divide(inv_images, inversion_class.norms)
        inv_images_thresholded = np.zeros((transfert_matrix.shape[1], images.shape[0]))
        inv_images_thresholded_normed = np.zeros((transfert_matrix.shape[1], images.shape[0]))
        images_retrofit = np.zeros_like(images)
        for i in range(images.shape[0]):
        #     image = images[i, :]
        #     mfr = Mfr_Cherab(transfert_matrix, derivative_matrix, data = image)
        #     inv_image, norms = mfr.solve()
        #     inv_images[i, :] = inv_image
        #     inv_normed[i, :] = inv_image
            image = images.T[:, i]
            inv_images_thresholded[:, i] = np.squeeze(inversion_class.thresholding(image[:, np.newaxis], c = c))
        #     inv_images_thresholded_normed[i, :] = inv_image
            inv_image = inv_images[:, i] 
            images_retrofit[i, :] = transfert_matrix.dot(inv_image)
        #put back the inversion in the times, nodes order
        inv_images = inv_images.T 
        inv_normed = inv_normed.T
        inv_images_thresholded = inv_images_thresholded.T
        inv_images_thresholded_normed = inv_images_thresholded_normed.T

    elif inversion_method == 'SparseBob':
        from tomotok.core.inversions import Bob, SparseBob, CholmodMfr, Mfr
        from tomotok.core.derivative import compute_aniso_dmats
        from tomotok.core.geometry import RegularGrid
        
        c = c or 0



        inversion_class = SparseBob()
        simple_base = csr_matrix(np.eye( transfert_matrix.shape[1] ))
        transfert_matrix = csr_matrix(transfert_matrix)


        path_inverse_matrix = folder_inverse_matrix 
        path_norm_matrix = folder_inverse_matrix + 'norm'

        try:
            inversion_class.load_decomposition(path_inverse_matrix)
            print('successfully loaded inverse matrix')
        except:
            inversion_class.decompose(transfert_matrix, simple_base)
            inversion_class._normalise_wo_mat()
            os.makedirs(os.path.dirname(path_inverse_matrix), exist_ok = True)
            inversion_class.save_decomposition(path_inverse_matrix)

        inv_images = inversion_class(images.T) #put the image in the #pixels, times dimension order
        inv_normed = np.divide(inv_images, inversion_class.norms)
        inv_images_thresholded = np.zeros((transfert_matrix.shape[1], images.shape[0]))
        inv_images_thresholded_normed = np.zeros((transfert_matrix.shape[1], images.shape[0]))
        images_retrofit = np.zeros_like(images)
        for i in range(images.shape[0]):
        #     image = images[i, :]
        #     mfr = Mfr_Cherab(transfert_matrix, derivative_matrix, data = image)
        #     inv_image, norms = mfr.solve()
        #     inv_images[i, :] = inv_image
        #     inv_normed[i, :] = inv_image
            image = images.T[:, i]
            inv_images_thresholded[:, i] = np.squeeze(inversion_class.thresholding(image[:, np.newaxis], c = c))
        #     inv_images_thresholded_normed[i, :] = inv_image
            inv_image = inv_images[:, i] 
            images_retrofit[i, :] = transfert_matrix.dot(inv_image)
        #put back the inversion in the times, nodes order
        inv_images = inv_images.T 
        inv_normed = inv_normed.T
        inv_images_thresholded = inv_images_thresholded.T
        inv_images_thresholded_normed = inv_images_thresholded_normed.T
    elif inversion_method == 'Mfr':
        inversion_class = Mfr()
        simple_base = csr_matrix(np.eye( transfert_matrix.shape[1] ))
        inversion_class.decompose(transfert_matrix, simple_base, solver_kw= inversion_parameter)
        inv_images = inversion_class(images, simple_base)

        images_retrofit = transfert_matrix.dot(inv_image.T)
    elif inversion_method == 'nnls':
        try:
            alpha = inversion_parameter["alpha"]
        except:
            alpha = 0.01

        inv_images = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_normed = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_images_thresholded = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_images_thresholded_normed = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        images_retrofit = np.zeros_like(images)
        for i in range(images.shape[0]):
            image = images[i, :]
            inv_image, norms = invert_regularised_nnls(transfert_matrix.toarray(), image, alpha=alpha, tikhonov_matrix=None)
            inv_images[i, :] = inv_image
            inv_normed[i, :] = inv_image
            inv_images_thresholded[i, :] = inv_image
            inv_images_thresholded_normed[i, :] = inv_image
            images_retrofit[i, :] = transfert_matrix.dot(inv_image)  

    elif inversion_method == 'Mfr_Cherab':
        from cherab.inversion import Mfr as Mfr_Cherab

        derivative_matrix = get_derivative_matrix(inversion_method, mask = mask_copy)
        inv_images = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_normed = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_images_thresholded = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_images_thresholded_normed = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        images_retrofit = np.zeros_like(images)
        for i in range(images.shape[0]):
            image = images[i, :]
            mfr = Mfr_Cherab(transfert_matrix, derivative_matrix, data = image)
            inv_image, norms = mfr.solve()
            inv_images[i, :] = inv_image
            inv_normed[i, :] = inv_image
            inv_images_thresholded[i, :] = inv_image
            inv_images_thresholded_normed[i, :] = inv_image
            images_retrofit[i, :] = transfert_matrix.dot(inv_image)

    elif inversion_method == 'SART':
        from cherab.tools.inversions.sart import invert_sart

        transfert_matrix = np.array(transfert_matrix)
        inv_images = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_normed = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_images_thresholded = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_images_thresholded_normed = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        images_retrofit = np.zeros_like(images)
        for i in range(images.shape[0]):
            image = images[i, :]
            try:
                inv_image, conv = invert_sart(transfert_matrix, np.squeeze(images[i, :]), max_iterations=100)
            except:
                inv_image = np.zeros(transfert_matrix.shape[1])
            # pdb.set_trace()
            inv_images[i, :] = inv_image
            inv_normed[i, :] = inv_image
            inv_images_thresholded[i, :] = inv_image
            inv_images_thresholded_normed[i, :] = inv_image
            images_retrofit[i, :] = transfert_matrix.dot(inv_image)  


    elif inversion_method == 'OPENSART':
        from cherab.tools.inversions.opencl.sart_opencl import SartOpencl

        import pyopencl as cl
        platform = cl.get_platforms()[0]
        device = platform.get_devices()[0]

        invert_sart = SartOpencl(transfert_matrix, device = device)
        max_iterations = inversion_parameter.get('max_iterations')
        max_iterations = max_iterations or 250
        beta_laplace = inversion_parameter.get('beta_laplace')
        beta_laplace = beta_laplace or 0.01
        inv_images = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_normed = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_images_thresholded = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_images_thresholded_normed = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        images_retrofit = np.zeros_like(images)
        for i in range(images.shape[0]):
            image = images[i, :]
        #     if i>0:
        #         inv_image, residual = invert_sart(image, initial_guess=inv_images[i-1, :])
        #     else:
        #         inv_image, residual = invert_sart(image)
            inv_image, residual = invert_sart(np.squeeze(image), beta_laplace = beta_laplace, max_iterations=max_iterations, conv_tol=0.0001)
            inv_images[i, :] = inv_image
            inv_normed[i, :] = inv_image
            inv_images_thresholded[i, :] = inv_image
            inv_images_thresholded_normed[i, :] = inv_image
            images_retrofit[i, :] = transfert_matrix.dot(inv_image)  
                                          
    else:
        raise Exception('unrecognised inversion method')
    # raise RuntimeError(f"Failed to deserialize input: {inv_image}")

    return inv_images, inv_normed, inv_images_thresholded, inv_images_thresholded_normed, images_retrofit, transfert_matrix
        


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
      
def inverse_vid(transfert_matrix, mask_pixel,mask_noeud, pixels, noeuds, vid, R_noeud, Z_noeud, inversion_method, inversion_parameter, folder_inverse_matrix,dict_vid, derivative_matrix = None):
    """
    main function for inverting video

    Inputs :
    transfert_matrix : csr_matrix
    geometry matrix, in the #pixels, #nodes order
    mask_pixel : 2D array 
    mask for pixels with vision
    mask_noeud : 2D array 
    mask for visible nodes
    vid : 2D array
    video data, in the #times, #pixels order (pixels grid has been previously flattened)
    R_noeud, Z_noeud : 1D arrays, coordinates of the emissivity grid (m)
    inversion_method : string
    name of the inversion method used
    folder_inverse_matrix : string of a folder
    folder where to save or load the inverse matrix
    inversion_parameter : dictionnary, optionnal
    dictionnary to add optionnal parameters for inversion
    dict_vid : dictionnary, optionnal
    dictionnary to add instructions on operations annex to the inversion (filtering, denoising, etc...)

    Outputs : 
    Same as inversion_and_thresholding
    
    """
    
    
    
    #initisalisation
    # images = np.reshape(vid, (vid.shape[0], vid.shape[1]*vid.shape[2]))
    # images = images[:, mask_pixel]

    #prep transfert matrix for inversion

    import time
    start = time.time()
    transfert_matrix, pixels, noeuds, mask_pixel, mask_noeud = prep_inversion(transfert_matrix, mask_pixel, mask_noeud, pixels, noeuds, inversion_parameter, R_noeud, Z_noeud)
    end = time.time()
    elapsed = end-start
    print(f"Preparation transfert_matrix : {elapsed:.3f} seconds")

    images = vid[:, mask_pixel]
    inversion_results, inversion_results_normed, inversion_results_thresholded,inversion_results_thresholded_normed, images_retrofit, transfert_matrix = inversion_and_thresolding(images, 
                                                                                                                                                               transfert_matrix, 
                                                                                                                                                               inversion_method,
                                                                                                                                                               folder_inverse_matrix,
                                                                                                                                                               dict_vid,
                                                                                                                                                               inversion_parameter=inversion_parameter)

	
    return inversion_results, inversion_results_normed, inversion_results_thresholded,inversion_results_thresholded_normed, images_retrofit, mask_noeud, mask_pixel, transfert_matrix




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




       

def prep_inversion(transfert_matrix, mask_pixel, mask_noeud, pixels, noeuds, inversion_parameter, R_noeud, Z_noeud):
    
    if 'min_visibility_node' in inversion_parameter.keys():
        
        min_visibility_node = inversion_parameter.get('min_visibility_node')
        sum_over_pix = np.sum(transfert_matrix, 0)
        relevant_nodes = np.squeeze(np.array(sum_over_pix<min_visibility_node))
        transfert_matrix[:, relevant_nodes] = 0
        transfert_matrix, pixels, noeuds, mask_pixel, mask_noeud = reindex_transfert_matrix(transfert_matrix, pixels, noeuds, mask_pixel, mask_noeud)
    if 'node_min_value' in inversion_parameter.keys():
        node_min_value = inversion_parameter.get('node_min_value')
        transfert_matrix[transfert_matrix<node_min_value] = 0
        transfert_matrix, pixels, noeuds, mask_pixel, mask_noeud = reindex_transfert_matrix(transfert_matrix, pixels, noeuds, mask_pixel, mask_noeud)


        

    return transfert_matrix, pixels, noeuds, mask_pixel, mask_noeud


def reindex_transfert_matrix(transfert_matrix, pixels, noeuds, mask_pixel, mask_noeud):
    pixel_shape = mask_pixel.shape
    visible_pixel  = np.where(np.sum(transfert_matrix, 1))[0] #sum over nodes
    visible_node = np.where(np.sum(transfert_matrix, 0))[0] #sum over pixels
    pixels  = np.squeeze(pixels)[visible_pixel] 
    noeuds  = np.squeeze(noeuds)[visible_node] 
    #save results
    mask_pixel = np.zeros(pixel_shape, dtype = bool).flatten()
    mask_pixel[pixels] = True
    mask_pixel = mask_pixel.reshape(pixel_shape)

    
    mask_noeud[:] = False
    try:
        rows_noeud, indphi, cols_noeud = np.unravel_index(noeuds, mask_noeud.shape)

        mask_noeud[rows_noeud,indphi, cols_noeud] = True
    except:
        rows_noeud, cols_noeud = np.unravel_index(noeuds, mask_noeud.shape)

        mask_noeud[rows_noeud, cols_noeud] = True

    

    transfert_matrix = transfert_matrix[visible_pixel,:][:, visible_node]

    return transfert_matrix, pixels, noeuds, mask_pixel, mask_noeud 


def reshape_transfert_matrix(transfert_matrix, pixels, noeuds, mask_pixel, mask_noeud, mask_inversion):
        mask_pixel_out = mask_inversion*(mask_pixel)
        mask_pixel_out = mask_pixel_out>0
        visible_pixel = mask_pixel_out.flatten()[pixels]
        visible_pixel = visible_pixel>0

        pixels_out = pixels[visible_pixel]

        transfert_matrix_out = transfert_matrix[visible_pixel, :]
        sum_transfert_matrix_out = np.sum(transfert_matrix_out, 0)
        visible_nodes = np.where(sum_transfert_matrix_out)[1]

        noeuds_out = noeuds[visible_nodes]
        transfert_matrix_out = transfert_matrix_out[:, visible_nodes]
        noeuds_mask = np.where(mask_noeud.flatten())[0]
        mask_noeud_out =mask_noeud.flatten()
        mask_noeud_out[:] = 0
        mask_noeud_out[noeuds_mask[visible_nodes]] = 1
        mask_noeud_out = mask_noeud_out.reshape(mask_noeud.shape)
        return transfert_matrix_out, pixels_out, noeuds_out, mask_pixel_out, mask_noeud_out



def inverse_vid_from_class(Transfert_Matrix, Inversion_results, ParamsMachine, ParamsGrid, ParamsVid):
    import time
    start = time.time()

    #prep transfert matrix for inversion
    transfert_matrix, pixels, noeuds, mask_pixel, mask_noeud = prep_inversion(Transfert_Matrix.transfert_matrix, Transfert_Matrix.mask_pixel, Transfert_Matrix.mask_noeud, Transfert_Matrix.pixels, Transfert_Matrix.noeuds, ParamsVid.inversion_parameter, Transfert_Matrix.R_noeud, Transfert_Matrix.Z_noeud)
    end = time.time()
    elapsed = end-start
    print(f"Preparation transfert_matrix : {elapsed:.3f} seconds")


    images = Inversion_results.vid[:, mask_pixel]
    inversion_results, inversion_results_normed, inversion_results_thresholded,inversion_results_thresholded_normed, images_retrofit, transfert_matrix = inversion_and_thresolding(images, 
                                                                                                                                                               transfert_matrix,                                                                                                                                                             
                                                                                                                                                               ParamsVid.inversion_method,
                                                                                                                                                               Inversion_results.path_inverse_matrix,
                                                                                                                                                               ParamsVid.dict_vid,
                                                                                                                                                               inversion_parameter=ParamsVid.inversion_parameter)
    Inversion_results.inversion_results = inversion_results
    Inversion_results.images_retrofit = images_retrofit
    Inversion_results.inversion_results_thresholded = inversion_results_thresholded
    return Inversion_results


def denoising(Inversion_results):
    from tomotok.core.inversions import Bob, SparseBob, CholmodMfr, Mfr
    if Inversion_results.ParamsVid.inversion_method == 'Bob':
        



        inversion_class = Bob()
        simple_base = csr_matrix(np.eye( Inversion_results.transfert_matrix.shape[1] ))
        transfert_matrix = csr_matrix(Inversion_results.transfert_matrix)
        rcond = Inversion_results.ParamsVid.inversion_parameter.get('rcond')
        solver_dict = {'rcond' : rcond}
        c = Inversion_results.ParamsVid.c
        c = c or 0

        try:
            inversion_class.load_decomposition(Inversion_results.path_inverse_matrix)
            print('successfully loaded inverse matrix')
        except:
            inversion_class.decompose(transfert_matrix, simple_base, solver_kw = solver_dict)
            inversion_class._normalise_wo_mat()
            os.makedirs(os.path.dirname(Inversion_results.path_inverse_matrix), exist_ok = True)
            inversion_class.save_decomposition(Inversion_results.path_inverse_matrix)
        images = Inversion_results.vid[:, Inversion_results.mask_pixel]

        inv_images_thresholded = np.squeeze(inversion_class.thresholding(images.T, c = c))
        return inv_images_thresholded.T

    elif Inversion_results.ParamsVid.inversion_method == 'SparseBob':
       

        inversion_class = SparseBob()
        simple_base = csr_matrix(np.eye( Inversion_results.transfert_matrix.shape[1] ))
        transfert_matrix = csr_matrix(Inversion_results.transfert_matrix)
        
        c = Inversion_results.ParamsVid.c
        c = c or 0

        try:
            inversion_class.load_decomposition(Inversion_results.path_inverse_matrix)
            print('successfully loaded inverse matrix')
        except:
            inversion_class.decompose(transfert_matrix, simple_base)
            inversion_class._normalise_wo_mat()
            os.makedirs(os.path.dirname(Inversion_results.path_inverse_matrix), exist_ok = True)
            inversion_class.save_decomposition(Inversion_results.path_inverse_matrix)
        images = Inversion_results.vid[:, Inversion_results.mask_pixel]


        inv_images_thresholded = np.zeros((transfert_matrix.shape[1], images.shape[0]))
        for i in range(images.shape[0]):
            image = images.T[:, i]
            inv_images_thresholded[:, i] = np.squeeze(inversion_class.thresholding(image[:, np.newaxis], c = c))

        return inv_images_thresholded.T
    

def validate_denoising_method(inversion_method):
    denoising_method = ['Bob', "SparseBob"]
    if inversion_method in denoising_method:
        return True
    else:
        return False


def inversion(images, transfert_matrix, inversion_method, folder_inverse_matrix, dict_vid= {}, inversion_parameter = {"rcond" : -1}):
    """
    Inputs :
    images : 2D array
    video data, in the #times, #pixels order (pixels grid has been previously flattened)
    transfert_matrix : csr_matrix
    geometry matrix, in the #pixels, #nodes order
    inversion_method : string
    name of the inversion method used
    inversion_parameter : dictionnary, optionnal
    dictionnary to add optionnal parameters for inversion
    dict_vid : dictionnary, optionnal
    dictionnary to add instructions on operations annex to the inversion (filtering, denoising, etc...)

    
    Outputs :

    inv_images : 2D array
    video inversed, in the #times, #nodes order 
    inv_normed : 2D array
    video inversed and normalized, in the #times, #nodes order 
    inv_images_thresholded :2D array
    video inversed and denoized, in the #times, #nodes order  
    inv_images_thresholded_normed : 2D array
    video inversed and normalized and denoized, in the #times, #nodes order 
    images_retrofit: 2D array
    video reconstructed from nodes profile, in the #times, #pixels order 
    transfert_matrix : csr_matrix 
    geometry matrix, in the #pixels, #nodes order
    """
    
    
    
    
    from cherab.tools.inversions import invert_regularised_nnls
    # images = images[:, None] #add a dimension for the computation to procede correctly
    # transfert_matrix = transfert_matrix.todense()
    # transfert_matrix[:, np.sum(transfert_matrix, 0)<1e-3] = 0
    # transfert_matrix = csr_matrix(transfert_matrix)


    if inversion_method == 'Bob':
        from tomotok.core.inversions import Bob, SparseBob, CholmodMfr, Mfr
        from tomotok.core.derivative import compute_aniso_dmats
        from tomotok.core.geometry import RegularGrid
 
        inversion_class = Bob()
        simple_base = csr_matrix(np.eye( transfert_matrix.shape[1] ))
        transfert_matrix = csr_matrix(transfert_matrix)
        rcond = inversion_parameter.get('rcond')
        solver_dict = {'rcond' : rcond}


        path_inverse_matrix = folder_inverse_matrix
        path_norm_matrix = folder_inverse_matrix + 'norm'
        try:
            inversion_class.load_decomposition(path_inverse_matrix)
            print('successfully loaded inverse matrix')
        except:
            inversion_class.decompose(transfert_matrix, simple_base, solver_kw = solver_dict)
            inversion_class._normalise_wo_mat()
        os.makedirs(os.path.dirname(path_inverse_matrix), exist_ok = True)
        inversion_class.save_decomposition(path_inverse_matrix)

        inv_images = inversion_class(images.T) #put the image in the #pixels, times dimension order
        images_retrofit = np.zeros_like(images)
        for i in range(images.shape[0]):
            inv_image = inv_images[:, i] 
            images_retrofit[i, :] = transfert_matrix.dot(inv_image)
        #put back the inversion in the times, nodes order
        inv_images = inv_images.T 
    elif inversion_method == 'SparseBob':
        from tomotok.core.inversions import Bob, SparseBob, CholmodMfr, Mfr
        from tomotok.core.derivative import compute_aniso_dmats
        from tomotok.core.geometry import RegularGrid
        inversion_class = SparseBob()
        simple_base = csr_matrix(np.eye( transfert_matrix.shape[1] ))
        transfert_matrix = csr_matrix(transfert_matrix)

        path_inverse_matrix = folder_inverse_matrix
        path_norm_matrix = folder_inverse_matrix + 'norm'
        try:
            inversion_class.load_decomposition(path_inverse_matrix)
            print('successfully loaded inverse matrix')
        except:
            inversion_class.decompose(transfert_matrix, simple_base)
            inversion_class._normalise_wo_mat()
            os.makedirs(os.path.dirname(path_inverse_matrix), exist_ok = True)
            inversion_class.save_decomposition(path_inverse_matrix)

        inv_images = inversion_class(images.T) #put the image in the #pixels, times dimension order
        images_retrofit = np.zeros_like(images)
        for i in range(images.shape[0]):
            inv_image = inv_images[:, i] 
            images_retrofit[i, :] = transfert_matrix.dot(inv_image)
        #put back the inversion in the times, nodes order
        inv_images = inv_images.T 
    elif inversion_method == 'Mfr':
        inversion_class = Mfr()
        simple_base = csr_matrix(np.eye( transfert_matrix.shape[1] ))
        inversion_class.decompose(transfert_matrix, simple_base, solver_kw= inversion_parameter)
        inv_images = inversion_class(images, simple_base)
        images_retrofit = transfert_matrix.dot(inv_image.T)
    elif inversion_method == 'nnls':
        try:
            alpha = inversion_parameter["alpha"]
        except:
            alpha = 0.01

        inv_images = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_normed = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_images_thresholded = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_images_thresholded_normed = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        images_retrofit = np.zeros_like(images)
        for i in range(images.shape[0]):
            image = images[i, :]
            inv_image, norms = invert_regularised_nnls(transfert_matrix.toarray(), image, alpha=alpha, tikhonov_matrix=None)
            inv_images[i, :] = inv_image
            inv_normed[i, :] = inv_image
            inv_images_thresholded[i, :] = inv_image
            inv_images_thresholded_normed[i, :] = inv_image
            images_retrofit[i, :] = transfert_matrix.dot(inv_image)  

    elif inversion_method == 'SART':
        from cherab.tools.inversions.sart import invert_sart

        transfert_matrix = np.array(transfert_matrix)
        inv_images = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_normed = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_images_thresholded = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        inv_images_thresholded_normed = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        images_retrofit = np.zeros_like(images)
        for i in range(images.shape[0]):
            image = images[i, :]
            try:
                inv_image, conv = invert_sart(transfert_matrix, np.squeeze(images[i, :]), max_iterations=100)
            except:
                inv_image = np.zeros(transfert_matrix.shape[1])
            # pdb.set_trace()
            inv_images[i, :] = inv_image
            inv_normed[i, :] = inv_image
            inv_images_thresholded[i, :] = inv_image
            inv_images_thresholded_normed[i, :] = inv_image
            images_retrofit[i, :] = transfert_matrix.dot(inv_image)  


    elif inversion_method == 'OPENSART':
        from cherab.tools.inversions.opencl.sart_opencl import SartOpencl

        import pyopencl as cl
        platform = cl.get_platforms()[0]
        device = platform.get_devices()[0]

        invert_sart = SartOpencl(transfert_matrix, device = device)
        max_iterations = inversion_parameter.get('max_iterations')
        max_iterations = max_iterations or 250
        beta_laplace = inversion_parameter.get('beta_laplace')
        beta_laplace = beta_laplace or 0.01
        relaxation =  inversion_parameter.get('relaxation')
        relaxation = relaxation or 1

        inv_images = np.zeros((images.shape[0], transfert_matrix.shape[1]))
        images_retrofit = np.zeros_like(images)
        for i in range(images.shape[0]):
            image = images[i, :]
        #     if i>0:
        #         inv_image, residual = invert_sart(image, initial_guess=inv_images[i-1, :])
        #     else:
        #         inv_image, residual = invert_sart(image)
            inv_image, residual = invert_sart(np.squeeze(image), beta_laplace = beta_laplace, max_iterations=max_iterations, relaxation = relaxation, conv_tol=0.0001)
            inv_images[i, :] = inv_image
            images_retrofit[i, :] = transfert_matrix.dot(inv_image)  
        invert_sart.clean()
    else:
        raise Exception('unrecognised inversion method')
    # raise RuntimeError(f"Failed to deserialize input: {inv_image}")

    return inv_images, images_retrofit
        




def prep_inversion_dataset(rt_ds, ParamsInversion):
    transfert_matrix = rt_ds.transfert_matrix.to_numpy()
    pixel = rt_ds.pixel.to_numpy()
    node =  rt_ds.node.to_numpy()
    mask_node = np.zeros(rt_ds.node_shape, dtype = bool)
    mask_node[rt_ds.rows_node.to_numpy(), rt_ds.cols_node.to_numpy()] = True
    mask_pixel = np.zeros(rt_ds.pixel_shape, dtype = bool)
    mask_pixel[rt_ds.rows_pixel.to_numpy(), rt_ds.cols_pixel.to_numpy()] = True
    inversion_parameter = ParamsInversion.inversion_parameter
    if 'min_visibility_node' in inversion_parameter.keys():
        
        min_visibility_node = inversion_parameter.get('min_visibility_node')
        sum_over_pix = np.sum(transfert_matrix, 0)
        relevant_nodes = np.squeeze(np.array(sum_over_pix<min_visibility_node))
        transfert_matrix[:, relevant_nodes] = 0
        transfert_matrix, pixel, node, mask_pixel, mask_node = reindex_transfert_matrix(transfert_matrix, pixel, node, mask_pixel, mask_node)
    if 'node_min_value' in inversion_parameter.keys():
        node_min_value = inversion_parameter.get('node_min_value')
        transfert_matrix[transfert_matrix<node_min_value] = 0
        transfert_matrix, pixel, node, mask_pixel, mask_node = reindex_transfert_matrix(transfert_matrix, pixel, node, mask_pixel, mask_node)
    if 'remove_inner_components' in inversion_parameter.keys():
        new_wall = inversion_parameter.get('remove_inner_components')
        cell_r = np.array(rt_ds.cell_r)
        cell_z = np.array(rt_ds.cell_z)
        if isinstance(new_wall, list):
            for wall_element in new_wall:
                RZwall = np.load(wall_element)
                
                mask_node = utility_functions.update_mask(cell_r, cell_z, mask_node, RZwall, remove_side = 'inner')
                transfert_matrix, pixel, node, mask_pixel, mask_node = reindex_transfert_matrix(transfert_matrix, pixel, node, mask_pixel, mask_node)
        elif isinstance(new_wall, str):
            RZwall = np.load(wall_element)
            mask_node = utility_functions.update_mask(cell_r, cell_z, mask_node, RZwall, remove_side = 'inner')
            transfert_matrix, pixel, node, mask_pixel, mask_node = reindex_transfert_matrix(transfert_matrix, pixel, node, mask_pixel, mask_node)
    if 'new_wall' in inversion_parameter.keys():
        new_wall = inversion_parameter.get('new_wall')
        cell_r = np.array(rt_ds.cell_r)
        cell_z = np.array(rt_ds.cell_z)
        if isinstance(new_wall, list):
            for wall_element in new_wall:
                RZwall = np.load(wall_element)
                mask_node = utility_functions.update_mask(cell_r, cell_z, mask_node, RZwall, remove_side = 'outer')
                transfert_matrix, pixel, node, mask_pixel, mask_node = reindex_transfert_matrix_from_mask_node(transfert_matrix, pixel, node, mask_pixel, mask_node)
        elif isinstance(new_wall, str):
            RZwall = np.load(new_wall)
            mask_node = utility_functions.update_mask(cell_r, cell_z, mask_node, RZwall, remove_side = 'outer')
            transfert_matrix, pixel, node, mask_pixel, mask_node = reindex_transfert_matrix_from_mask_node(transfert_matrix, pixel, node, mask_pixel, mask_node)
    rows_node, cols_node = np.unravel_index(node, mask_node.shape)
    rows_pixel, cols_pixel = np.unravel_index(pixel, mask_pixel.shape)

    return transfert_matrix, mask_node, mask_pixel, node, pixel, rows_node, cols_node, rows_pixel, cols_pixel




def denoising_dataset(inv_ds, images, ParamsDenoising):
    inversion_method = inv_ds.attrs["ParamsInversion"]["inversion_method"]
    inversion_parameter = inv_ds.attrs["ParamsInversion"]["inversion_parameter"]
    folder_inverse_matrix = inv_ds.attrs["folder_inverse_matrix"]
    from tomotok.core.inversions import Bob, SparseBob, CholmodMfr, Mfr
    if inversion_method == 'Bob':
        



        inversion_class = Bob()
        rcond = inversion_parameter.get('rcond')
        solver_dict = {'rcond' : rcond}
        c = ParamsDenoising.c
        c = c or 0
        print(f"denoising parameter :  {c}")
        inversion_class.load_decomposition(folder_inverse_matrix)
           
        inv_images_thresholded = np.zeros((images.shape[0], inversion_class.dec_mat.shape[1]))
        inv_images_thresholded = inv_images_thresholded.T
        for i in range(images.shape[0]):
            image = images.T[:, i]
            inv_images_thresholded[:, i] = np.squeeze(inversion_class.thresholding(image[:, np.newaxis], c = c))

        return inv_images_thresholded.T, np.squeeze(inversion_class.norms)

    elif inversion_method == 'SparseBob':
       

        inversion_class = SparseBob()
        
        c = ParamsDenoising.c
        c = c or 0
        print(f"denoising parameter :  {c}")
        inversion_class.load_decomposition(folder_inverse_matrix)
        print('successfully loaded inverse matrix')
            

        inv_images_thresholded = np.zeros((images.shape[0], inversion_class.dec_mat.shape[1]))
        inv_images_thresholded = inv_images_thresholded.T
        for i in range(images.shape[0]):
            image = images.T[:, i]
            inv_images_thresholded[:, i] = np.squeeze(inversion_class.thresholding(image[:, np.newaxis], c = c))

        return inv_images_thresholded.T, np.squeeze(inversion_class.norms)
    


def reindex_transfert_matrix_from_mask_node(transfert_matrix, pixel, node, mask_pixel, mask_node):
    flat_indices = np.zeros(mask_node.size, dtype='bool')
    new_node = np.flatnonzero(mask_node)

    flat_indices[new_node] =True
    reduced_flat_indices = flat_indices[node]
    transfert_matrix = transfert_matrix[:, reduced_flat_indices]
    visible_pixel  = np.where(np.sum(transfert_matrix, 1))[0] #sum over nodes
    if len(visible_pixel) == len(pixel):
        print("no new blind pixels")
    else:
        transfert_matrix = transfert_matrix[visible_pixel,:][:, reduced_flat_indices]
        pixel = pixel[visible_pixel]
        mask_pixel[:] = False
        mask_pixel.ravel()[pixel] = True

    return transfert_matrix, pixel, new_node, mask_pixel, mask_node


