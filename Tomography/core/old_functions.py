
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




def plot_transfert_matrix_and_synthetic(vmax = None):
    path_parameters =utility_functions.get_file(path_root  = '/Home/LF276573/Zone_Travail/Python/CHERAB/transfert_matrix/west/', full_path=0)
    # path_parameters =utility_functions.get_file(path_root  = '/Home/LF276573/Documents/Python/CHERAB/failure/', full_path=1)
    name_parameters, ext = os.path.splitext(path_parameters)

    path_transfert_matrix = name_parameters + '_transfert_matrix.npz'
    try:
        loaded_raytracing = np.load(path_parameters, allow_pickle = True)
        transfert_matrix = load_npz(path_transfert_matrix)
    except:
        loaded_raytracing = np.load(path_parameters, allow_pickle = True)
        folder_transfert_matrix = os.path.dirname(name_parameters)
        path_transfert_matrix = folder_transfert_matrix + '/transfert_matrix.npz'
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
    see_nodes = np.sum(transfert_matrix, 0)
    see_pix = np.sum(transfert_matrix, 1)
    see_nodes_full = reconstruct_2D_image(see_nodes, mask_noeud)
    see_pix_full = reconstruct_2D_image(see_pix, mask_pixel)

    utility_functions.plot_image(see_nodes_full, title=name_parameters, vmax=vmax)

    utility_functions.plot_image(see_pix_full, title=name_parameters, vmax=vmax)
    return transfert_matrix, see_nodes_full, see_pix_full, mask_noeud, mask_pixel


def FULL_MASK(mask,image_dim_y,image_dim_x,plot_image_reduction,path_remove_blind_pixels = 0):
    ## gives the indices of the reduced mask inside the full mask

    #
    # 1) Reduction of the frame size from the calibration frame size to the
    # experimental one
    # 2) Suppression of the blind or non-sense pixels
    #
    # Creation 2024-02-07
    
    
    ## 1) Reduction to the right frame size
    decalage_centre=0
    Nber_pixels_y,Nber_pixels_x =np.shape(mask)
    corner_min_y=int(image_dim_y/2-Nber_pixels_y/2-decalage_centre)
    corner_max_y=int((image_dim_y/2+Nber_pixels_y/2-decalage_centre))
    corner_min_x=int((image_dim_x/2-Nber_pixels_x/2))
    corner_max_x=int(image_dim_x/2+Nber_pixels_x/2)
    full_mask = np.zeros((image_dim_y, image_dim_x))
    
    
    ## 2) Pixels blind or non-sense
    try:
        f = loadmat(path_remove_blind_pixels)
        mask = f['mask']
    except:
        mask = np.load(path_remove_blind_pixels)

    #full_mask[corner_min_y:corner_max_y,corner_min_x: corner_max_x] = mask

    ## 3) Check how the image was reduced
    """
    if plot_image_reduction==1:
        full_image=np.zeros((Nber_pixels_y,Nber_pixels_x,3))
        indice_im_reduced_y, indice_im_reduced_x = np.meshgrid(im_reduced_y, im_reduced_x, indexing = 'ij')
        full_image[indice_im_reduced_y,indice_im_reduced_x]=los_direction_reduced_and_blind
        plt.figure()
        plt.imshow(full_image)
        plt.plot([corner_min_x, corner_min_x],[corner_min_y, corner_max_y],'w-')
        plt.plot([corner_min_x, corner_max_x],[corner_max_y, corner_max_y],'w-')
        plt.plot([corner_max_x, corner_max_x],[corner_min_y, corner_max_y],'w-')
        plt.plot([corner_min_x, corner_max_x],[corner_min_y, corner_min_y],'w-')
    
        plt.title('Check the image reduction')
        
        # Diplay points corresponding to LOS
        get_non_nan=np.isnan(los_direction_reduced_and_blind[:,:,1])
        [row,col]=np.where(get_non_nan==0)
    
        I1=min(row)+corner_min_y
        I2=max(row)+corner_min_y
        J1=min(col)+corner_min_x
        J2=max(col)+corner_min_x
    
        plt.plot(J1,I1,'yo')
        plt.plot(J1,I2,'yo')
        plt.plot(J2,I1,'yo')
        plt.plot(J2,I2,'yo')
        #scale_picture
        """
    return corner_min_y, corner_max_y,corner_min_x, corner_max_x