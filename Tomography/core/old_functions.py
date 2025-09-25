



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



def extract_name_variables(input_str):
    PARTS =  input_str.split("/") 
    nshot = int(PARTS[0])
    parameters = PARTS[1]

    parts = parameters.split("_")  # Split the string by "_"

    i = 0
    j = 0
    regular_names = ['mask', 'calibration', 'wall', 'material', 'dz', 'dr', 'decimation', 'cropping', 'reflexion_dict']
    new_names = []
    while i < len(parts):
        name_part = ''
        if parts[i] == regular_names[j]:  # If it's a regular name, store its value
            j = j+1
            i = i+1
            while parts[i] != regular_names[j]:
                name_part = name_part+parts[i]
                i = i+1
            new_names[j-1] = name_part
    name_mask = new_names[0]
    name_calib = new_names[1]
    type_wall = new_names[2]
    name_material = new_names[3]
    dz_grid = int(new_names[4])
    dr_grid = int(new_names[5])
    decimation = int(new_names[6])
    dr_grid = int(new_names[5])
    param_fit = new_names[6]
    dict_transfert_matrix = new_names[7]




    name_parameters = ( 
        '_mask_' + name_mask + 
        '_calibration_' + name_calib+ 
        '_wall_' + type_wall + 
        '_material_' + name_material + 
        '_dz_' + str(int(dz_grid*1e3)) + 
        '_dr_' + str(int(dr_grid*1e3)) + 
        '_decimation_' + str(decimation) +
        '_cropping_' + str(param_fit) + 
        '_reflexion_dict_' + str(dict_transfert_matrix)
    )

    return name_parameters


def get_name_machine(machine):
    # get the name of the machine and assure it is the correct syntax in lowercase
    if machine.lower()== 'compass':
        machine = 'COMPASS'
    elif machine.lower()== 'west':
        machine = 'WEST'
    else:
        raise Exception('unrecognised machine')
    return machine



def compare_transfert_matrix(nshot = None, path_reflexion = None, path_absorb = None, path_resultat_inversion = None, path_vid = None):
    if not path_reflexion:
        path_reflexion = utility_functions.get_file('get npz file of reflexion data', path_root= '/Home/LF276573/Documents/Python/CHERAB/transfert_matrix/west/',full_path = 1)
    path_reflexion, ext = os.path.splitext(path_reflexion)
    if not path_absorb:
        path_absorb = utility_functions.get_file('get npz file of absorb data', path_root='/Home/LF276573/Documents/Python/CHERAB/transfert_matrix/west/',full_path = 1)
    path_absorb, ext = os.path.splitext(path_absorb)
    if not path_resultat_inversion:
        path_resultat_inversion = utility_functions.get_file('get npz file of inversion results', path_root='/Home/LF276573/Documents/Python/CHERAB/resultat_inversion/west/',full_path = 1)
    path_resultat_inversion, ext = os.path.splitext(path_resultat_inversion)
    if not path_vid:
        path_vid = utility_functions.get_file('get video file', path_root = '/Home/LF276573/Documents/Python/CHERAB/videos/west/', full_path = 1)

    load_refl = np.load(path_reflexion + '.npz', allow_pickle=True)
    data_refl = {key: load_refl[key] for key in load_refl}
    data_refl['transfert_matrix'] = load_npz(path_reflexion + '_transfert_matrix.npz')
    load_absorb = np.load(path_absorb + '.npz', allow_pickle=True)
    data_absorb = {key: load_absorb[key] for key in load_absorb}
    data_absorb['transfert_matrix'] = load_npz(path_absorb + '_transfert_matrix.npz')

    inversion_results, mask_inversion, mask_noeud_inversion, frame_input = load_results_inversion(os.path.dirname(path_resultat_inversion), os.path.basename(path_resultat_inversion))
    frame_input = [frame_input[0], frame_input[1]]
    path_vid, ext = os.path.splitext(path_vid)
    if ext == '.png':
        vid, len_vid,image_dim_y,image_dim_x, fps, frame_input = get_img(path_vid, nshot)
        image = vid[0, :,:]
        image = np.flip(image, 1)
    else: #load videos
        vid, len_vid,image_dim_y,image_dim_x, fps, frame_input, name_time, t0 = get_vid(None, frame_input, path_vid = path_vid, nshot = nshot)
        image = vid[0, :,:]
        image = np.flip(image, 0)
        image = image.T
    inversion = inversion_results[0, :]

    inversion_full = reconstruct_2D_image(inversion, mask_noeud_inversion)
    inversion_refl, inversion_absorb, image_retrofit_full_absorb, image_retrofit_full_refl, diff = recreate_retrofit(image, inversion_full, data_refl, data_absorb)
    return image, image_retrofit_full_absorb, image_retrofit_full_refl, diff, inversion_full, nshot, path_absorb, path_reflexion, path_resultat_inversion, path_vid, data_refl, data_absorb

def compare_already_loaded_transfert_matrix(nshot, data_refl, data_absorb, path_resultat_inversion = None, path_vid = None):
    
    if not path_resultat_inversion:
        path_resultat_inversion = utility_functions.get_file('get npz file of inversion results', path_root='/Home/LF276573/Documents/Python/CHERAB/resultat_inversion/west/',full_path = 1)
    path_resultat_inversion, ext = os.path.splitext(path_resultat_inversion)
    if not path_vid:
        path_vid = utility_functions.get_file('get video file', path_root = '/Home/LF276573/Documents/Python/CHERAB/videos/west/', full_path = 1)

    

    inversion_results, mask_inversion, mask_noeud_inversion, frame_input = load_results_inversion(os.path.dirname(path_resultat_inversion), os.path.basename(path_resultat_inversion))
    frame_input = [frame_input[0], frame_input[1]]
    path_vid, ext = os.path.splitext(path_vid)
    if ext == '.png':
        vid, len_vid,image_dim_y,image_dim_x, fps, frame_input = get_img(path_vid, nshot)
        image = vid[0, :,:]
        image = np.flip(image, 1)
    else: #load videos
        vid, len_vid,image_dim_y,image_dim_x, fps, frame_input, name_time, t0 = get_vid(None, frame_input, path_vid = path_vid, nshot = nshot)
        image = vid[0, :,:]
        image = np.flip(image, 0)
        image = image.T
    inversion = inversion_results[0, :]

    inversion_full = reconstruct_2D_image(inversion, mask_noeud_inversion)
    inversion_refl, inversion_absorb, image_retrofit_full_absorb, image_retrofit_full_refl, diff = recreate_retrofit(image, inversion_full, data_refl, data_absorb)
    return image, image_retrofit_full_absorb, image_retrofit_full_refl, diff, inversion_full, nshot, path_resultat_inversion, path_vid, data_refl, data_absorb


# def verify_integrity(realcam, mask_pixel):
    for i in range(mask_pixel.shape[0]):
        for j in range(mask_pixel.shape[1]):
            if mask_pixel[i, j] == 1:
                if np.isnan(realcam.pixel_directions[i, j].x) or np.isnan(realcam.pixel_origins[i, j].x):
                    return False
            elif mask_pixel[i, j] == 0:
                if not np.isnan(realcam.pixel_directions[i, j].x) or not np.isnan(realcam.pixel_origins[i, j].x):
                    return False
            else:
                raise('mask data cannot be used to mask pixels of camera. Please check datatype')

    return True



def get_name_parameters_inversion(inversion_parameter):
    name_inversion_parameters = ''
    for key in inversion_parameter.keys():
        name_inversion_parameters = name_inversion_parameters + '_' + key
        if type(inversion_parameter[key]) != str:
            name_inversion_parameters = name_inversion_parameters + '_' + str(inversion_parameter[key])
        else:
            name_inversion_parameters = name_inversion_parameters + '_' + inversion_parameter[key]
    if not name_inversion_parameters:
        name_inversion_parameters = 'None'
    return name_inversion_parameters


def compare_reflection_results_parameters(parameters = None):
    if not parameters:
        name_parameters = utility_functions.get_file()

    comparison = 1
    return comparison 


def compare_synth_image(image, image_refl, image_absorb):
    std = np.load('std_noise.npy')
    diff_refl = image-image_refl
    diff_absorb = image-image_absorb
    pixel_reflections = np.abs(diff_refl)>1.2*std
    pixel_absorb = np.abs(diff_absorb)>1.2*std
    diff_refl[np.invert(pixel_reflections)] = 0
    diff_absorb[np.invert(pixel_absorb)] = 0
    utility_functions.plot_comparaison_image(np.abs(diff_refl), np.abs(diff_absorb), vmax = 50)
    return diff_refl, diff_absorb


    

def recreate_retrofit(image, inversion_full, data_refl, data_absorb):

    inversion_refl = inversion_full[data_refl['mask_noeud']]
    inversion_absorb = inversion_full[data_absorb['mask_noeud']]
    
    # image_retrofit_absorb = transfert_matrix_absorption.dot(inversion)
    # image_retrofit_full_absorb = reconstruct_2D_image(image_retrofit_absorb, mask_inversion)
    image_retrofit_refl = data_refl['transfert_matrix'].dot(inversion_refl)
    image_retrofit_full_refl = reconstruct_2D_image(image_retrofit_refl, data_refl['mask_pixel'])
    image_retrofit_absorb = data_absorb['transfert_matrix'].dot(inversion_absorb)
    image_retrofit_full_absorb = reconstruct_2D_image(image_retrofit_absorb, data_absorb['mask_pixel'])
    norm = 'linear'
    cmap = 'PiYG'
    import matplotlib.colors as mcolors
    # print(np.min(image), np.min(image_retrofit_full_absorb), np.min(image_retrofit_full_refl))
    diff = image-image_retrofit_full_refl
    if norm == 'log':
        image = np.log2(image+1e-10)   
        image_retrofit_full_absorb = np.log2(image_retrofit_full_absorb+1e-10)
        image_retrofit_full_refl = np.log2(image_retrofit_full_refl+1e-10)
        diff = np.log2(np.abs(diff)+1e-10)

    # if norm == 'log':
    #     image = np.log2(image-min(-1, np.min(image)))   
    #     image_retrofit_full_absorb = np.log2(image_retrofit_full_absorb-min(-1, np.min(image_retrofit_full_absorb)))
    #     image_retrofit_full_refl = np.log2(image_retrofit_full_refl+1)
    vmin = image.min()
    # vmin = 12
    # vmax = image.max()
    vmax = 200
    
    
    #synthetic image
    plt.figure()
    plt.subplot(2,2,1)
    plt.imshow(image, cmap = cmap, vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.title('real image')

    #retro fit
    plt.subplot(2, 2,2)
    plt.imshow(image_retrofit_full_absorb, cmap = cmap, vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.title('Retro fit absorption')

    #retro fit
    plt.subplot(2, 2,3)
    plt.imshow(image_retrofit_full_refl, cmap = cmap, vmin = vmin, vmax = vmax)
    plt.colorbar()
    plt.title('Retro fit reflexion')
    #retro fit

    plt.subplot(2, 2,4)
    plt.imshow(diff, cmap = cmap)
    plt.colorbar()
    plt.title('difference retro fit refl and real image')
    plt.show(block = False)
    return inversion_refl, inversion_absorb, image_retrofit_full_absorb, image_retrofit_full_refl, diff


