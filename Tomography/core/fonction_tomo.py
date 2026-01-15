import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
import time
import xarray as xr
from scipy.io import loadmat, savemat
from scipy.sparse import csr_matrix
import os
#custom imports
from . import utility_functions, result_inversion, inversion_module
from Tomography.ressources import components, paths
import importlib
importlib.reload(utility_functions)
importlib.reload(inversion_module)

#cherab imports
from cherab.tools.raytransfer import RayTransferPipeline2D, RayTransferCylinder
from cherab.tools.primitives.axisymmetric_mesh import axisymmetric_mesh_from_polygon
from cherab.tools.raytransfer import RoughIron, RoughTungsten, RoughSilver

#raysect imports
from raysect.optical import World, Node, translate, rotate, Point3D, rotate_z
from raysect.optical.observer import PinholeCamera, FullFrameSampler2D, RGBPipeline2D, VectorCamera,  RGBAdaptiveSampler2D
from raysect.primitive import import_stl, import_obj, Box, import_ply
from raysect.optical.material import UniformVolumeEmitter, AbsorbingSurface, Lambert
from raysect.optical.library.spectra.colours import *
colours = [yellow, orange, red_orange, red, purple, blue, light_blue, cyan, green]

import pdb




def compute_raytracing(ParamsMachine, ParamsGrid):

    world, RZwall = setup_raytracing_world(ParamsMachine, ParamsGrid)


    realcam, fit_shape= get_camera(ParamsMachine, ParamsGrid, world)

    transfert_matrix, mask_node, mask_pixel, node, pixel, rows_node, cols_node, rows_pixel, cols_pixel, cell_r, cell_z = get_transfert_matrix(realcam, 
                                                                                                                                              world, 
                                                                                                                                              ParamsMachine, 
                                                                                                                                              ParamsGrid, 
                                                                                                                                              RZwall)

    rt_ds = xr.Dataset(
        data_vars={
            "transfert_matrix": (
                ("pixel", "node"),
                transfert_matrix,
                {"units": "m"}
            ),
            
        },
        coords={
            "pixel": ("pixel", pixel),
            "node": ("node", node),
            "rows_pixel": ("pixel", rows_pixel),
            "cols_pixel": ("pixel", cols_pixel),
            "rows_node": ("node", rows_node),
            "cols_node": ("node", cols_node),
            
        },
        attrs={
            "image_shape": mask_pixel.shape,
            "node_shape": mask_node.shape,
            "mask_node" : mask_node,
            "mask_pixel" : mask_pixel,
            "mask_description": "Camera mask applied",
            "cell_r" : cell_r,
            "cell_z" : cell_z,
            "ParamsMachine":ParamsMachine.to_dict(),
            "ParamsGrid":ParamsGrid.to_dict(),
            "fit_shape" : fit_shape,
        }

    )

    return rt_ds


def compute_inversion(rt_ds, ParamsVid):
    ParamsMachine = result_inversion.ParamsMachine(**rt_ds.ParamsMachine)
    
    vid, len_vid,image_dim_y,image_dim_x, fps, frame_input, name_time, t_start, t0, t_inv = utility_functions.get_vid(ParamsVid)
    vid = np.swapaxes(vid, 1,2)
    # utility_functions.save_array_as_img(vid, main_folder_image + 'image_vid_mid_rotated.png')
    if ParamsMachine.machine == 'WEST':
        vid = np.swapaxes(vid, 1,2)
        vid = np.flip(vid, 1)
    if ParamsMachine.param_fit is not None:
        vid = fit_size_vid(vid, rt_ds.fit_shape, decimation = ParamsMachine.decimation)

    if ParamsMachine.decimation is not None and ParamsMachine.decimation != 1:
        vid = reduce_vid_precision(vid, decimation = ParamsMachine.decimation)

    transfert_matrix, mask_node, mask_pixel, node, pixel, rows_node, cols_node, rows_pixel, cols_pixel = inversion_module.prep_inversion_dataset(rt_ds, ParamsVid)
    folder_inverse_matrix = 'inversion_matrix/' + rt_ds.hash
    images = vid[:, mask_pixel]
    print("starting inversion")
    inv_images, images_retrofit = inversion_module.inversion(images, 
                                    transfert_matrix, 
                                    inversion_method = ParamsVid.inversion_method, 
                                    folder_inverse_matrix =folder_inverse_matrix,
                                    dict_vid= ParamsVid.dict_vid, 
                                    inversion_parameter = ParamsVid.inversion_parameter)
    inv_ds = xr.Dataset(
        data_vars={
            "inversion":(
                ("t_inv", "node"),
                inv_images,
                {"units":"m"}
            ),
            "image_retrofit":(
                ("t_inv",  "pixel"),
                images_retrofit,
                {"units":"bits"}
            ),
            "images": (
                ("t_inv", "pixel"),
                images,
                {"units": "bits"},
            ),
        },
        coords={
            "t_inv": ("t_inv", t_inv, {"units":'seconds'}),
            "pixel": ("pixel", pixel),
            "node": ("node", node),
            "row_pixel": ("pixel", rows_pixel),
            "col_pixel": ("pixel", cols_pixel),
            "row_node": ("node", rows_node),
            "col_node": ("node", cols_node),
            

        },
        attrs={
            "pixel_shape": mask_pixel.shape,
            "node_shape": mask_node.shape,
            "mask_description": "Camera mask applied",
            "cell_r" : rt_ds.cell_r,
            "cell_z" : rt_ds.cell_z,
            "node_image_orientation": "swapaxes, flip Y axis"
            "pixel_image_orientation": "swapaxes"
        }
    )
    return inv_ds

def compute_denoising(inv_ds, ParamsVid):
    
    return inv_ds


def get_transfert_matrix(realcam, world, ParamsMachine, ParamsGrid, RZwall):
    """
    return transfert_matrix"
    """


    def trace_field_line_toroidal(self, *coordinates, R: np.array = None, Z: np.array = None,
                            coord_type=None, in_first_wall=True, phi_end = None, phi_eval = None, **coords):
        """
        Return traced field lines starting from the given set of at least 2d coordinates.
    Allows to choose the toroidal angle span on which to integrate
    Outter field lines are limited by z planes given be outermost z coordinates of the first wall.
        :param coordinates:
        :param R:
        :param Z:
        :param coord_type:
        :param in_first_wall: if True the only inner part of field line is returned.
        :param phi_end : toroidal angle value at wich to stop the integration. 
        :param phi_eval : optional, allow to choose the values of the toroidal angle returned 
            along the integration of the magnetic field line. 
            Must be contained in the range [phi_start, phi_end], the minimum and maximum value of the toroidal angle.
            If none specified, allow the solver to choose these values. 
        :param coords:
        :return: list of coordinates 
        """
    
        import pleque.utils.field_line_tracers as flt
        from scipy.integrate import solve_ivp
        coords = self.coordinates(*coordinates, R=R, Z=Z, coord_type=coord_type, **coords)

        res = []

        # XXXNOW
        coords_rz = coords.as_array(dim=2)

        

        dphifunc = flt.dphi_tracer_factory(self.B_R, self.B_Z, self.B_tor)

        r_lims = [np.min(self.first_wall.R), np.max(self.first_wall.R)]
        z_lims = [np.min(self.first_wall.Z), np.max(self.first_wall.Z)]

        for i in np.arange(len(coords)):
            
            if 1: #i==0 or i==len(coords)-1 or i == np.int(len(coords)/2):
                y0 = coords_rz[i]
                if coords.dim == 2:
                    phi_start = 0
                else:
                    phi_start = phi_eval[0]
                
                if phi_end is None:
                    phi_end = phi_start 

                atol = 1e-6
                if self.is_xpoint_plasma:
                    xp = self._x_point
                    xp_dist = np.sqrt(np.sum((xp - y0) ** 2))
                    atol = np.minimum(xp_dist * 1e-3, atol)

                if self._verbose:
                    print('>>> tracing from: {:3f},{:3f},{:3f}'.format(y0[0], y0[1], phi_start))
                    print('>>> atol = {}'.format(atol))

                # todo: define somehow sufficient tolerances
                sol = solve_ivp(dphifunc,
                                (phi_start, phi_end),
                                y0,
                                #                            method='RK45',
                                method='LSODA',
                                max_step=1e-2,  # we want high phi resolution
                                atol=atol,
                                rtol=1e-8,
                                t_eval = phi_eval
                                )

                if self._verbose:
                    print("{}, {}".format(sol.message, sol.nfev))

                phi = sol.t
                R, Z = sol.y

                fl = self.coordinates(R, Z, phi)

                # XXX add condirtion to stopper
                if in_first_wall:
                    mask = self.in_first_wall(fl)

                    imask = mask.astype(int)
                    in_idxs = np.where(imask[:-1] - imask[1:] == 1)[0]

                    last_idx = False
                    if len(in_idxs) >= 1:
                        # Last point is still in (+1)
                        last_idx = in_idxs[0]
                        mask[last_idx + 1:] = False

                    Rs = fl.R[mask]
                    Zs = fl.Z[mask]
                    phis = fl.phi[mask]

                    intersec = self.first_wall.intersection(fl, dim=2)
                    if intersec is not None and len(in_idxs) >= 1:
                        R_last = Rs[-1]
                        Z_last = Zs[-1]

                        inter_idx = np.argmin((intersec.R - R_last) ** 2 + (intersec.Z - Z_last) ** 2)

                        Rx = intersec.R[inter_idx]
                        Zx = intersec.Z[inter_idx]
                        # last_idx = len(phis) - 1

                        coef = np.sqrt((Rx - fl.R[last_idx]) ** 2 + (Zx - fl.Z[last_idx]) ** 2 /
                                    (fl.R[last_idx + 1] - fl.R[last_idx]) ** 2 +
                                    (fl.Z[last_idx + 1] - fl.Z[last_idx]) ** 2)

                        phix = fl.phi[last_idx] + coef * (fl.phi[last_idx + 1] - fl.phi[last_idx])

                        Rs = np.append(Rs, Rx)
                        Zs = np.append(Zs, Zx)
                        phis = np.append(phis, phix)

                    fl = self.coordinates(Rs, Zs, phis)

                res.append(fl)


        return res



    def FL_lookup(eq, phi_grid, R_noeud, Z_noeud, phi_min, phi_max, IntegrationDistanceAlongLOS, dPhi = 1, psi_lim = 2.5):
    
        phi_grid = phi_grid/180*np.pi
        dPhirad = dPhi*np.pi/180 #conversion en radians
        phi_max = phi_max+dPhirad
        phi_min = phi_min-dPhirad
        dim_fl = int(np.ceil((phi_max-phi_min)/dPhirad))
        phi_mem = np.arange(phi_min, phi_max, dPhirad)
        dPhirad = np.mean(np.diff(phi_mem))
        ind_PHI = int(np.ceil((phi_grid-phi_min)/dPhirad))
        # phi_mem[ind_PHI-1] = phi_grid
        FL_MATRIX = np.zeros((len(R_noeud)*len(Z_noeud),  2, dim_fl))
        FL_MATRIX[:] = np.NaN
        RZ_noeud_R, RZ_noeud_Z = np.meshgrid(R_noeud, Z_noeud, indexing = 'ij')
        R_line = np.reshape(RZ_noeud_R, len(R_noeud)*len(Z_noeud))
        Z_line = np.reshape(RZ_noeud_Z, len(R_noeud)*len(Z_noeud))

        # chord = eq.coordinates(R = RZ_noeud_R, Z = RZ_noeud_Z, phi = np.ones(len(R_noeud)*len(Z_noeud))*phi_grid)
        # row_fl, col_fl = np.where(chord.psi_n<= psi_lim)

        
        chord = eq.coordinates(R = R_line, Z = Z_line, phi = np.ones(len(Z_line))*phi_grid)
        ind_psi = eq.in_first_wall(chord)

        # ind_psi = chord.psi_n<=psi_lim
        # ind_psi_ind = np.where(chord.psi_n<= psi_lim)[0]
        chord = eq.coordinates(R = R_line[ind_psi], Z = Z_line[ind_psi], phi = np.ones(len(Z_line[ind_psi]))*phi_grid)
        field_line_down = trace_field_line_toroidal(eq, chord, in_first_wall=False, phi_end = phi_min, phi_eval = np.flip(phi_mem[:ind_PHI], axis = 0))


        field_line_up = trace_field_line_toroidal(eq, chord, in_first_wall=False, phi_end = phi_max, phi_eval = phi_mem[ind_PHI:])




        FL_R = np.zeros((len(chord), len(phi_mem)))
        FL_Z = np.zeros((len(chord), len(phi_mem)))
        for i in range(len(field_line_down)):
            R_fl = np.concatenate((np.flip(field_line_down[i].R, axis=0), field_line_up[i].R))
            Z_fl = np.concatenate((np.flip(field_line_down[i].Z, axis=0), field_line_up[i].Z))
            phi_fl = np.concatenate((np.flip(field_line_down[i].phi, axis=0), field_line_up[i].phi))
            if not np.array_equiv(phi_fl, phi_mem):
                raise ValueError("integration on toroidal angle incorrectly borned")
            chord = eq.coordinates(R = R_fl, Z = Z_fl, phi = phi_fl)
            #wall_mask = eq.in_first_wall(chord)
            #R_fl[np.invert(wall_mask)] = np.NaN
            #Z_fl[np.invert(wall_mask)] = np.NaN
            FL_R[i, :] = R_fl
            FL_Z[i, :] = Z_fl
        FL_MATRIX[ind_psi, 0, :] = FL_R
        FL_MATRIX[ind_psi, 1, :] = FL_Z
        return FL_MATRIX, dPhirad, phi_min, phi_mem
        
        
    def check_intersection_wall(realcam, world):
        Intersection = np.empty(realcam.pixels, dtype = 'object')
        RPHIZ = np.full((realcam.pixels[0], realcam.pixels[1], 3), np.NaN)
        realcam.parent = world
        from cherab.tools.observers.intersections import find_wall_intersection
        for i in range(realcam.pixels[0]):
            for j in range(realcam.pixels[1]):
                if not np.isnan(realcam.pixel_origins[i, j][0]):
                    try:
                        point, obj = find_wall_intersection(world, realcam.pixel_origins[i,j], realcam.pixel_directions[i, j])
                        Intersection[i, j] = obj.name
                        POINT = np.array([point.x, point.y, point.z])
                        pointRPHIZ = utility_functions.xyztorphiz(POINT)
                        RPHIZ[i, j, :] = pointRPHIZ
                    except:
                        # raise Exception('pixel %02d, %02d does not intersect with wall' % (i, j))
                        print('pixel %02d, %02d does not intersect with wall' % (i, j))
        Intersection[Intersection==None] = 'None'

        return RPHIZ, Intersection
            
    def optimize_grid_from_los(realcam, world, pos_camera_RPHIZ):
        RPHIZ, Intersection = check_intersection_wall(realcam, world)
        R_min_noeud = pos_camera_RPHIZ[0] 
        R_max_noeud = pos_camera_RPHIZ[0] 
        Z_min_noeud = pos_camera_RPHIZ[2] 
        Z_max_noeud = pos_camera_RPHIZ[2] 
        for i in range(RPHIZ.shape[0]): 
            for j in range(RPHIZ.shape[1]):

                R_min, R_max, Z_min, Z_max = utility_functions.find_RZextrema_between_2_points(pos_camera_RPHIZ, RPHIZ[i, j, :])
                R_min_noeud = min(R_min_noeud, R_min)
                R_max_noeud = max(R_max_noeud, R_max)
                Z_min_noeud = min(Z_min_noeud, Z_min)
                Z_max_noeud = max(Z_max_noeud, Z_max)

        #setting [-pi, pi] intervall back into [0, 2pi] intervall
        RPHIZ[:, :, 1] = (RPHIZ[:, :, 1] +2*np.pi) % (2*np.pi)
        
        phi_min = min(pos_camera_RPHIZ[1], np.nanmin(RPHIZ[:, :, 1]))
        phi_max = max(pos_camera_RPHIZ[1], np.nanmax(RPHIZ[:, :, 1]))
        
        return R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, phi_max, phi_min, RPHIZ




    def optimize_grid_from_camera(realcam, phi_grid, RZwall, pos_camera_RPHIZ):
        from matplotlib.pyplot import ion, ioff
        world_intersection = World()
        realcam_intersection = VectorCamera(realcam.pixel_origins, realcam.pixel_directions, parent = world_intersection, frame_sampler = FullFrameSampler2D(realcam.frame_sampler.mask))
        x_lower = np.nanmin(RZwall[:, 0])
        y_lower = 0
        z_lower = np.nanmin(RZwall[:, 1])
        lower_point = Point3D(x_lower, y_lower, z_lower)

        x_upper = np.nanmax(RZwall[:, 0])
        y_upper = 0.01
        z_upper = np.nanmax(RZwall[:, 1])
        upper_point = Point3D(x_upper, y_upper, z_upper)

        fake_wall = Box(lower_point, upper_point, material=AbsorbingSurface(), parent=world_intersection, transform = rotate_z(phi_grid))
        RPHIZ, Intersection = check_intersection_wall(realcam_intersection, world_intersection)
        R_min_noeud = pos_camera_RPHIZ[0] 
        R_max_noeud = pos_camera_RPHIZ[0] 
        Z_min_noeud = pos_camera_RPHIZ[2] 
        Z_max_noeud = pos_camera_RPHIZ[2] 


        # for i in range(RPHIZ.shape[0]): 
        #     for j in range(RPHIZ.shape[1]):

        #         R_min, R_max, Z_min, Z_max = utility_functions.find_RZextrema_between_2_points(pos_camera_RPHIZ, RPHIZ[i, j, :])
        #         R_min_noeud = min(R_min_noeud, R_min)
        #         R_max_noeud = max(R_max_noeud, R_max)
        #         Z_min_noeud = min(Z_min_noeud, Z_min)
        #         Z_max_noeud = max(Z_max_noeud, Z_max)

        R_max_noeud = np.nanmax(RPHIZ[:, :, 0])
        R_min_noeud = np.nanmin(RPHIZ[:, :, 0])
        Z_max_noeud = np.nanmax(RPHIZ[:, :, 2])
        Z_min_noeud = np.nanmin(RPHIZ[:, :, 2])
        
        
        #setting [-pi, pi] intervall back into [0, 2pi] intervall
        RPHIZ[:, :, 1] = (RPHIZ[:, :, 1] +2*np.pi) % (2*np.pi)
        
        phi_min = min(pos_camera_RPHIZ[1], np.nanmin(RPHIZ[:, :, 1]))
        phi_max = max(pos_camera_RPHIZ[1], np.nanmax(RPHIZ[:, :, 1]))

        R_max_noeud = min(R_max_noeud, np.nanmax(RZwall[:, 0]))
        R_min_noeud = max(R_min_noeud, np.nanmin(RZwall[:, 0]))
        Z_max_noeud = min(Z_max_noeud, np.nanmax(RZwall[:, 1]))
        Z_min_noeud = max(Z_min_noeud, np.nanmin(RZwall[:, 1]))
            
        world_rgb =World()
        wall_limit = axisymmetric_mesh_from_polygon(RZwall)
        wall_limit.parent = world_rgb
        wall_limit.material = UniformVolumeEmitter(colours[1])
        fake_wall = Box(lower_point, upper_point, material=UniformVolumeEmitter(colours[0]), parent=world_rgb, transform = rotate_z(phi_grid))

        rgb = RGBPipeline2D()
        sampler = RGBAdaptiveSampler2D(rgb, min_samples=100, fraction=0.2, cutoff=0.01)
        # camera = VectorCamera(realcam.pixel_origins, realcam.pixel_directions, parent=world_rgb, pipelines=[rgb], frame_sampler=sampler)

        camera = PinholeCamera((128, 256), parent=world_rgb, pipelines=[rgb], frame_sampler=sampler, transform = translate(0, 0, -4))
        camera.spectral_rays = 1
        camera.spectral_bins = 15
        camera.pixel_samples = 250

        ion()
        name = 'transform test'
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        render_pass = 1
        camera.observe()
        # rgb.save("{}_{}_pass_{}.png".format(name, timestamp, render_pass))
        # while not camera.render_complete:

        #     print("Rendering pass {}...".format(render_pass))
        #     camera.observe()
        #     rgb.save("{}_{}_pass_{}.png".format(name, timestamp, render_pass))
        #     print()

        #     render_pass += 1

        ioff()
        rgb.display()

        return R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, phi_max, phi_min, RPHIZ

    #get wall coordinates to get the cherab object
    if ParamsGrid.symetry == 'magnetic': #overwrites wall with the one saved in pleque
        t_pleque = ParamsGrid.t_grid #choose middle time of the inversion for time of equilibrium
        t_pleque = t_pleque*1000 # converting in ms for pleque
        import pleque.io.compass as plq
        print("magnetic symmetry")
        revision_mag = ParamsGrid.revision
        revision_mag = revision_mag or 1
        variant_mag =  ParamsGrid.variant_mag
        variant_mag = variant_mag or ''
        eq = plq.cdb(ParamsGrid.nshot, t_pleque, revision = revision_mag, variant = variant_mag)
        RZwall = np.array([eq.first_wall.R,eq.first_wall.Z]).T
        RZwall =RZwall[:-1, :] 
        RZwall = RZwall[::-1]
    start = time.time()

    wall_limit = axisymmetric_mesh_from_polygon(RZwall)
    R_wall = RZwall[:, 0]
    Z_wall = RZwall[:, 1]


    visible_pix = np.where(realcam.frame_sampler.mask) 
    pos_camera = realcam.pixel_origins[visible_pix[0][0]  , visible_pix[1][0]]
    pos_camera = np.array([pos_camera.x,pos_camera.y, pos_camera.z] )
    pos_camera_RPHIZ = utility_functions.xyztorphiz(pos_camera)
    if pos_camera_RPHIZ[1]<0: #setting [-pi, pi] interval into [0, 2pi] interval
        pos_camera_RPHIZ[1] = pos_camera_RPHIZ[1] + 2 *np.pi
    # R_max_noeud = pos_camera_RPHIZ[0] 
    # R_min_noeud = pos_camera_RPHIZ[0] 
    # Z_max_noeud = pos_camera_RPHIZ[2] 
    # Z_min_noeud = pos_camera_RPHIZ[2] 
    # R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, phi_max, phi_min = optimize_boundary_grid(realcam, world, R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud)
    if ParamsMachine.machine == 'WEST':
        R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud =[3.129871200000000, 1.834345700000000,0.798600000000000,-0.789011660000000]
    else:
        if ParamsGrid.phi_grid == 'auto':
            R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, phi_max, phi_min, RPHIZ = optimize_grid_from_los(realcam, world, pos_camera_RPHIZ)
            print(R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, phi_max, phi_min)
            phi_grid = (phi_max+phi_min)/2*180/np.pi
            print(phi_grid)

        else:
            phi_grid = ParamsGrid.phi_grid
        R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, dud1,dud2,  RPHIZ = optimize_grid_from_camera(realcam, phi_grid, RZwall,pos_camera_RPHIZ)
        print(R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, phi_max, phi_min)
    if ParamsGrid.extra_steps:
        R_max_noeud = R_max_noeud+ParamsGrid.extra_steps*ParamsGrid.dr_grid
        R_min_noeud = R_min_noeud-ParamsGrid.extra_steps*ParamsGrid.dr_grid
        Z_max_noeud = Z_max_noeud+ParamsGrid.extra_steps*ParamsGrid.dz_grid
        Z_min_noeud = Z_min_noeud-ParamsGrid.extra_steps*ParamsGrid.dz_grid
     # if verbose:
    #     fig = utility_functions.plot_cylindrical_coordinates(RPHIZ)
    #     fig = utility_functions.plot_line_from_cylindrical(pos_camera_RPHIZ, RPHIZ[0,0,:], fig, color = 'blue', label = 'point [0,0]')
    #     fig = utility_functions.plot_line_from_cylindrical(pos_camera_RPHIZ, RPHIZ[-1,0,:], fig, color = 'red', label = 'point [-1,0]')
    #     fig = utility_functions.plot_line_from_cylindrical(pos_camera_RPHIZ, RPHIZ[0,-1,:], fig, color = 'green', label = 'point [0,-1]')
    #     fig = utility_functions.plot_line_from_cylindrical(pos_camera_RPHIZ, RPHIZ[-1,-1,:], fig, color = 'yellow', label = 'point [-1,-1]')
    #     plt.savefig(main_folder_image + 'images line of sight and wall')
    extent_RZ =[R_min_noeud, R_max_noeud, Z_min_noeud, Z_max_noeud] 
    nb_node_r = int((R_max_noeud-R_min_noeud)/ParamsGrid.dr_grid)
    nb_node_z = int((Z_max_noeud-Z_min_noeud)/ParamsGrid.dz_grid)
    cell_r, cell_z, grid_mask, cell_dr, cell_dz = get_mask_from_wall(R_min_noeud, R_max_noeud, Z_min_noeud, Z_max_noeud, nb_node_r, nb_node_z, wall_limit, ParamsGrid.crop_center, ParamsGrid)
    # The RayTransferCylinder object is fully 3D, but for simplicity we're only
    # working in 2D as this case is axisymmetric. It is easy enough to pass 3D
    # views of our 2D data into the RayTransferCylinder object: we just ues a
    # numpy.newaxis (or equivalently, None) for the toroidal dimension.
    grid_mask = grid_mask[:, np.newaxis, :]
    if ParamsGrid.symetry =='magnetic':
        n_polar = ParamsGrid.n_polar
    else:
        n_polar = 1
    RZ_mask_grid = np.copy(grid_mask)
    grid_mask = np.tile(grid_mask, (1, n_polar, 1))
    num_points_rz = nb_node_r*nb_node_z
    # R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, phi_min, phi_max = optimize_grid(R_max_noeud, R_min_noeud, Z_max_noeud, Z_min_noeud, realcam, phi_grid, RZwall)
    
    #recalculate the extremities of the grid; the grid starts at (r_min, z_min) and its last point is (r_max-dr_grid, z_max-dz_grid)

   
    if ParamsGrid.symetry =='magnetic':
                
        start = time.time()
        
        FL_MATRIX, dPhirad, phi_min, phi_mem = FL_lookup(eq, phi_grid, cell_r, cell_z, phi_min, phi_max, 0.5e-3, 0.2)
        end = time.time()

        elapsed = end - start
        print(f"Magnetic field lines calculation : {elapsed:.3f} seconds")

        
        cell_r_precision, cell_z_precision, grid_mask_precision, cell_dr_precision, cell_dz_precision = get_mask_from_wall(R_min_noeud, R_max_noeud, Z_min_noeud, Z_max_noeud, nb_node_r*ParamsGrid.grid_precision_multiplier, nb_node_z*ParamsGrid.grid_precision_multiplier, wall_limit, ParamsGrid.crop_center, ParamsGrid)
        grid_mask_precision = grid_mask_precision[:, np.newaxis, :]
        grid_mask_precision = np.tile(grid_mask_precision, (1, n_polar, 1))
        phi_tour = np.linspace(0, 360, n_polar, endpoint = False)
        phi_vision = (phi_tour>phi_min*180/np.pi) & (phi_tour<phi_max*180/np.pi) #only keep points in the grid into the esteemed line of sights
        grid_mask_precision[:, np.invert(phi_vision), :] = True
        phi_tour = phi_tour[phi_vision]
        plasma = RayTransferCylinder(radius_outer = R_max_noeud, 
                                    height= Z_max_noeud-Z_min_noeud, 
                                    n_radius = nb_node_r*ParamsGrid.grid_precision_multiplier, 
                                    n_height = nb_node_z*ParamsGrid.grid_precision_multiplier, 
                                    radius_inner = R_min_noeud,  
                                    transform=translate(0., 0., Z_min_noeud), 
                                    n_polar = n_polar, 
                                    mask = grid_mask_precision,
                                    period = 360)
        

        # plasma = RayTransferCylinder(R_max_noeud, nb_node_z*dz_grid, nb_node_r*grid_precision_multiplier, nb_node_z*grid_precision_multiplier, radius_inner = R_min_noeud,  parent = world, transform=translate(0., 0., Z_min_noeud), n_polar = n_polar, period = 360, voxel_map = plasma.voxel_map)
        # plasma.voxel_map[~grid_mask] = -1 
        # pdb.set_trace()
        # voxel_map = create_voxel_map_from_equilibrium(FL_MATRIX, plasma, cell_r_precision, cell_z_precision, grid_mask_precision, cell_dr_precision, cell_dz_precision, phi_mem, dPhirad, wall_limit, dict_transfert_matrix)
        voxel_map = create_voxel_map_from_equilibrium_query(FL_MATRIX, plasma, cell_r_precision, cell_z_precision, grid_mask_precision, cell_dr_precision, cell_dz_precision, phi_mem, dPhirad, wall_limit)
    
        plasma2 = RayTransferCylinder(
            radius_outer=R_max_noeud,
            radius_inner=R_min_noeud,
            height=nb_node_z*ParamsGrid.dz_grid,
            n_radius=nb_node_r*ParamsGrid.grid_precision_multiplier, 
            n_height=nb_node_z*ParamsGrid.grid_precision_multiplier,  
            n_polar=n_polar,
            mask = grid_mask_precision,
            voxel_map = voxel_map,
            period = 360,
            parent = world,
            transform=translate(0, 0, Z_min_noeud)
        )
        

    elif ParamsGrid.symetry == 'toroidal':
        phi_tour = n_polar
        plasma2 = RayTransferCylinder(radius_outer=cell_r[-1],
                                        radius_inner=cell_r[0],
                                        height=cell_z[-1] - cell_z[0],
                                        n_radius=nb_node_r, n_height=nb_node_z, 
                                        mask=grid_mask, n_polar=n_polar, 
                                        parent = world, transform=translate(0, 0, cell_z[0]))
    
    else:
        raise(NameError('unrecognized symetry, write toroidal or magnetic'))
    # if verbose:
    #     plt.figure()
    #     plt.imshow(np.sum(plasma.voxel_map, 1).T, extent= extent_RZ, origin = 'lower' )
    #     plt.show(block = False)
    #     plt.savefig(main_folder_image + '2D_voxel_map.png')
    
    #calculate inversion matrix
    print(plasma2.bins)
    print(num_points_rz)
    realcam.spectral_bins = plasma2.bins #set the grid to a size (NR, NZ) plus 1 extra node for elements of the grid too far from calculated field lines
    if realcam.spectral_bins >realcam.pixels[0]*realcam.pixels[1]:
        raise Exception("more nodes than pixels, inversion is impossible. Lower dr_grid or dz_grid")
    if realcam.spectral_bins >10000:
        print("careful, huge number of nodes")
    # if verbose:
    #     compare_voxel_map_and_pleque(plasma2, FL_MATRIX, Z_min_noeud, phi_mem)

    # if verbose:
    #     create_synth_cam_emitter(realcam, full_wall, R_wall, Z_wall, mask, path_CAD, variant = variant)
    
    
    end = time.time()

    elapsed = end - start
    print(f"time setup camera : {elapsed:.3f} seconds")

    realcam.observe()
    pipelines = realcam.pipelines[0]
    print('shape full transfert matrix = ' + str(pipelines.matrix.shape))
    flattened_matr = pipelines.matrix.reshape(pipelines.matrix.shape[0] * pipelines.matrix.shape[1], pipelines.matrix.shape[2])
    
    if flattened_matr.shape[1] > num_points_rz:
        #some elements of the grid don't see the field lines. Checking if they are out of the field of view of the camera
        invisible_nodes = np.sum(flattened_matr, 0)[-1]
        if invisible_nodes>0:
            print('nodes not seen, choose bigger grid limits, or wall limits differ between CAD model and magnetic equilibrium')
            # pdb.set_trace()
        flattened_matr = flattened_matr[:, :-1]
    print('flattened_matr shape', flattened_matr.shape)


    pixel,  = np.where(np.sum(flattened_matr, 1)) #sum over nodes
    node,  = np.where(np.sum(flattened_matr, 0)) #sum over pixels
    #save results

    mask_pixel = np.zeros(flattened_matr.shape[0], dtype = bool)
    mask_pixel[pixel] = True
    mask_pixel = mask_pixel.reshape(pipelines.matrix.shape[0:2])

    # x, y, z = np.where(RZ_mask_grid)
    # x = x[node]
    # y = y[node]
    # # z = z[node]
    if ParamsGrid.symetry == 'magnetic':
        mask_node = np.zeros_like(RZ_mask_grid, dtype = bool)
        rows_noeud, indphi, cols_noeud = np.unravel_index(node, mask_node.shape)
        mask_node[rows_noeud,indphi, cols_noeud] = True
    elif ParamsGrid.symetry == 'toroidal':
        true_nodes = np.flatnonzero(RZ_mask_grid)
        mask_node = np.zeros_like(RZ_mask_grid, dtype = bool)
        mask_node.ravel()[true_nodes[node]] = True
    mask_node = np.squeeze(mask_node)
    print('shape voxel_map ', plasma2.voxel_map.shape)
    print('shape mask_node ', mask_node.shape)
    
    transfert_matrix = flattened_matr[pixel,:][:, node]

    nb_visible_node = len(np.unique(node))
    nb_vision_pixel = len(np.unique(pixel))
    print('visible node = ' + str(nb_visible_node) + 'out of ' + str(nb_node_r*nb_node_z))
    print('vision pixel = ' + str(nb_vision_pixel) + 'out of ' + str(pipelines.matrix.shape[0] * pipelines.matrix.shape[1]))

    print(transfert_matrix.shape)
    pixel = np.squeeze(pixel)
    node = np.squeeze(node)
    
    print('shape reduced transfert matrix = ' + str(transfert_matrix.shape))
    rows_node, cols_node = np.unravel_index(node, mask_node.shape)
    rows_pixel, cols_pixel = np.unravel_index(pixel, mask_pixel.shape)
    
    return transfert_matrix, mask_node, mask_pixel, node, pixel, rows_node, cols_node, rows_pixel, cols_pixel, cell_r, cell_z




def get_mask_from_wall(R_min_noeud, R_max_noeud, Z_min_noeud, Z_max_noeud, nb_node_r, nb_node_z, wall_limit, crop_center, ParamsGrid):

    cell_r, cell_dr = np.linspace(R_min_noeud, R_max_noeud, nb_node_r, retstep=True, endpoint = False)
    cell_z, cell_dz = np.linspace(Z_min_noeud, Z_max_noeud, nb_node_z, retstep=True, endpoint = False)
    cell_r_grid, cell_z_grid = np.broadcast_arrays(cell_r[:, None], cell_z[None, :])
    cell_centres = np.stack((cell_r_grid, cell_z_grid), axis=-1)  # (nx, ny, 2) array
    # Define the positions of the vertices of the voxels
    cell_vertices_r = np.linspace(cell_r[0] - 0.5 * cell_dr, cell_r[-1] + 0.5 * cell_dr, nb_node_r + 1)
    cell_vertices_z = np.linspace(cell_z[0] - 0.5 * cell_dz, cell_z[-1] + 0.5 * cell_dz, nb_node_z + 1)

    # Build a mask, only including cells within the wall
   
    vertex_mask = np.zeros((len(cell_vertices_r), len(cell_vertices_z)))
    

    for i in range(nb_node_r):
        for j in range(nb_node_z):
            r_noeud = cell_vertices_r[i]
            
            z_noeud = cell_vertices_z[j]
            pointrz = Point3D(r_noeud, 0, z_noeud)
            if   wall_limit.contains(pointrz):          
                vertex_mask[i, j]= 1

    #remove points for psi_norm<0.9
    if crop_center:
        vertex_mask = remove_center_from_inversion(vertex_mask, cell_vertices_r, cell_vertices_z, ParamsGrid)

    # Cell is included if at least one vertex is within the wall
    grid_mask = (vertex_mask[1:, :-1] + vertex_mask[:-1, :-1]
                + vertex_mask[1:, 1:] + vertex_mask[:-1, 1:])
    grid_mask=grid_mask>0
    return cell_r, cell_z, grid_mask, cell_dr, cell_dz


def remove_center_from_inversion(vertex_mask, cell_vertices_r, cell_vertices_z, ParamsGrid):

    def is_point_in_contour(x, y, contour_points):
        from matplotlib.path import Path

        contour_path = Path(contour_points)
        return contour_path.contains_point((x, y))
    
    nshot = ParamsGrid.nshot
    try:
        magflux = utility_functions.call_module_function_in_environment('west_functions','get_equilibrium', 'python-3.11', nshot)

        r = magflux.interp2D.r
        z = magflux.interp2D.z
        idx = magflux.interp2D.psi.shape[0]//2
        psi = magflux.interp2D.psi[idx, ...]

        #get psi norm
        # self.r0 = 
        # self.z0 = 
        # self.psi_norm = magflux.interp2D.psi[idx, :, :]/magflux.boundary.psi[idx]
        # self.rsep = magflux.boundary.r[idx, :] doesn't exist
        # self.zsep = magflux.boundary.z[idx, :] doesn't exist
        psisep = magflux.boundary.psi[idx]
        psi0 = np.nanmax(psi)

        psi_int = 0.9*(psisep-psi0)+psi0
    except:
        magflux = loadmat('/Home/LF285735/Documents/MATLAB/shot 61357 sep 7_7s.mat', simplify_cells = True)
        magflux = magflux['equi']
        r = magflux['interp2D']['r']
        z = magflux['interp2D']['z']
        idx = magflux['interp2D']['psi'].shape[0]//2
        psi = magflux['interp2D']['psi'][idx, :, :]
        psisep = magflux['boundary']['psi'][idx]
        psi0 = np.nanmax(psi)
        psi_int = 0.9*(psisep-psi0)+psi0
    # Create contour for just this isovalue
    contour = plt.contour(r, z, psi, levels=[psi_int], colors='red')
    plt.close()
    # Extract the contour line(s)
    paths = contour.collections[0].get_paths()

    # Optional: grab the first contour line only
    contour_points = paths[0].vertices  # shape (N, 2), columns are [x, y]

    for i in range(len(cell_vertices_r)):
            for j in range(len(cell_vertices_z)):
                r_noeud = cell_vertices_r[i]
                z_noeud = cell_vertices_z[j]
                if   is_point_in_contour(r_noeud, z_noeud, contour_points):          
                    vertex_mask[i, j]= 0

    
    return vertex_mask


def create_voxel_map_from_equilibrium_query(FL_MATRIX, plasma, cell_r_precision, cell_z_precision, grid_mask_precision, cell_dr_precision, cell_dz_precision, phi_mem, dPhirad, wall_limit):
    def closest_points(grid_points, query_points):
        """
        Find the closest points in a 2D irregular grid for many queries.

        Parameters
        ----------
        grid_points : ndarray of shape (N, 2)
            Array of (x, y) grid points.
        query_points : ndarray of shape (M, 2)
            Array of query points (x, y).

        Returns
        -------
        closest_idx : ndarray of shape (M,)
            Indices of the closest grid points.
        closest_points : ndarray of shape (M, 2)
            Coordinates of the closest grid points.
        distances : ndarray of shape (M,)
            Distances to the closest points.
        """
        from scipy.spatial import cKDTree

        tree = cKDTree(grid_points)
        dists, idxs = tree.query(query_points)
        return idxs, grid_points, dists



    # turning angle back into degrees

    nb_node_r = len(cell_r_precision)
    nb_node_z = len(cell_z_precision)
    R, Z = np.meshgrid(cell_r_precision, cell_z_precision, indexing = 'ij')

    phi_min = phi_mem[0]*180/np.pi
    phi_max = phi_mem[-1]*180/np.pi
    # seuil = np.sqrt((plasma.material.dr*grid_precision_multiplier)**2+(plasma.material.dz*grid_precision_multiplier)**2)*3
    seuil = np.sqrt((np.mean(np.diff(cell_r_precision)))**2+(np.mean(np.diff(cell_z_precision)))**2)
    n_polar = plasma.material.grid_shape[1]
    voxel_map = -1*np.ones_like(plasma.voxel_map) #setting all nodes to blind 
    query_points = np.column_stack((R.flatten(), Z.flatten()))

    for nphi in range(n_polar):
        if nphi*plasma.material.dphi>phi_max or nphi*plasma.material.dphi<phi_min:
            voxel_map[:, nphi, :] = -1
            # print('phi out of range')
        else:
        
            ind_phi_closest = np.round((nphi*plasma.material.dphi-phi_min)/(dPhirad*180/np.pi)).astype('int')
            #print('phi in range')
            grid_points = FL_MATRIX[:, :, ind_phi_closest] 

            idxs, _,  dists = closest_points(grid_points, query_points)
            voxel_map[:, nphi, :] = idxs.reshape(nb_node_r, nb_node_z)

    voxel_map[np.invert(grid_mask_precision)] = -1
    print(np.max(voxel_map))
    return voxel_map



def get_camera(ParamsMachine, ParamsGrid, world):
    print('loading camera')
    real_pipeline = RayTransferPipeline2D()

    realcam = load_camera(ParamsMachine.path_calibration)
    mask_pixel, name_mask = load_mask(ParamsMachine.path_calibration, ParamsMachine.path_mask)
    print('camera loaded')
    mask_pixel = mask_pixel.T #necessary to rotate mask for correct pleque convention. Calcam handles the rotation correction for the camera object, not for its mask 
    mask_pixel = np.ascontiguousarray(mask_pixel)
    if ParamsMachine.param_fit is not None:
        metadata_vid = utility_functions.read_vid_metadata(ParamsMachine, ParamsGrid)
        realcam, mask_pixel, fit_shape = fit_size_all(realcam, mask_pixel, metadata_vid, ParamsMachine.param_fit)
    else:
        fit_shape = None
    if ParamsMachine.decimation is not None and ParamsMachine.decimation != 1:
        realcam, mask_pixel = reduce_camera_precision(realcam, mask_pixel, decimation = ParamsMachine.decimation)

    realcam.frame_sampler=FullFrameSampler2D(mask_pixel)
    realcam.pipelines=[real_pipeline]
    realcam.parent=world
    realcam.pixel_samples = ParamsGrid.pixel_samples
    realcam.min_wavelength = 640
    realcam.max_wavelength = realcam.min_wavelength +1
    realcam.render_engine.processes = 32        
    return realcam, fit_shape



def load_camera(path_calibration):
    try: 
        import calcam
        calcam_camera = calcam.Calibration(load_filename = path_calibration)
        realcam = calcam_camera.get_raysect_camera(coords = 'Display')
        pixel_origins = realcam.pixel_origins
        mask = calcam_camera.subview_mask
        pixel_directions = realcam.pixel_directions
    except:
        calcam_camera = np.load(path_calibration, allow_pickle=True)
        pixel_origins = calcam_camera['pixel_origins']
        pixel_directions = calcam_camera['pixel_directions']
        realcam = VectorCamera(pixel_origins.T, pixel_directions.T)
    return realcam



def load_mask(path_calibration, path_mask):
    if path_mask is None:
        try:
            import calcam
            calcam_camera = calcam.Calibration(path_calibration)
            mask_pixel = calcam_camera.subview_mask
            mask_pixel = np.invert(mask_pixel)
            mask_pixel = np.ascontiguousarray(mask_pixel)
            name_mask = 'calibration_calcam'  
        except:
            try:
                calcam_camera = np.load(path_calibration, allow_pickle=True)

                mask_pixel = calcam_camera['mask']
                mask_pixel = np.invert(mask_pixel)
                name_mask = 'calibration_calcam'
            except:
                name_mask = utility_functions.get_name(path_mask)
                try:
                    fmask = loadmat(path_mask)
                    mask_pixel = fmask["mask"]
                except:
                    mask_pixel = np.load(path_mask)

    else:
        try:
            mask_pixel = calcam_camera['mask']
            mask_pixel = np.invert(mask_pixel)
            name_mask = 'calibration_calcam'
        except:
            name_mask = utility_functions.get_name(path_mask)
            try:
                fmask = loadmat(path_mask)
                mask_pixel = fmask["mask"]
            except:
                mask_pixel = np.load(path_mask)
    mask_pixel[np.isnan(mask_pixel)] = 0 #handle ill defined mask
    mask_pixel = np.abs(mask_pixel)# handle calcam convention, set all the non zero indices to positive for further process 
    mask_pixel.dtype = bool
    return mask_pixel, name_mask



def fit_size_all(camera, mask, metadata_vid, param_fit = None):
    if param_fit == 'mask':
        target_shape = mask.shape
    elif param_fit == 'camera':
        target_shape = camera.pixel_directions.shape
    elif param_fit == 'vid':
        target_shape = metadata_vid['Image Shape']
    else:
        if mask.shape != camera.pixel_directions.shape or mask.shape !=metadata_vid['Image Shape']:
            raise Exception('careful, discrepancy in elements shape')
        else:
            return camera, mask
    mask = resize_matrix(mask, target_shape)
    pixel_directions = resize_matrix(camera.pixel_directions, target_shape)
    pixel_origins = resize_matrix(camera.pixel_origins, target_shape)
    camera = VectorCamera(pixel_origins, pixel_directions)
    
    return camera, mask, target_shape



def resize_matrix(matrix, target_shape):
    """
    Resize a matrix to match the target shape by cropping or zero-padding.
    Cropping and padding are centered along each axis.
    """
    target_rows, target_cols = target_shape
    current_rows, current_cols = matrix.shape

    # Determine row cropping/padding
    if current_rows > target_rows:
        start_row = (current_rows - target_rows) // 2
        matrix = matrix[start_row:start_row + target_rows, :]
    else:
        pad_before = (target_rows - current_rows) // 2
        pad_after = target_rows - current_rows - pad_before
        matrix = np.pad(matrix, ((pad_before, pad_after), (0, 0)), mode='constant')

    # Determine column cropping/padding
    if current_cols > target_cols:
        start_col = (current_cols - target_cols) // 2
        matrix = matrix[:, start_col:start_col + target_cols]
    else:
        pad_before = (target_cols - current_cols) // 2
        pad_after = target_cols - current_cols - pad_before
        matrix = np.pad(matrix, ((0, 0), (pad_before, pad_after)), mode='constant')

    return matrix



def reduce_camera_precision(camera, mask, decimation =1):
    decimation = decimation or 1
    mask = downsample_with_avg(mask, decimation)
    # mask = mask[::decimation, ::decimation]
    


    pixel_directions = downsample_with_avg(camera.pixel_directions, decimation)
    # pixel_directions = pixel_directions[::decimation, ::decimation]
    pixel_origins = camera.pixel_origins[::decimation, ::decimation]
    pixel_origins[np.invert(mask)] = Point3D(np.NaN, np.NaN, np.NaN)

    camera = VectorCamera(pixel_origins, pixel_directions)

    return camera, mask



def downsample_with_avg(matrix, block_size=4):
    # Get the shape of the matrix
    rows, cols = matrix.shape

     # Create a result matrix initialized with the original values
    

    result = matrix[::block_size, ::block_size]
    
    
     # Loop over the matrix with the step of block_size (e.g., 4)
    for i in range(0, rows, block_size):
        for j in range(0, cols, block_size):
            # Determine the block to average (handle edge cases where the block is smaller)
            block = matrix[i:i+block_size, j:j+block_size]
        
            # Check if the block is a full block (all 4x4 elements)
            if block.shape[0] == block_size and block.shape[1] == block_size:
                 # Compute the mean of the block
                block_mean = np.mean(block)
                 
                 # Assign the mean to the result matrix at the block's top-left corner
                if matrix.dtype == bool:
                    if block.all() == True:
                        result[i//block_size, j//block_size] = True
                    else:
                        result[i//block_size, j//block_size] = False

                else:
                    result[i//block_size, j//block_size] = block_mean
     
    return result



def setup_raytracing_world(ParamsMachine, ParamsGrid):
    world = World()

    try:
        fwall = loadmat(ParamsMachine.path_wall)
        RZwall = fwall['RZwall']

    except:
        RZwall = np.load(ParamsMachine.path_wall, allow_pickle=True)
    #check that the last element is the neighbor of the first one and not the same one
    if(RZwall[0]==RZwall[-1]).all():
        RZwall = RZwall[:-1]
        print('erased last element of wall')
    #check that the wall coordinates are stocked in a counter clockwise position. If not, reverse it
    Trigo_orientation, signed_area = utility_functions.polygon_orientation(RZwall[:, 0], RZwall[:, 1])
    if Trigo_orientation:
        RZwall = RZwall[::-1]
        print('wall reversed')


    full_wall = load_walls(ParamsMachine, world)
    print('walls loaded, with components {}'.format(full_wall.name))
    #calculate transfert matrix
    return world, RZwall




def load_walls(ParamsMachine, world):
    filename, ext = os.path.splitext(ParamsMachine.path_CAD)
    if ext == '':
        full_wall = read_CAD_from_raw(ParamsMachine, world)
    elif ext == '.ccm': 
        full_wall = read_CAD_from_components(ParamsMachine, world)
    else:
        raise(NameError('cannot read 3D files, extension {ext} unrecognised' ))
        
    return full_wall



def read_CAD_from_raw(ParamsMachine, world):
    #fetch enabled wall components in main components file in Tomography/ressources
    components_list = set(components.keys())

    #load names of all components stored in the raw files
    features_files = os.listdir(ParamsMachine.path_CAD)
    features = [os.path.splitext(features)[0] for features in features_files] #remove raw extension from name 
    
    #select only features listed in the main components files
    enabled_features = list(set(features) & components_list)
    ext_map = {os.path.splitext(f)[0]: os.path.splitext(f)[1] for f in features_files}

    extensions = [ext_map[name] for name in enabled_features]
    path_enabled_features = [ParamsMachine.path_CAD + features + extensions[i] for i, features in enumerate(enabled_features)] #full path for enabled components

    #set up Node for better hierarchy of wall components (relevant for raytracing computing memory)
    wall_group = Node(parent=world)

    #get extension type for file
    if ParamsMachine.name_material == "absorbing_surface":
        for i, f in enumerate(path_enabled_features):
            if extensions[i] == '.ply':
                full_wall =  import_ply(f, parent = wall_group , material = AbsorbingSurface(), name = enabled_features[i]) 
            elif extensions[i] == '.stl':
                full_wall =  import_stl(f, parent = wall_group , material = AbsorbingSurface(), name = enabled_features[i]) 
            elif extensions[i] == '.npy':
                RZ = np.load(f)
                full_wall = axisymmetric_mesh_from_polygon(RZ)
                full_wall.parent = wall_group
                full_wall.material = AbsorbingSurface()
                full_wall.name = enabled_features[i]
            else:
                raise(NameError('unrecognised 3D files extension'))
    else:
        # ParamsMachine.name_material should be the path to a components file containing each type of the enabled components
        wall_materials = load_components(ParamsMachine.name_material, enabled_features)
        for i, f in enumerate(path_enabled_features):
            if extensions[i] == '.ply':
                full_wall =  import_ply(f, parent = wall_group , material = wall_materials[i], name = enabled_features[i]) 
            elif extensions[i] == '.stl':
                full_wall =  import_stl(f, parent = wall_group , material = wall_materials[i], name = enabled_features[i]) 
            elif extensions[i] == '.npy':
                    RZ = np.load(f)
                    full_wall = axisymmetric_mesh_from_polygon(RZ)
                    full_wall.parent = wall_group
                    full_wall.material = wall_materials[i]
                    full_wall.name = enabled_features[i]
            else:
                raise(NameError('unrecognised 3D files extension'))
    return full_wall




def read_CAD_from_components(ParamsMachine, world):
    import calcam
    CAD = calcam.CADModel(ParamsMachine.path_CAD, model_variant = ParamsMachine.variant_CAD)
    features = CAD.get_enabled_features()

    #read the file Tomography/ressources/components.yaml, listing each elements of the CAD model that need to be enabled.
    components_list = set(components.keys())
    enabled_features = list(components_list & set(features))
    print(features)
    CAD.enable_only(enabled_features)
    path_stl = [CAD.features[feature].filename for feature in enabled_features] 

    #create node to load all meshes for more efficient loading
    wall_group = Node(parent=world)

    #load each enabled features, depending on the type of materials
    if ParamsMachine.name_material == "absorbing_surface":
        # meshes in the CAD model on West are oriented (up direction) in the +Y direction, they need to be rotated.
        if ParamsMachine.machine == 'WEST':
            full_wall =  [import_stl(f, parent = wall_group, scaling = 0.001 , material = AbsorbingSurface(), name = features[i], transform = rotate(0, -90,0)) for i, f in enumerate(path_stl)]
        # meshes in the CAD model on Compass are oriented (up direction) in the +Z direction, no need for rotation
        elif ParamsMachine.machine == 'COMPASS':
            full_wall =  [import_stl(f, parent = wall_group, scaling = 0.001 , material = AbsorbingSurface(), name = features[i]) for i, f in enumerate(path_stl)]
        else:
            raise(NameError('unrecognized machine'))
    else:
        # Look into the file (path in ParamsMachine.name_material) to assign material type to each enabled features
        wall_materials = load_components(ParamsMachine.name_material, enabled_features)

        # meshes in the CAD model on West are oriented (up direction) in the +Y direction, they need to be rotated.
        if ParamsMachine.machine == 'WEST':
            full_wall =  [import_stl(f, parent = wall_group, scaling = 0.001 , material = wall_materials[i], name = features[i],transform = rotate(0, -90,0)) for i, f in enumerate(path_stl)]
        # meshes in the CAD model on Compass are oriented (up direction) in the +Z direction, no need for rotation
        elif ParamsMachine.machine == 'COMPASS':
            full_wall =  [import_stl(f, parent = wall_group, scaling = 0.001 , material = wall_materials[i], name = features[i]) for i, f in enumerate(path_stl)]
        else:
            raise(NameError('unrecognized machine'))
    CAD.unload()
    return full_wall



def load_components(namefile, features):
    import yaml
    instances = []
    namefile = namefile if namefile.endswith(".yaml") else namefile + ".yaml"
    with open(namefile, "r") as f:
        config = yaml.safe_load(f)

    components_cfg = config["components"]
    for name in features:
        comp_cfg = components_cfg.get(name)
        if comp_cfg is None:
            raise ValueError(f"Component '{name}' not found in YAML.")

        cls_name = comp_cfg["class"]
        params = comp_cfg.get("params", {})
        
        # Assume class is already imported
        cls = globals()[cls_name]
        instance = cls(**params)
        instances.append(instance) 
    
    return instances


def fit_size_all(camera, mask, vid, param_fit = None):
    if param_fit == 'mask':
        target_shape = mask.shape
    elif param_fit == 'camera':
        target_shape = camera.pixel_directions.shape
    elif param_fit == 'vid':
        target_shape = vid.shape[1:]
    else:
        if mask.shape != camera.pixel_directions.shape or mask.shape !=vid.shape[1:]:
            raise Exception('careful, discrepancy in elements shape')
        else:
            return camera, mask, vid
    mask = resize_matrix(mask, target_shape)
    pixel_directions = resize_matrix(camera.pixel_directions, target_shape)
    pixel_origins = resize_matrix(camera.pixel_origins, target_shape)
    camera = VectorCamera(pixel_origins, pixel_directions)
    vidnew = np.zeros((vid.shape[0], target_shape[0], target_shape[1]))
    for i in range(vid.shape[0]):
        vidnew[i, :, :] = resize_matrix(vid[i, :, :], target_shape)
    return camera, mask, vidnew 

def fit_size_vid(vid, fit_shape):
    vidnew = np.zeros((vid.shape[0], fit_shape[0], fit_shape[1]))
    for i in range(vid.shape[0]):
        vidnew[i, :, :] = resize_matrix(vid[i, :, :], fit_shape)
    return vid 




def reduce_vid_precision(vid, decimation =1):
    decimation = decimation or 1
    # mask = mask[::decimation, ::decimation]
    image_2D = vid[0, :, :] 
    image_2D = downsample_with_avg(image_2D, decimation)
    vid_downsize = np.zeros((vid.shape[0], image_2D.shape[0], image_2D.shape[1]))
    for i in range(vid.shape[0]):
        vid_downsize[i, :, :] = downsample_with_avg(vid[i, :, :] , decimation)


    return  vid_downsize




def reconstruct_2d(
    ds,
    var,
    t_inv=0,
    index_dim="pixel",
    row_coord=None,
    col_coord=None,
    shape_attr=None,
    fill_value=np.nan,
):
    """
    Reconstruct a 2D image from (time, index) stored data.
    """

    # Infer coordinate names automatically
    if row_coord is None:
        row_coord = f"row_{index_dim}"
    if col_coord is None:
        col_coord = f"col_{index_dim}"

    if shape_attr is None:
        shape_attr = f"{index_dim}_shape"

    # Extract data
    
    values = ds[var].sel(t_inv=t_inv, method = 'nearest').values
    rows = ds[row_coord].values
    cols = ds[col_coord].values
    shape = ds.attrs[shape_attr]

    # Allocate full image
    img = np.full(shape, fill_value, dtype=values.dtype)

    # Scatter back to 2D
    img[rows, cols] = values

    return img



@xr.register_dataset_accessor("image2d")
class Image2DAccessor:
    def __init__(self, ds):
        self._ds = ds

    def reconstruct(
        self,
        var,
        t_inv=0,
        index_dim="pixel",
        fill_value=np.nan,
    ):
        ds = self._ds

        row_coord = f"row_{index_dim}"
        col_coord = f"col_{index_dim}"
        shape_attr = f"mask_{index_dim}_shape"

        if shape_attr not in ds.attrs:
            raise KeyError(f"Dataset missing attribute '{shape_attr}'")

        data = ds[var].sel(t_inv=t_inv, method = 'nearest').values
        rows = ds[row_coord].values
        cols = ds[col_coord].values
        shape = ds.attrs[shape_attr]

        img = np.full(shape, fill_value, dtype=data.dtype)
        img[rows, cols] = data

        return img

    def plot(
        self,
        var,
        t_inv=0,
        index_dim="pixel",
        ax=None,
        **imshow_kwargs,
    ):
        img = self.reconstruct(var, t_inv, index_dim)

        if ax is None:
            fig, ax = plt.subplots()
        orientation_attr = f"{index_dim}_image_orientation"
        if orientation_attr == "swapaxes, flip Y axis":
            im = np.swapaxes(im, 1, 2)
            im = np.flip(im, 1)
        elif orientation_attr == "swapaxes":
            im = np.swapaxes(im, 1, 2)

        if index_dim=="pixel":
            im = ax.imshow(img, origin="lower", **imshow_kwargs)
        elif index_dim=="node":
            extent = [np.min(self._ds.cell_r), np.max(self._ds.cell_r), np.min(self._ds.cell_z), np.max(self._ds.cell_z)] 
            im = ax.imshow(img, origin="lower", extent = extent, **imshow_kwargs)
        ax.set_title(f"{var} (t={self._ds.sel(t_inv=t_inv, method = 'nearest').t_inv.values:.3f}s)")

        units = self._ds[var].attrs.get("units", "")
        if units:
            plt.colorbar(im, ax=ax, label=units)
        else:
            plt.colorbar(im, ax=ax)

        return ax