'testing script'


from cherab.tools.raytransfer import RayTransferPipeline2D, RayTransferCylinder
from raysect.optical import World, translate, rotate, ConstantSF, Point3D
from cherab.tools.primitives.axisymmetric_mesh import axisymmetric_mesh_from_polygon

from Tomography.core import fonction_tomo
import numpy as np
import pleque.io.compass as plq
import time


nr = 15
nz = 15
eq = plq.cdb(15487, 1000)
cell_r = np.linspace(0.5, 0.7, nr)
cell_z = np.linspace(-0.2, 0, nz)  
RZwall = np.array([eq.first_wall.R,eq.first_wall.Z]).T     

start = time.time()
FL_MATRIX, dPhirad, phi_min, phi_mem = fonction_tomo.FL_lookup(eq, 145, cell_r, cell_z, 100/180*np.pi, 170/180*np.pi, 2e-3, 0.2)
end = time.time()

elapsed = end - start
print(f"Magnetic field lines calculation : {elapsed:.3f} seconds")

wall_limit = axisymmetric_mesh_from_polygon(RZwall)

dict_transfert_matrix = {}
cell_r_precision, cell_z_precision, grid_mask_precision, cell_dr_precision, cell_dz_precision = fonction_tomo.get_mask_from_wall(0.5, 0.7, -0.2, 0, nr, nz, wall_limit, dict_transfert_matrix)
grid_mask_precision = grid_mask_precision>0
plasma = RayTransferCylinder(radius_outer = 0.7, 
                            height= 0.2, 
                            n_radius = nr, 
                            n_height = nz, 
                            radius_inner = 0.5,  
                            transform=translate(0., 0., -0.2), 
                            n_polar = 360, 
                            period = 360)

nphi = 50
ind_phi_closest = np.round((nphi*plasma.material.dphi-phi_min)/(dPhirad*180/np.pi)).astype('int')

# import matplotlib.pyplot as plt

# fig = plt.figure()
# ax = fig.gca(projection='3d')



# for fl in traces:
#     ax.scatter(fl.X, fl.Y, fl.Z, s=0.3, marker='.')
# for fl in lines:
#     ax.scatter(fl.X, fl.Y, fl.Z, s=0.6, marker='.')

# ax.set_aspect('equal')
# ax.set_xlabel('x [m]')
# ax.set_ylabel('y [m]')
# ax.set_zlabel('z [m]')

# fig = plt.figure()
# ax = fig.gca()

# for fl in traces:
#     ax.scatter(fl.R, fl.Z, s=0.3, marker='.')
# for fl in lines:
#     #ax.scatter(fl.R, fl.Z, s=0.3, marker='.')
#     ax.plot(fl.R, fl.Z)

# print(dists)

# eq.first_wall.plot(color='k')
# eq.lcfs.plot(color='y', lw=0.5)

# ax.set_xlabel('R [m]')
# ax.set_ylabel('Z [m]')
# ax.set_aspect('equal')

# fig = plt.figure()
# ax = fig.gca()

# for fl in traces:
#     ax.plot(fl.X, fl.Y)
# for fl in lines:
#     ax.plot(fl.X, fl.Y)

# ax.set_xlabel('X [m]')
# ax.set_ylabel('Y [m]')
# ax.set_aspect('equal')