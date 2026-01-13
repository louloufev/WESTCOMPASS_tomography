# quick script to load 2D walls with the tofu module. 
# this script then plots some elements of the wall, and prompts the user to draw polygons to remove undesired points until the structure is simplified enough


import matplotlib.pyplot as plt
import numpy as np
plt.ion()
import tofu as tf
conf = tf.load_config('WEST')
from Tomography.core import utility_functions

# conf.plot() #to plot everything

# Pour voir la liste des polygones constitutifs
conf

# Pour acceder au polygones
name_wall_elements = 'DivUpV3' 
R, Z = getattr(conf.PFC, name_wall_elements).Poly #get the 2d Contour of the walls


f, ax = plt.subplots()

remove_all_points = 0
# RZ = np.array([R, Z])
# RZ = RZ.T
RZ = out
R = RZ[:, 0]
Z = RZ[:, 1]
while remove_all_points == 0:
    plt.plot(R, Z)
    points = utility_functions.draw_polygon(f, ax)
    print('wait', points)

    RZ = utility_functions.clean_points_inside_poly(points, RZ)
    R = RZ[:, 0]
    Z = RZ[:, 1]
    plt.plot(R, Z, 'r')
    remove_all_points = int(input('enter 1 if all points removed, 0 else'))

np.save(name_wall_elements + '.npy', RZ)