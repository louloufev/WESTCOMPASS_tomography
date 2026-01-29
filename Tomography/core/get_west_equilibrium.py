import imas_west
import sys
import pickle
import json
import numpy as np
from scipy.sparse import isspmatrix

nshot =int(sys.argv[1])
magflux = imas_west.get(nshot, 'equilibrium', 0, 1)


r = magflux.interp2D.r
z = magflux.interp2D.z
rx = magflux.boundary.x_point.r
zx = magflux.boundary.x_point.z
psi = magflux.interp2D.psi
time = magflux.time
psisep = magflux.boundary.psi
outline_r = magflux.boundary.outline.r
outline_z = magflux.boundary.outline.z
mag_dict = dict(r = r, z = z, psi = psi, time = time, psisep = psisep, rx = rx, zx = zx, outline_r = outline_r, outline_z = outline_z)
# print(pickle.dumps(mag_dict))
sys.stdout.buffer.write(pickle.dumps(mag_dict))