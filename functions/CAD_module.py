import numpy as np
import trimesh
import os
import pdb

def convert_obj_to_mesh(path_folder):
    os.makedirs(path_folder + '/stl/', exist_ok=True)
    for file in os.listdir(path_folder):

        if (file.endswith('.stl') or file.endswith('.obj')):
            pdb.set_trace()
            mesh = trimesh.load(path_folder+ file)
            file,ext = os.path.splitext(file)
            mesh.export(path_folder + 'stl/' + file + '.stl')


