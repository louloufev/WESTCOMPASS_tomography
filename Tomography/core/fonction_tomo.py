import numpy as np
import matplotlib.pyplot as plt
import pdb
import sys
import time
import xarray as xr




from . import utility_functions, result_inversion








def compute_raytracing(ParamsMachine, ParamsGrid):

    transfert_matrix, mask_node, mask_pixel, node, pixel, rows_node, cols_node, rows_pixel, cols_pixel, cell_r, cell_z = get_transfert_matrix()


    rt_ds = xr.Dataset(
    data_vars={
        "transfert_matrix": (
            ("pixel", "node"),
            transfert_matrix,
            {"units": "m"}
        ),
        "mask_pixel": (
            ("pixel",),
            mask_pixel,   # All pixel stored are valid
            {"description": "Mask of valid pixel"},
        )
        "mask_node": (
            ("node",),
            mask_node,   # All pixel stored are valid
            {"description": "Mask of valid node"},
        )
    },
    coords={
        "pixel": ("pixel", pixel),
        "node": ("node", node),
        "row_pixel": ("pixel", row_pixel),
        "col_pixel": ("pixel", col_pixel),
        "row_node": ("node", row_node),
        "col_node": ("node", col_node),
    },
    attrs={
        "image_shape": mask_pixel.shape,
        "node_shape": mask_node.shape
        "mask_description": "Camera mask applied"
    }

    )

    return rt_ds


def compute_inversion(rt_ds, inv_params):
    


    inv_ds = xr.Dataset()
    inv_ds = inv_ds.assign_coords(rt_ds.coords)
    inv_ds.data_vars={
        "vid": (
            ("pixel", "node"),
            vid,
            {"units": "m"},
        ),
        "mask_pixel": (
            ("pixel",),
            mask_pixel,   # All pixel stored are valid
            {"description": "Mask of valid pixel"},
        )
        "mask_node": (
            ("node",),
            mask_node,   # All pixel stored are valid
            {"description": "Mask of valid node"},
        )
    }
    inv_ds.coords={
        "pixel": ("pixel", pixel),
        "node": ("node", node),
        "row_pixel": ("pixel", row_pixel),
        "col_pixel": ("pixel", col_pixel),
        "row_node": ("node", row_node),
        "col_node": ("node", col_node),
    }
    inv_ds.attrs={
        "image_shape": mask_pixel.shape,
        "node_shape": mask_node.shape
        "mask_description": "Camera mask applied",
    }

    return inv_ds