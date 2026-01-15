import numpy as np
import xarray as xr

# ---- Shapes ----
Ny_node, Nx_node = 40, 50
Ny_pix, Nx_pix = 480, 640

# ---- Masks (2D) ----
mask_node_2d = np.random.rand(Ny_node, Nx_node) > 0.2
mask_pixel_2d = np.random.rand(Ny_pix, Nx_pix) > 0.3

# ---- Flatten masks ----
node_idx = np.flatnonzero(mask_node_2d)
pix_idx = np.flatnonzero(mask_pixel_2d)

node_y, node_x = np.unravel_index(node_idx, (Ny_node, Nx_node))
pix_y, pix_x = np.unravel_index(pix_idx, (Ny_pix, Nx_pix))

Nn = len(node_idx)
Np = len(pix_idx)
from scipy.sparse import csr_matrix, save_npz, csc_matrix, load_npz, isspmatrix

# ---- Data stored as (node, pixel) ----
data = np.random.rand(Nn, Np)  # replace with real data
data_csr = csr_matrix(data)
# ---- Dataset ----
ds = xr.Dataset(
    data_vars={
        "response": (
            ("node", "pixel"),
            data_csr,
            {"description": "Node-pixel response"}
        ),
        "mask_node": (
            ("node",),
            node_idx
        ),
        "mask_pixel": (
            ("pixel",),
            pix_idx
        )
    },
    coords={
        # Node coordinates
        "node": ("node", np.arange(Nn)),
        "node_y": ("node", node_y),
        "node_x": ("node", node_x),

        # Pixel coordinates
        "pixel": ("pixel", np.arange(Np)),
        "pixel_y": ("pixel", pix_y),
        "pixel_x": ("pixel", pix_x),
    },
    attrs={
        "node_shape": (Ny_node, Nx_node),
        "pixel_shape": (Ny_pix, Nx_pix),
        "storage": "node-pixel flattened with masks",
    }
)

print(ds)


def node_image(ds, pixel_index, fill_value=np.nan):
    Ny, Nx = ds.attrs["node_shape"]
    img = np.full((Ny, Nx), fill_value)

    img[
        ds.node_y,
        ds.node_x
    ] = ds["response"].isel(pixel=pixel_index).values

    return img


def node_image(ds, pixel_index, fill_value=np.nan):
    Ny, Nx = ds.attrs["node_shape"]
    img = np.full((Ny, Nx), fill_value)

    img[
        ds.node_y,
        ds.node_x
    ] = ds["response"].isel(pixel=pixel_index).values

    return img


import matplotlib.pyplot as plt

def show_image(img, title="", cmap="viridis"):
    plt.ion()
    plt.figure(figsize=(5, 5))
    plt.imshow(img, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.ioff()