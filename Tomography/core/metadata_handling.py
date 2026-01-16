from pathlib import Path
import os
import json
import xarray as xr
from Tomography.core import result_inversion, fonction_tomo, utility_functions
from datetime import datetime
if 'compass' in os.getcwd():
# Read-only locations (NFS, Windows server, etc.)
    READ_ROOTS = [
        Path("/compass/home/fevre/WESTCOMPASS_tomography"),
    ]

    # Single writable location (Linux local/server)
    WRITE_ROOT = Path("/compass/home/fevre/WESTCOMPASS_tomography")

else:
    # Read-only locations (NFS, Windows server, etc.)
    READ_ROOTS = [
        Path("/Home/LF285735/Documents/Python/WESTCOMPASS_tomography/"),
        Path("/Home/LF285735/Documents/Python/mnt/nunki/camera_rapide/Images CCD rapide/Tomographic_Inversion"),   # NFS mount (read-only)
    ]

    # Single writable location (Linux local/server)
    WRITE_ROOT = Path("/Home/LF285735/Documents/Python/WESTCOMPASS_tomography/")


# Subdirectories
RT_SUBDIR = "raytracing"
INV_SUBDIR = "inversion"
INV_MATRIX_SUBDIR = "inversion_matrix"
TREATED_VIDEOS_SUBDIR = "treated_videos"
DENOISING_SUBDIR = "denoising"

date = datetime.now().isoformat()
version = 1.0

def find_existing(subdir: str, name: str):
    # Look for file in each root folders. returns path if it exists

    # First check writable root (most recent data)
    path = WRITE_ROOT / subdir / name
    if path.exists():
        return path

    # Then check read-only roots
    for root in READ_ROOTS:
        path = root / subdir / name
        if path.exists():
            return path

    return None

def get_writable_root() -> Path:
    #check if user has the right to write in write_root folder 
    try:
        WRITE_ROOT.mkdir(parents=True, exist_ok=True)
        test = WRITE_ROOT / ".write_test"
        test.touch()
        test.unlink()
        return WRITE_ROOT
    except Exception:
        raise RuntimeError("No writable data location found.")

def ensure_write_dirs():
    # create folders for data saving
    (WRITE_ROOT / RT_SUBDIR).mkdir(parents=True, exist_ok=True)
    (WRITE_ROOT / INV_SUBDIR).mkdir(parents=True, exist_ok=True)
    (WRITE_ROOT / INV_MATRIX_SUBDIR).mkdir(parents=True, exist_ok=True)
    (WRITE_ROOT / TREATED_VIDEOS_SUBDIR).mkdir(parents=True, exist_ok=True)
    (WRITE_ROOT / DENOISING_SUBDIR).mkdir(parents=True, exist_ok=True)
    


def get_or_create_raytracing(ParamsMachine, ParamsGrid) -> Path:
    #return path name of transfert matrix dataset from raytracing. Creates the dataset if it does not exist. 
    ensure_write_dirs()

    #get unique name for given parameters
    machine_hash = utility_functions.hash_params(ParamsMachine.to_dict())
    grid_hash = utility_functions.hash_params(ParamsGrid.to_dict())
    rt_hash = machine_hash+grid_hash
    name = f"rt_{rt_hash}.zarr"

    #check if file is already created, returns its path if yes
    existing = find_existing(RT_SUBDIR, name)
    if existing:
        return existing

    #full path name of dataset
    path = WRITE_ROOT / RT_SUBDIR / name

    #creates dataset
    ds = fonction_tomo.compute_raytracing(ParamsMachine, ParamsGrid)
    ds.attrs.update({
        "stage": "raytracing",
        "hash": rt_hash,
        "storage": "linux",
        "version": version,
        "date": date,
    })
    #save dataset for later use
    ds.to_zarr(path, mode="w")
    return path


def get_or_create_treated_videos(ParamsVideo):
    ensure_write_dirs()

    videos_hash = utility_functions.hash_params(ParamsVideo)
    name = f"vid_{videos_hash}.zarr"

    
    existing = find_existing(TREATED_VIDEOS_SUBDIR, name)
    if existing:
        return existing



    path = WRITE_ROOT / TREATED_VIDEOS_SUBDIR / name
    treated_video_ds = fonction_tomo.treat_videos(ParamsVideo)
    treated_video_ds.attrs.update({
        "stage": "treatment videos",
        "hash": videos_hash,
        "storage": "linux",
        "version": version,
    })
    treated_video_ds.to_zarr(path, mode="w")

    return path







def get_or_create_inversion(ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion) -> Path:
    ensure_write_dirs()

    rt_path = get_or_create_raytracing(ParamsMachine, ParamsGrid)
    rt_ds = xr.open_zarr(rt_path, chunks="auto")


    treated_videos_path= get_or_create_treated_videos(ParamsVideo)
    treated_video_ds = xr.open_zarr(treated_videos_path, chunks="auto")



    inv_hash = utility_functions.hash_params(ParamsInversion.to_dict())
    inv_hash = utility_functions.hash_params(rt_ds.attrs["hash"] + treated_video_ds.attrs["hash"] + inv_hash)
    name = f"inv_{inv_hash}.zarr"

    existing = find_existing(INV_SUBDIR, name)
    if existing:
        return existing

    path = WRITE_ROOT / INV_SUBDIR / name

    inv_ds = fonction_tomo.compute_inversion(rt_ds, treated_video_ds, ParamsInversion)
    inv_ds.attrs.update({
        "stage": "inversion",
        "hash": inv_hash,
        "raytracing_hash": rt_ds.attrs["hash"],
        "storage": "linux",
        "date":date,
        "version": version,
    })

    inv_ds.to_zarr(path, mode="w")
    return path





def get_or_create_denoising(ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion, ParamsDenoising) -> Path:
    ensure_write_dirs()
    inv_path = get_or_create_inversion(ParamsMachine, ParamsGrid, ParamsVideo, ParamsInversion)
    inv_ds = xr.open_zarr(inv_path, chunks="auto")

    denoising_hash = utility_functions.hash_params(ParamsDenoising.to_dict())
    denoising_hash = inv_ds.attrs["hash"] + denoising_hash
    name = f"inv_{denoising_hash}.zarr"

    existing = find_existing(DENOISING_SUBDIR, name)
    if existing:
        return existing

    path = WRITE_ROOT / DENOISING_SUBDIR / name

    treated_videos_path= get_or_create_treated_videos(ParamsVideo)
    treated_video_ds = xr.open_zarr(treated_videos_path, chunks="auto")


    denoising_ds = fonction_tomo.compute_denoising(inv_ds, treated_video_ds, ParamsDenoising)
    denoising_ds.attrs.update({
        "stage": "denoising",
        "hash": denoising_hash,
        "inversion_hash": inv_ds.attrs["hash"],
        "storage": "linux",
        "date" :date,
        "version": version,
    })

    denoising_ds.to_zarr(path, mode="w")
    return path
