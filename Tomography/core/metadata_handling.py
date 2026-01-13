from pathlib import Path
import os
from . import fonction_tomo 
import json
import hashlib
import xarray as xr
from . import result_inversion
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


def find_existing(subdir: str, name: str) -> Path | None:
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
    try:
        WRITE_ROOT.mkdir(parents=True, exist_ok=True)
        test = WRITE_ROOT / ".write_test"
        test.touch()
        test.unlink()
        return WRITE_ROOT
    except Exception:
        raise RuntimeError("No writable data location found.")

def ensure_write_dirs():
    (WRITE_ROOT / RT_SUBDIR).mkdir(parents=True, exist_ok=True)
    (WRITE_ROOT / INV_SUBDIR).mkdir(parents=True, exist_ok=True)



def get_or_create_raytracing(ParamsMachine, ParamsGrid) -> Path:
    ensure_write_dirs()

    machine_hash = hash_params(ParamsMachine)
    grid_hash = hash_params(ParamsGrid)
    rt_hash = machine_hash+grid_hash
    name = f"rt_{rt_hash}.zarr"

    existing = find_existing(RT_SUBDIR, name)
    if existing:
        return existing

    # Must be computed locally
    path = WRITE_ROOT / RT_SUBDIR / name

    ds = fonction_tomo.compute_raytracing(ParamsMachine, ParamsGrid)
    ds.attrs.update({
        "stage": "raytracing",
        "hash": rt_hash,
        "storage": "linux",
    })

    ds.to_zarr(path, mode="w")
    return path


def get_or_create_inversion(ParamsMachine, ParamsGrid, ParamsVid) -> Path:
    ensure_write_dirs()

    rt_path = get_or_create_raytracing(ParamsMachine, ParamsGrid)
    rt_ds = xr.open_zarr(rt_path, chunks="auto")

    combined = {
        "raytracing_hash": rt_ds.attrs["hash"],
        **ParamsVid.to_dict(),
    }
    inv_hash = hash_params(combined)
    name = f"inv_{inv_hash}.zarr"

    existing = find_existing(INV_SUBDIR, name)
    if existing:
        return existing

    path = WRITE_ROOT / INV_SUBDIR / name

    inv_ds = fonction_tomo.compute_inversion(ParamsMachine, ParamsGrid, ParamsVid)
    inv_ds.attrs.update({
        "stage": "inversion",
        "hash": inv_hash,
        "raytracing_hash": rt_ds.attrs["hash"],
        "storage": "linux",
    })

    inv_ds.to_zarr(path, mode="w")
    return path









def hash_params(params, length=10) -> str:
    if isinstance(params, result_inversion.Params):
        params = params.to_dict()
    norm = normalize_params(params)
    payload = json.dumps(norm, sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:length]


def normalize_params(obj):
    if isinstance(obj, dict):
        return {k: normalize_params(v) for k, v in sorted(obj.items())}
    elif isinstance(obj, (list, tuple)):
        return [normalize_params(v) for v in obj]
    elif hasattr(obj, "__dict__"):
        return normalize_params(obj.__dict__)
    else:
        return obj
    
