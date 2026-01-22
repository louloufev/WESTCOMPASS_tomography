from pathlib import Path
import os
import json
import xarray as xr
from Tomography.core import result_inversion, fonction_tomo, utility_functions
import pandas as pd
import pdb
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
        Path("/Home/LF285735/Documents/Python/mnt/nunki/camera_rapide/Images CCD rapide/Tomographic_Inversion/"),
        Path("/Home/LF285735/Documents/Python/mnt/nunki/camera_rapide/Images CCD rapide/"),   # NFS mount (read-only)
        Path("/Home/LF285735/Zone_Travail/")
    ]

    # Single writable location (Linux local/server)
    WRITE_ROOT = Path("/Home/LF285735/Documents/Python/WESTCOMPASS_tomography/")


# Subdirectories
RT_SUBDIR = "raytracing"
INV_SUBDIR = "inversion"
INV_MATRIX_SUBDIR = "inversion_matrix"
TREATED_VIDEOS_SUBDIR = "treated_videos"
DENOISING_SUBDIR = "denoising"


SUBDIRS = [RT_SUBDIR, INV_SUBDIR, TREATED_VIDEOS_SUBDIR, DENOISING_SUBDIR]

date = datetime.now().isoformat()
version = 1.2

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




def parse_subdir(subdir: str):
    # Look for file in each root folders. returns path if it exists
    paths = []
    # First check writable root (most recent data)
    path = WRITE_ROOT / subdir 
    if path.exists():
        paths.append(path)

    # Then check read-only roots
    for root in READ_ROOTS:
        path = root / subdir 
        if path.exists():
            paths.append(path)

    return paths

def create_index_inversion(excel_path):
    directories = parse_subdir('inversion')

    collect_zarr_metadata_to_excel(directories,
    'ParamsInversion',
    excel_path,
    sheet_name="index")





def collect_zarr_metadata_to_excel(
    directories,
    excel_path,
    
    stage_attr="stage",
    sheets=("raytracing", "treatment videos", "inversion", "denoising"),
):
    excel_path = Path(excel_path)

    # Load existing Excel sheets (if file exists)
    existing = {}
    if excel_path.exists():
        with pd.ExcelFile(excel_path) as xls:
            for sheet in sheets:
                if sheet in xls.sheet_names:
                    existing[sheet] = pd.read_excel(xls, sheet)
                else:
                    existing[sheet] = pd.DataFrame()
    else:
        existing = {sheet: pd.DataFrame() for sheet in sheets}

    # Track existing zarr names per sheet
    existing_names = {
        sheet: set(df["zarr_name"]) if "zarr_name" in df else set()
        for sheet, df in existing.items()
    }

    new_records = {sheet: [] for sheet in sheets}

    # Scan directories
    for base_dir in directories:
        base_dir = Path(base_dir)
        if not base_dir.exists():
            continue

        for zarr_path in base_dir.rglob("*.zarr"):
            zarr_name = zarr_path.name

            try:
                ds = xr.open_zarr(zarr_path, consolidated=False)
            except Exception as e:
                print(f"[WARN] Cannot open {zarr_path}: {e}")
                continue

            stage = ds.attrs.get(stage_attr)
            if stage not in sheets:
                continue

            if zarr_name in existing_names[stage]:
                continue  # already indexed
            if stage == "raytracing":
                params_machine = ds.attrs.get(
                    "ParamsMachine"
                )
                params_grid = ds.attrs.get(
                    "ParamsGrid"
                )
                try:
                    params = {**params_machine, **params_grid}
                except:
                    print(f"[WARN] {zarr_path} missing params dict")
                    continue
            elif stage == "inversion":
                params = ds.attrs.get(
                    "ParamsInversion" 
                )
            elif stage == "denoising":
                params = ds.attrs.get(
                    "ParamsDenoising" 
                )
            elif stage == "treatment videos":
                params = ds.attrs.get(
                    "ParamsVideo" 
                )

            if not isinstance(params, dict):
                print(f"[WARN] {zarr_path} missing params dict")
                continue

            row = {
                "zarr_name": zarr_name,
                "zarr_path": str(zarr_path),
            }
            row.update(params)
            new_records[stage].append(row)

    # Write back to Excel
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
        for sheet in sheets:
            old_df = existing[sheet]
            new_df = pd.DataFrame(new_records[sheet])

            if not new_df.empty:
                combined = pd.concat([old_df, new_df], ignore_index=True)
            else:
                combined = old_df

            combined.to_excel(writer, sheet_name=sheet, index=False)

    print(f"Zarr metadata index updated: {excel_path}")




def cleanup_old_version_files(directory, max_version, dry_run=True):
    #scan all zarr files in directory, returning all those with version < max_version
    # set dry_run to false to effectively remove those files
    #  
    paths = []
    if not dry_run:
        Warning(f"careful all file with version strictly lower than {max_version} will be removed ! Continue ? [Y/n]")
        remove_file_flag = input("type Y to continue with removal")
        if remove_file_flag!= "Y":
            print("aborting removal")
            return paths
        else:
            print("proceeding with file removal")
    
    for path in directory.glob("*.zarr"):
        
        try:
            ds = xr.open_zarr(path, consolidated=False)
        except Exception as e:
            print(f"[WARN] Cannot open {path}: {e}")
            continue
        version_file = ds.version
        if version_file < max_version:
            if dry_run:
                print("[DRY RUN] Would remove:", path)
                paths.append(path)
            else:
                print("Removing:", path)
                __import__("shutil").rmtree(path)
                paths.append(path)
    return paths

def scan_zarr_for_index(excel_path = "index.xlsx"):
    for subdir in SUBDIRS:
        directories = parse_subdir(subdir)
        collect_zarr_metadata_to_excel(
            directories,
            excel_path)
            