import zipfile

with zipfile.ZipFile('/Home/LF276573/Zone_Travail/Python/CHERAB/videos/west/61023 steady_C001H001S0001/61023 steady_C001H001S0001.npz', 'r') as zf:
    print(zf.namelist())  # see included files
    zf.extractall('tmp_folder')  # try extracting