mamba create -n tomography_env #only once
conda activate tomography_env #every time before launching python


module load pleque #every time before launching python

conda install netCDF4 zarr
pip install imageio imageio-ffmpeg #only once
pip install calcam #only once
pip install cherab #only once

#you can launch python now !

# for heavy calculations, connect to calculation servers, write in terminal 
#ssh username@gpu-titan 
#it does not handle graphics (no plots and interactive prompting), use ltserv for easy plots




