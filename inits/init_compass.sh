mamba create -n tomography_env
conda activate tomography_env

module load pleque
pip install imageio imageio-ffmpeg
pip install calcam
pip install cherab

# mamba env create -f environment/inversion_env.yaml inversion_env

