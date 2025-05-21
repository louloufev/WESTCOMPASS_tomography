
echo "Starting setup"
mamba env create -n base -f ../environment/main_WEST.yaml
mamba env create -n inversion_env -f ../environment/inversion_env.yaml

echo "done creating environments, launching python"
mamba activate base
python


