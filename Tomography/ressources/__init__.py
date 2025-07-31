import os
import yaml

# Path to the YAML file
_path = os.path.join(os.path.dirname(__file__), "folder_paths.yaml")
_components = os.path.join(os.path.dirname(__file__), "components.yaml")
# Load YAML

with open(_path, "r") as f:
    paths = yaml.safe_load(f)

with open(_components, "r") as f:
    components = yaml.safe_load(f)


__all__ = ["paths", "components"]