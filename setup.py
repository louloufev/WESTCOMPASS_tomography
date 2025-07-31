import os.path as path
from setuptools import setup, find_packages

setup(
packages=find_packages(),
    install_requires=[
        ],
)

setup(
        name="Tomography",
    version="0.1.0",
    author="Louis Fevre",
    author_email="louis.fevre@cea.fr",
    description="Tomography package with auto-import and reload (Python 3.5 compatible)",
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.5',
)