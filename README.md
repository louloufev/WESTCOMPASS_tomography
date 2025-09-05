

# 🚀 WESTCOMPASS tomography

Package to handle tomographic inversion of visible camera on either the Compass or WEST tokamak
---

## 📂 Table of Contents
- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Tutorial](#tutorial)

---

## 📖 About


---

## ✨ Features
- ✅ Feature 1
- ✅ Feature 2
- ✅ Feature 3

---

## ⚙️ Installation
```bash
# Clone the repository
git clone https://github.com/louloufev/WESTCOMPASS_tomography.git

# Navigate into the folder
cd WESTCOMPASS_tomography


# Install dependencies
# run in the terminal (change file name to either compass or west)

chmod +x inits/init_compass.sh
inits/init_compass.sh

module load pleque


# after setting up the environment, open python in the functions folder

cd functions
python

# you can now execute script or function from Tomography, there is an exemple script for each machine :
# NB : it might be necessary to change the path to the different ressources used for the inversion (path parameters) 
exec(open('Tomography/core/example_compass.py').read())
```




---

## Tutorial
# Please refer to the example in Tomography/core/ folder for quick tutorial

# the parameters are chosen to run quickly for a quick test, but the paths to the necessary ressources have to be specified by the user (path parameters)

---