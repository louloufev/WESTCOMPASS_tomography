
import importlib
import numpy as np
import matplotlib.pyplot as plt
import pdb
from Tomography.core import utility_functions, result_inversion
import importlib
importlib.reload(result_inversion)
importlib.reload(utility_functions)
Inversion_results = result_inversion.Inversion_results()
Inversion_results = Inversion_results.from_file()


Inversion_results.plot_simple(10, vmin = -300, vmax = 300)

Inversion_results.create_video()
