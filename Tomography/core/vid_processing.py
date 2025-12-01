
import importlib
import numpy as np
import matplotlib.pyplot as plt
import pdb
from Tomography.core import utility_functions, result_inversion
import importlib
importlib.reload(result_inversion) #useful when debugging and modyfing code.
importlib.reload(utility_functions)

######load data 
#if you have parameters already inputted
# Inversion_results = full_inversion_toroidal(ParamsMachine,ParamsGrid, ParamsVid)  

#from a file name
Inversion_results = result_inversion.Inversion_results() #create an empty class instance
Inversion_results = Inversion_results.from_file() #use its built in method for loading its data. Prompts you to choose the file
#Inversion_results = Inversion_results.from_file(filename) if you already have the name

#####plot results
frame_number = 197 #choose a frame in the video to plot. 
Inversion_results.plot_simple(frame_number, vmin = -300, vmax = 300) #simple plot of inversion results + retrofit

Inversion_results.denoising()
Inversion_results.plot_bob(frame_number, vmin = -300, vmax = 300) #plot of inversion results (raw, denoised, normalized, denoised + normalized)

Inversion_results.create_video() #create video of the inversion results 
Inversion_results.create_video_holes() #create video of the inversion results keeping only negative values (reindexed on [0, 255])
Inversion_results.create_video_peaks() #create video of the inversion results keeping only positive values (reindexed on [0, 255])


#change inversion method
Inversion_results.ParamsVid.inversion_parameter['min_visibility_node'] = 1
Inversion_results.redo_inversion_results() #recalculate inversion
Inversion_results.denoising() #recalculate denoising (new parameters, need to redo also denoising)
Inversion_results.plot_bob(frame_number)
plt.savefig('new inversion method.png')
plt.close()

#change video parameters

Inversion_results.ParamsVid.dict_vid['sigma'] = 4
Inversion_results.redo_video()
Inversion_results.redo_inversion_results() #recalculate inversion

Inversion_results.denoising() #recalculate denoising
Inversion_results.plot_bob(frame_number)

plt.savefig('higher gaussian filter.png')
plt.close()
