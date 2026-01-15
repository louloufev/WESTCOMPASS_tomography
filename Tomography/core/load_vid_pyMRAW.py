import json
import pyMRAW
import sys
import numpy as np

mraw_file = sys.argv[1]
out_file  = sys.argv[2]

arr, md = pyMRAW.load_video(mraw_file)  # full array
np.save(out_file, arr)
