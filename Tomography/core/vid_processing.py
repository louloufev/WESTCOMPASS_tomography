filter_median = 10
from scipy import ndimage
vid_med = vid - ndimage.median_filter(vid, size=(filter_median,1,1), mode = 'nearest')
inversion_results_full_med = inversion_results_full - ndimage.median_filter(inversion_results_full, size=(filter_median,1,1), mode = 'nearest')
images_retrofit_full_med = images_retrofit_full - ndimage.median_filter(images_retrofit_full, size=(filter_median,1,1), mode = 'nearest')
