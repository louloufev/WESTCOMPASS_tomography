"""
Suite of tools for RIS
"""

from pyCDB import client
cdb = client.CDBClient()
import numpy as np
import psutil

def get_info(shot_number, RIS = 3, origin = 'RAW'):
    """
    Fetch the camera parameters

    Parameters
    ----------
    shot_number : integer
        Shot number for which the parameters should be fetched
    RIS : integer between 1 and 4
        Select the camera for which the parameters should be fetched
    origin : String
        RAW or VIDEO, in case one of them is missing in the database.

    Returns
    ---------
    sr : Dictionnary
        Contains the camera parameters
    """

    id_ref = 'RIS.RISEye_' + str(RIS) + '.' + str(origin) + '/RIS:' + str(shot_number)

    try:
        sr = cdb.get_signal_parameters(id_ref)
    except:
        print('Probably no data for RIS' + str(RIS) + '. You may try to change the source by providing a different "origin" parameter.')

    return sr

def get_time(shot_number, RIS = 3, frame = None, origin = 'RAW', convention = 'MIDDLE'):
    """
    Fetch the time trace of RIS camera in second. Can provide the particular time for one particular frame.

    Parameters
    ----------
    shot_number : integer
        Shot number for which the time should be fetched
    RIS : integer between 1 and 4
        Select the camera for which the time should be fetched
    frame : integer
        Select the frame for which the time should be calculated. If None is provided, returns the whole time trace.
    origin : String
        RAW or VIDEO, in case one of them is missing in the database.
    convention : String MIDDLE, BEGIN, END
        Change the convention for calculating the time stamp.
    Returns
    ---------
    time_stamp : time stamp of the camera signal in second (s)
    """

    sr = get_info(shot_number, RIS, origin)

    frame_rate = sr.daq_parameters.FrameRate # Hz
    trigger_time = sr.daq_parameters.TriggerTime + sr.daq_parameters.TriggerDelay # s

    if frame is None:
        frame_max = sr.daq_parameters.Images
        frame_stamp = np.arange(0,frame_max)
    else:
        frame_stamp = frame

    if convention == 'MIDDLE':
        shift = 0.5
    if convention == 'BEGIN':
        shift = 0
    elif convention == 'END':
        shift = 1

    time_stamp = trigger_time + (frame_stamp + shift) / frame_rate # s

    return time_stamp #s

def time_to_frame(shot_number, time, RIS = 3, origin = 'RAW', convention = 'MIDDLE'):
    """
    Convert time to frame number.

    Parameters
    ----------
    shot_number : integer
        Shot number for which the parameters should be fetched
    time: number
        Can be in s or in ms
    RIS : integer between 1 and 4
        Select the camera for which the parameters should be fetched
    origin : String
        RAW or VIDEO, in case one of them is missing in the database.
    convention : String MIDDLE, BEGIN, END
        Change the convention for calculating the time stamp.

    Returns
    ---------
    frame : number (not necessarely an integer)
        Frame number corresponding to the input time. It may not be an integer
    """

    sr = get_info(shot_number, RIS, origin)

    frame_rate = sr.daq_parameters.FrameRate # Hz
    trigger_time = sr.daq_parameters.TriggerTime + sr.daq_parameters.TriggerDelay # s

    if time >= 900:
        #Mean that time is in ms
        time = time / 1000 # s

    if convention == 'MIDDLE':
        shift = 0.5
    if convention == 'BEGIN':
        shift = 0
    elif convention == 'END':
        shift = 1

    frame = (time - trigger_time) * frame_rate - shift

    return frame

def get_resolution(shot_number, RIS = 3, origin = 'RAW'):
    """
    Fetch the camera parameters

    Parameters
    ----------
    shot_number : integer
        Shot number for which the parameters should be fetched
    RIS : integer between 1 and 4
        Select the camera for which the parameters should be fetched
    origin : String
        RAW or VIDEO, in case one of them is missing in the database.

    Returns
    ---------
    resolution : tuple?
        Contains the camera resolution (height, width)
    """
    sr = get_info(shot_number, RIS, origin)

    resolution = (sr.daq_parameters.FrameH, sr.daq_parameters.FrameW)

    return resolution

def load(shot_number, image_fetch = None, RIS = 3, origin = 'RAW', stamp = 'frame', convention = 'MIDDLE', threshold = 0.9, pixel_selection = None):
    """
    Fetch the camera data. The provided camera positions will be automatically sorted by ascending order

    Parameters
    ----------
    shot_number : integer
        Shot number for which the data should be fetched
    image_fetch : Number or array
        Frames or times at which the data should be fetched. If None, will fetch all the data
    RIS : integer between 1 and 4
        Select the camera for which the data should be fetched
    origin : String
        RAW or VIDEO, in case one of them is missing in the database.
    stamp : String 'frame' or 'time'
        Frame or time can be specified, thanks to this option.
    convention : String MIDDLE, BEGIN, END
        Change the convention for calculating the time stamp.
    threshold : number between 0 and 1
        Percentage of the memory that should be available for flag to be False
    pixel_selection : (height_min, height_max, width_min, width_max)

    Returns
    ---------
    video :
        Camera data for the required frame period (time, height, width)
    frame_bounds :
        Boundaries of the frame stamp
    """
    
    flag = check_in_resolution(shot_number, pixel_selection, RIS, origin)
    if flag == False:
        return
    if image_fetch is None:
        sr = get_info(shot_number, RIS = RIS, origin = origin)
        image_fetch = np.arange(0, sr.daq_parameters.Images)
        stamp = 'frame'
    flag, memory_required, available_memory = check_memory(shot_number, image_fetch, RIS = RIS, origin = origin,
                                                           threshold = threshold, stamp = stamp,
                                                           convention = 'MIDDLE', pixel_selection = pixel_selection)
    flag = 0
    if flag == True:
        return

    image_initial = np.min(image_fetch)
    image_final = np.max(image_fetch)

    if stamp == 'time':
        image_initial = np.uint64( np.round( time_to_frame(shot_number, image_initial, RIS, origin, convention) ) )
        image_final = np.uint64( np.round( time_to_frame(shot_number, image_final, RIS, origin, convention) ) )

    if image_initial == image_final:
        image_final = image_initial +1

    try:
        id_ref = 'RIS.RISEye_' + str(RIS) + '.' + str(origin) + '/RIS:' + str(shot_number)
        if pixel_selection is None:
            video = cdb.get_signal(id_ref + '['+str(image_initial)+':'+str(image_final)+',:,:]')
        else:
            pixel_txt = str(pixel_selection[0]) + ':' + str(pixel_selection[1]) + ',' + str(pixel_selection[2]) + ':' + str(pixel_selection[3])
            video = cdb.get_signal(id_ref + '['+str(image_initial)+':'+str(image_final)+',' + pixel_txt +']')
    except:
        print('There was a problem in fetching the data. Checkout image initial and final:')
        print(str(image_initial))
        print(str(image_final))
        print(image_fetch)
        return

    frame_bounds = (image_initial, image_final)

    return video, frame_bounds

def check_memory(shot_number, image_fetch, RIS = 3, origin = 'RAW', threshold = 0.9, stamp = 'frame', convention = 'MIDDLE', pixel_selection = None):
    """
    Fetch the camera data. The provided camera positions will be automatically sorted by ascending order

    Parameters
    ----------
    shot_number : integer
        Shot number for which the data should be fetched
    image_fetch : Number or array
        Frames or times at which the data should be fetched.
    RIS : integer between 1 and 4
        Select the camera for which the data should be fetched
    origin : String
        RAW or VIDEO, in case one of them is missing in the database.
    threshold : number between 0 and 1
        Percentage of the memory that should be available for flag to be False
    stamp : String 'frame' or 'time'
        Frame or time can be specified, thanks to this option.
    convention : String MIDDLE, BEGIN, END
        Change the convention for calculating the time stamp.
    pixel_selection : (height_min, height_max, width_min, width_max)

    Returns
    ---------
    flag : Bool
        True if required memory exceed threshold*available_memory. False otherwise
    memory_required : number
        Memory consumption according to input parameters. In octet
    available_memory : Number
        Available memory
    """
    flag = check_in_resolution(shot_number, pixel_selection, RIS, origin)
    if flag == False:
        return None, None, None

    if image_fetch is not None:
        sr = get_info(shot_number, RIS = RIS, origin = origin)
        image_fetch = np.arange(0, sr.daq_parameters.Images)

    image_initial = np.min(image_fetch)
    image_final = np.max(image_fetch)

    if stamp == 'time':
        image_initial = np.uint64( time_to_frame(shot_number, image_initial, RIS, origin, convention) )
        image_final = np.uint64( time_to_frame(shot_number, image_final, RIS, origin, convention) )

    image_fetch = np.arange(image_initial, image_final)

    if pixel_selection is None:
        resolution = get_resolution(shot_number, RIS, origin)
        nber_pixels = resolution[0]*resolution[1]
    else:
        nber_pixels = (pixel_selection[1]-pixel_selection[0]) * (pixel_selection[3]-pixel_selection[2])

    if origin == 'RAW':
        if RIS == 1 or RIS == 2 or RIS == 3 or RIS == 4:
            bits = 16.
        else:
            print('Camera not implemented')
            return
    if origin == 'VIDEO':
        bits = 8.

    memory_required = np.shape(image_fetch)[0] * nber_pixels * bits / 8 # octet
    available_memory = psutil.virtual_memory()[1]
    flag = memory_required > threshold*available_memory
    if flag == True:
        print('Required memory ' + str(memory_required) + ' would exceed ' + str(threshold) + '*(available one) =' + str(available_memory) + '.')

    return flag, memory_required, available_memory

def check_in_resolution(shot_number, pixel_selection, RIS = 3, origin = 'RAW'):
    """
    Fetch the camera data. The provided camera positions will be automatically sorted by ascending order

    Parameters
    ----------
    shot_number : integer
        Shot number for which the data should be fetched
    pixel_selection :
        (height_min, height_max, width_min, width_max)
    RIS : integer between 1 and 4
        Select the camera for which the data should be fetched
    origin : String
        RAW or VIDEO, in case one of them is missing in the database.

    Returns
    ---------
    flag : Bool
        True if fit the resolution of the camera, False otherwise
    """
    resolution = get_resolution(shot_number, RIS = RIS, origin = origin)

    flag = True
    if pixel_selection is not None:
        if pixel_selection[1]>resolution[0] or pixel_selection[3]>resolution[1]:
            print('Problem in the pixel index chosen')
            print('pixel_selection: ' + str(pixel_selection))
            print('resolution: ' + str(resolution))
            flag = False

    return flag

def process_MINIUX(video, camera_selection):
    # This works on MINI UX data after rotation, i.e. on the raw data

    ##############
    # DISRUPTION #
    ##############
    # Rotate picture
    video.data = np.flip(np.flip(video.data,axis = 1),axis = 2)

    # Select colors
    video_red = video.data[:,1::2,::2]
    video_blue = video.data[:,::2,1::2]
    video_green1 = video.data[:,::2,::2]
    video_green2 = video.data[:,1::2,1::2]

    ##########
    # Normal #
    ##########
    if camera_selection == 1:
        # Rotate picture
        video.data = np.flip(np.flip(video.data,axis = 1),axis = 2)
        video_temp = video.data

        # Select colors
        ## Correct for 20919. Checked 2021/02/24
        video_red = video_temp[:,1::2,::2]
        video_blue = video_temp[:,::2,1::2]
        video_green1 = video_temp[:,::2,::2]
        video_green2 = video_temp[:,1::2,1::2]

    else:
        # Rotate picture
        video.data = np.flip(video.data,axis = 2)
        video_temp = video.data

        # Select colors
        ## Correct for 20919. Checked 2021/02/24
        video_red = video_temp[:,::2,::2]
        video_blue = video_temp[:,1::2,1::2]
        video_green1 = video_temp[:,1::2,::2]
        video_green2 = video_temp[:,::2,1::2]

    return video_red, video_blue, video_green1, video_green2, video
