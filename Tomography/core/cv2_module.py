import cv2
import numpy as np
import pdb
import os
import tkinter as tk
from tkinter import filedialog
from scipy.sparse import load_npz


# Global variables for mouse callback
rect_start = None
rect_end = None
drawing = False

def select_roi(event, x, y, flags, param):
    global rect_start, rect_end, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        rect_start = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            rect_end = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        rect_end = (x, y)
        drawing = False

def get_mask_from_selection(image_path = None, frame_input = None):
    global rect_start, rect_end
    frame_input = frame_input or 0

    if not image_path:
        # Hide the main tkinter window
        root = tk.Tk()
        root.withdraw()

        # Open a file dialog
        image_path = filedialog.askopenfilename(title="Select an image to crop", filetypes=[("Images", "*.png"), ("CSV files", "*.csv"), ("All files", "*.*")])
    # Load image
    name = image_path.split('.')
    name = name[-1]
    if name == 'png':
        image = cv2.imread(image_path)
    elif name == 'npz':
        vid = np.load(image_path, allow_pickle=True)
        images = np.flip(vid['images'], 1)
        image = images[frame_input, :, :].T
        image = image.astype(np.uint8)
        image = np.repeat(image[:, :, np.newaxis], 3, axis = 2)
        
    else:
        raise Exception('unrecognized video file')
    clone = image.copy()
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", select_roi)

    while True:
        temp_image = clone.copy()
        if rect_start and rect_end:
            cv2.rectangle(temp_image, rect_start, rect_end, (0, 255, 0), 2)
        cv2.imshow("Select ROI", temp_image)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key to confirm selection
            break
        elif key == 27:  # Escape key to cancel
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()

    # Create mask
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    if rect_start and rect_end:
        x1, y1 = rect_start
        x2, y2 = rect_end
        mask[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)] = 255

    return mask


import cv2
import numpy as np

# Global variables
drawing = False
points = []

def draw_freeform(event, x, y, flags, param):
    global drawing, points
    if event == cv2.EVENT_LBUTTONDOWN:  # Start drawing
        drawing = True
        points = [(x, y)]  # Start a new shape

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append((x, y))  # Collect points

    elif event == cv2.EVENT_LBUTTONUP:  # Stop drawing
        drawing = False

def get_freeform_mask(image_path = None, frame_input = None):

    global points

    frame_input = frame_input or 0
    if not image_path:
        # Hide the main tkinter window
        root = tk.Tk()
        root.withdraw()

        # Open a file dialog
        image_path = filedialog.askopenfilename(title="Select an image to crop", filetypes=[("Images", "*.png"), ("CSV files", "*.csv"), ("All files", "*.*")])
    # Load image
    name = image_path.split('.')
    name = name[-1]
    if name == 'png':
        image = cv2.imread(image_path)
    elif name == 'npz':
        vid = np.load(image_path, allow_pickle=True)
        images = np.flip(vid['images'], 1)
        image = images[frame_input, :, :].T
        image = image.astype(np.uint8)
        image = np.repeat(image[:, :, np.newaxis], 3, axis = 2)
        
    else:
        raise Exception('unrecognized video file')
    clone = image.copy()

    # Create window and set mouse callback
    cv2.namedWindow("Draw Freeform Mask")
    cv2.setMouseCallback("Draw Freeform Mask", draw_freeform)

    while True:
        temp_image = clone.copy()

        # Draw the freeform shape as the user moves the mouse
        if len(points) > 1:
            cv2.polylines(temp_image, [np.array(points)], isClosed=False, color=(0, 65535, 0), thickness=2)

        cv2.imshow("Draw Freeform Mask", temp_image)

        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Enter key to confirm selection
            break
        elif key == 27:  # Escape key to cancel
            cv2.destroyAllWindows()
            return None

    cv2.destroyAllWindows()

    # Create mask
    mask = np.zeros(image.shape[:2], dtype=np.uint16)

    if len(points) > 2:  # Ensure we have enough points to form a closed shape
        cv2.fillPoly(mask, [np.array(points)], 65535)
    
    return mask


def get_and_save_mask(image_path = None, frame_input = None, nshot = ''):
    if nshot:
        nshot = str(nshot) + '/'
    directory = '/Home/LF276573/Documents/Python/CHERAB/masks/west/' + nshot 
    
    os.makedirs(directory, exist_ok=True)
    if not frame_input:
        frame_input = 0
    mask = get_freeform_mask(image_path = image_path, frame_input = frame_input)
    counter = 1
    base_filename = 'custom_frame_' + str(frame_input)
    filename = os.path.join(directory, f"{base_filename}_{counter}.npy")
    while os.path.exists(filename):
        filename = os.path.join(directory, f"{base_filename}_{counter}.npy")
        counter += 1
    np.save(filename, mask)



def smooth_line_image(image = None, window_size = 3):
    # Load the image 
    import utility_functions
    import matplotlib.pyplot as plt
    image_path = utility_functions.get_file(full_path=1)
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply a horizontal median filter to remove horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (image.shape[1], 20))
    horizontal_filtered = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, horizontal_kernel)

    # Apply a Gaussian blur to further smooth the image
    smoothed_image = cv2.GaussianBlur(horizontal_filtered, (1, 5), 0)

    # Convert the smoothed image back to color
    # smoothed_color_image = cv2.cvtColor(smoothed_image, cv2.COLOR_GRAY2BGR)
    smoothed_color_image = cv2.cvtColor(smoothed_image, cv2.COLOR_GRAY2BGR)

    # Display the original and smoothed images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Smoothed Image')
    plt.imshow(cv2.cvtColor(smoothed_color_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.show()

    # Save the smoothed image
    # cv2.imwrite('smoothed_image.jpg', smoothed_color_image)
    return smoothed_color_image, image