import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from scipy.ndimage import gaussian_gradient_magnitude
from skimage.filters import threshold_otsu
from scipy.optimize import curve_fit
import json

# Dictionary for pixel to actual length conversion
pixel_to_length = { 'Yag4': (1025 ,50),
                    'Yag6': (935, 47),  # (pixels, mm)
                    'Yag5': (207,25),
                    'Yag7': (1025, 50),
                    'Yag8': (857 ,50) }
date = '2024_07_18'
# Directory containing HDF5 files
directory_path = os.path.join(date, 'Corrected_FlatBeam')

def create_circular_mask(h, w, center=None, radius=None):
    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask, center, radius

# Directory to save extracted images
output_directory = os.path.join(directory_path, 'Extracted_Images')
os.makedirs(output_directory, exist_ok=True)

def remove_outliers(image, mask, threshold_factor=3):
    masked_image = np.ma.masked_array(image, mask=~mask)
    mean = masked_image.mean()
    std = masked_image.std()
    lower_bound = mean - threshold_factor * std
    upper_bound = mean + threshold_factor * std
    filtered_image = np.where((masked_image >= lower_bound) & (masked_image <= upper_bound), masked_image, 0)
    return filtered_image

files = [f for f in os.listdir(directory_path) if f.endswith('.h5')]

print(files)
for file in files:
    file_path = os.path.join(directory_path, file)

    # Extract the screen name
    screen = file.split('_')[0]

    # Extract the position from the filename
    position = file.split('_')[1]

    # Get the pixel to length conversion factor for the screen
    pixels, length_mm = pixel_to_length[screen]
    pixel_to_mm = length_mm / pixels

    # Open the HDF5 file
    with h5py.File(file_path, 'r') as h5file:
        # Access the 'images' dataset
        images = h5file['images'][1:2]
        print(np.shape(images))
        # Store results for each image
        results = []

        for i, image in enumerate(images):
            # Create a circular mask
            h, w = image.shape
            mask, center, radius = create_circular_mask(h, w, center=(w // 2, h // 2), radius=min(h, w) // 2)

            # Remove outliers
            #filtered_image = remove_outliers(image, mask)
            filtered_image=image
            # Convert pixel values to mm
            image_mm = filtered_image * pixel_to_mm
            print(pixel_to_mm)
            # Save the extracted image
            plt.imshow(image_mm, extent=(0, w * pixel_to_mm, 0, h * pixel_to_mm))
            plt.xlabel('x (mm)')
            plt.ylabel('y (mm)')
            plt.colorbar(label='Intensity (a.u.)')

            # Zoom in on the center region (e.g., 20% of the full width and height)
            zoom_factor = 1
            center_x, center_y = w * pixel_to_mm / 2, h * pixel_to_mm / 2
            zoom_width, zoom_height = w * pixel_to_mm, h * pixel_to_mm * zoom_factor

            plt.xlim(center_x - zoom_width / 2, center_x + zoom_width / 2)
            plt.ylim(center_y - zoom_height / 2, center_y + zoom_height / 2)

            # Save the image in mm scale
            output_file_path = os.path.join(output_directory, f'{screen}_{position}_image_{i}.png')
            plt.savefig(output_file_path)
            plt.close()

            # Display the image
            #plt.imshow(filtered_image)
            #plt.show()
