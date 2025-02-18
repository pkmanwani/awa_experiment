import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from scipy.ndimage import gaussian_gradient_magnitude
import json

path = 'Slitscan'
# Directory containing HDF5 files
directory_path = os.path.join(path, 'data')

# Directory to save extracted images
output_directory = os.path.join(path, 'Extracted_Images_Gradient_Lineouts')
os.makedirs(output_directory, exist_ok=True)

# Directory to save extracted data
results_directory = os.path.join(path, 'Gradient_results')
os.makedirs(results_directory, exist_ok=True)

# Dictionary for pixel to actual length conversion
pixel_to_length = {
    'Yag6': (422, 47),  # (pixels, mm)
    'Yag7': (1030, 50)  # (pixels, mm)
}

def create_circular_mask(h, w, center=None, radius=None):
    if center is None:
        center = (int(w / 2), int(h / 2))
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask, center, radius

def remove_outliers(image, mask, threshold_factor=3):
    masked_image = np.ma.masked_array(image, mask=~mask)
    mean = masked_image.mean()
    std = masked_image.std()

    lower_bound = mean - threshold_factor * std
    upper_bound = mean + threshold_factor * std
    filtered_image = np.where((masked_image >= lower_bound), masked_image, 0)

    return filtered_image

def fit_ellipse_from_gradient(filtered_image, mask, center, radius):
    # Compute gradient magnitude
    gradient_mag = gaussian_gradient_magnitude(filtered_image, sigma=0.6)

    # Threshold the gradient magnitude
    binary_image = gradient_mag > gradient_mag.mean()

    # Remove small objects from the binary image
    binary_image = remove_small_objects(binary_image, min_size=2)  # Adjust min_size as needed

    # Label the binary image
    labeled_image = label(binary_image)

    # Find the largest region
    regions = regionprops(labeled_image)
    if not regions:
        return None

    largest_region = max(regions, key=lambda region: region.area)

    # Fit an ellipse to the largest region
    yc, xc = largest_region.centroid
    orientation = np.pi/2 - largest_region.orientation
    major_axis_length = largest_region.major_axis_length / 2
    minor_axis_length = largest_region.minor_axis_length / 2

    # Ensure the ellipse is within the circular mask
    dist_to_boundary = radius - np.sqrt((xc - center[0]) ** 2 + (yc - center[1]) ** 2)
    if major_axis_length <= dist_to_boundary and minor_axis_length <= dist_to_boundary:
        # Calculate points along the semi-major and semi-minor axes
        cos_orientation = np.cos(orientation)
        sin_orientation = np.sin(orientation)
        axis_major_samples = np.linspace(-1.5 * major_axis_length, 1.5 * major_axis_length, num=100)
        axis_minor_samples = np.linspace(-1.5 * minor_axis_length, 1.5 * minor_axis_length, num=100)

        # Coordinates of points along the semi-major and semi-minor axes
        axis_major_points = np.array(
            [[xc + sample * cos_orientation, yc + sample * sin_orientation] for sample in axis_major_samples])
        axis_minor_points = np.array(
            [[xc - sample * sin_orientation, yc + sample * cos_orientation] for sample in axis_minor_samples])

        # Extract intensity values along the semi-major and semi-minor axes
        intensity_major = [filtered_image[int(point[1]), int(point[0])] for point in axis_major_points]
        intensity_minor = [filtered_image[int(point[1]), int(point[0])] for point in axis_minor_points]

        return largest_region, orientation, xc, yc, major_axis_length, minor_axis_length, axis_major_samples, intensity_major, axis_minor_samples, intensity_minor

    else:
        return None

def save_images(directory_path, output_directory,results_directory):
    files = [f for f in os.listdir(directory_path) if f.endswith('.h5')]

    for file in files:
        file_path = os.path.join(directory_path, file)

        # Extract the screen name
        screen = file.split('_')[0]
        print(screen)

        # Extract the position from the filename
        position = file.split('_')[2]
        print(position)

        # Open the HDF5 file
        with h5py.File(file_path, 'r') as h5file:
            # Access the 'images' dataset
            images = h5file['images'][:]

            # Store results for each image
            results = []

            for i, image in enumerate(images):
                # Create a circular mask
                h, w = image.shape
                mask, center, radius = create_circular_mask(h, w, center=(w // 2, h // 2), radius=min(h, w) // 2)

                # Remove outliers
                filtered_image = remove_outliers(image, mask)

                # Fit an ellipse to the particles using gradient magnitude
                result = fit_ellipse_from_gradient(filtered_image, mask, center, radius)

                if result:
                    results.append(result)

            if results:
                # Calculate the average and standard deviation for each parameter
                orientations = [res[1] for res in results]
                xcs = [res[2] for res in results]
                ycs = [res[3] for res in results]
                major_axis_lengths = [res[4] for res in results]
                minor_axis_lengths = [res[5] for res in results]

                avg_orientation = np.mean(orientations)
                avg_xc = np.mean(xcs)
                avg_yc = np.mean(ycs)
                avg_major_axis_length = np.mean(major_axis_lengths)
                avg_minor_axis_length = np.mean(minor_axis_lengths)

                std_orientation = np.std(orientations)
                std_xc = np.std(xcs)
                std_yc = np.std(ycs)
                std_major_axis_length = np.std(major_axis_lengths)
                std_minor_axis_length = np.std(minor_axis_lengths)

                # Convert pixel lengths to actual lengths
                pixels, mm = pixel_to_length.get(screen, (1, 1))  # Default to (1, 1) if screen not in dictionary
                avg_major_axis_length_mm = (avg_major_axis_length / pixels) * mm
                avg_minor_axis_length_mm = (avg_minor_axis_length / pixels) * mm
                std_major_axis_length_mm = (std_major_axis_length / pixels) * mm
                std_minor_axis_length_mm = (std_minor_axis_length / pixels) * mm

                # Prepare the image for plotting
                plot_image = np.where(mask, filtered_image, 0)

                # Create the main plot with the circular mask boundary and fitted ellipse
                fig, (ax_main, ax_lineout_major, ax_lineout_minor) = plt.subplots(1, 3, figsize=(15, 8),
                                                                                  gridspec_kw={
                                                                                      'width_ratios': [2, 1, 1]})

                ax_main.imshow(plot_image, cmap='gray')
                circle = patches.Circle(center, radius, edgecolor='red', facecolor='none', linewidth=2)
                ax_main.add_patch(circle)

                # Draw the fitted ellipse
                ellipse = patches.Ellipse((avg_xc, avg_yc), 2 * avg_major_axis_length, 2 * avg_minor_axis_length,
                                          angle=np.degrees(avg_orientation), edgecolor='yellow', facecolor='none',
                                          linewidth=1)
                ax_main.add_patch(ellipse)

                # Draw dotted lines for the semi-major and semi-minor axes
                major_line_x = [avg_xc - avg_major_axis_length * np.cos(avg_orientation),
                                avg_xc + avg_major_axis_length * np.cos(avg_orientation)]
                major_line_y = [avg_yc - avg_major_axis_length * np.sin(avg_orientation),
                                avg_yc + avg_major_axis_length * np.sin(avg_orientation)]
                minor_line_x = [avg_xc - avg_minor_axis_length * np.sin(avg_orientation),
                                avg_xc + avg_minor_axis_length * np.sin(avg_orientation)]
                minor_line_y = [avg_yc + avg_minor_axis_length * np.cos(avg_orientation),
                                avg_yc - avg_minor_axis_length * np.cos(avg_orientation)]

                ax_main.plot(major_line_x, major_line_y, 'w--')
                ax_main.plot(minor_line_x, minor_line_y, 'w--')

                # Plot intensity along semi-major axis with error bars
                axis_major_samples = results[0][6]
                intensity_major_avg = np.mean([res[7] for res in results], axis=0)
                intensity_major_std = np.std([res[7] for res in results], axis=0)
                ax_lineout_major.errorbar(axis_major_samples, intensity_major_avg, yerr=intensity_major_std, fmt='-o',
                                          label='Semi-major axis')
                ax_lineout_major.set_title('Intensity along Semi-major axis')
                ax_lineout_major.set_xlabel('Position along axis')
                ax_lineout_major.set_ylabel('Intensity')
                ax_lineout_major.legend()

                # Plot intensity along semi-minor axis with error bars
                axis_minor_samples = results[0][8]
                intensity_minor_avg = np.mean([res[9] for res in results], axis=0)
                intensity_minor_std = np.std([res[9] for res in results], axis=0)
                ax_lineout_minor.errorbar(axis_minor_samples, intensity_minor_avg, yerr=intensity_minor_std, fmt='-o',
                                          label='Semi-minor axis')
                ax_lineout_minor.set_title('Intensity along Semi-minor axis')
                ax_lineout_minor.set_xlabel('Position along axis')
                ax_lineout_minor.set_ylabel('Intensity')
                ax_lineout_minor.legend()

                # Prepare the dictionary to save
                result_data = {
                    'File': file,
                    'Centroid x': avg_xc,
                    'Centroid y': avg_yc,
                    'Average Orientation': np.degrees(avg_orientation),
                    'Std Orientation': np.degrees(std_orientation),
                    'Average Major axis length (pixels)': avg_major_axis_length,
                    'Std Major axis length (pixels)': std_major_axis_length,
                    'Average Major axis length (mm)': avg_major_axis_length_mm,
                    'Std Major axis length (mm)': std_major_axis_length_mm,
                    'Average Minor axis length (pixels)': avg_minor_axis_length,
                    'Std Minor axis length (pixels)': std_minor_axis_length,
                    'Average Minor axis length (mm)': avg_minor_axis_length_mm,
                    'Std Minor axis length (mm)': std_minor_axis_length_mm,
                    'Circular Beam Radius': np.sqrt(avg_minor_axis_length_mm * avg_major_axis_length_mm)
                }

                # Save as a JSON file in 'gradient results' directory
                result_file_path = os.path.join(results_directory, file.replace('.h5', '.json'))
                with open(result_file_path, 'w') as f_out:
                    json.dump(result_data, f_out, indent=4)

                print(f'Saved results for {file}')
                print(f'File: {file}')
                print(f'Average Centroid: ({avg_xc:.2f} ± {std_xc:.2f}, {avg_yc:.2f} ± {std_yc:.2f})')
                print(f'Average Orientation: {np.degrees(avg_orientation):.2f} ± {np.degrees(std_orientation):.2f} degrees')
                print(f'Average Major axis length: {avg_major_axis_length:.2f} ± {std_major_axis_length:.2f} pixels ({avg_major_axis_length_mm:.2f} ± {std_major_axis_length_mm:.2f} mm)')
                print(f'Average Minor axis length: {avg_minor_axis_length:.2f} ± {std_minor_axis_length:.2f} pixels ({avg_minor_axis_length_mm:.2f} ± {std_minor_axis_length_mm:.2f} mm)')
                print(f'Assuming circular beam R: {np.sqrt(avg_minor_axis_length_mm * avg_major_axis_length_mm)} ± {np.sqrt(std_major_axis_length_mm*std_minor_axis_length_mm):.2f} mm')

                ax_main.set_title(f'Screen {screen} - Position {position}')
                ax_main.axis('off')

                # Save the plot
                output_path = os.path.join(output_directory, f'{screen}_position_{position}.png')
                plt.savefig(output_path, bbox_inches='tight')
                plt.close()
            else:
                print('No result\n')

# Save images and perform analysis
save_images(directory_path, output_directory,results_directory)

