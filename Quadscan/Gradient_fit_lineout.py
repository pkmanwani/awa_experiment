
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

date = '2024_07_18'
# Directory containing HDF5 files
directory_path = os.path.join(date,'data')

# Directory to save extracted images
output_directory = os.path.join(directory_path,'Extracted_Images_Gradient_Lineouts')
os.makedirs(output_directory, exist_ok=True)

# Dictionary for pixel to actual length conversion
pixel_to_length = { 'Yag4': (1025 ,50),
                    'Yag6': (935, 47),  # (pixels, mm)
                    'Yag7': (1025, 50),
                    'Yag8': (857 ,50) }

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


def fit_ellipse(filtered_image, mask, center, radius):
    # Threshold the filtered image
    threshold = threshold_otsu(filtered_image)
    binary_image = filtered_image > threshold

    # Remove small objects from the binary image
    binary_image = remove_small_objects(binary_image, min_size=1)  # Adjust min_size as needed

    # Label the binary image
    labeled_image = label(binary_image)

    # Find the largest region
    regions = regionprops(labeled_image)
    if not regions:
        return None

    largest_region = max(regions, key=lambda region: region.area)

    # Fit an ellipse to the largest region
    yc, xc = largest_region.centroid
    orientation = np.pi / 2 - largest_region.orientation
    major_axis_length = largest_region.major_axis_length / 2
    minor_axis_length = largest_region.minor_axis_length / 2

    # Ensure the ellipse is within the circular mask
    dist_to_boundary = radius - np.sqrt((xc - center[0]) ** 2 + (yc - center[1]) ** 2)
    if major_axis_length <= dist_to_boundary and minor_axis_length <= dist_to_boundary:
        # Calculate points along the semi-major and semi-minor axes
        cos_orientation = np.cos(orientation)
        sin_orientation = np.sin(orientation)
        axis_major_samples = np.linspace(-1.2 * major_axis_length, 1.2 * major_axis_length, num=100)
        axis_minor_samples = np.linspace(-1.2 * minor_axis_length, 1.2 * minor_axis_length, num=100)

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

def fit_ellipse_from_gradient(filtered_image, mask, center, radius):
    # Compute gradient magnitude
    gradient_mag = gaussian_gradient_magnitude(filtered_image, sigma=3)

    # Threshold the gradient magnitude
    binary_image = gradient_mag > gradient_mag.mean()

    # Remove small objects from the binary image
    binary_image = remove_small_objects(binary_image, min_size=1)  # Adjust min_size as needed

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
        axis_major_samples = np.linspace(-1.2 * major_axis_length, 1.2 * major_axis_length, num=100)
        axis_minor_samples = np.linspace(-1.2* minor_axis_length, 1.2 * minor_axis_length, num=100)

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

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def create_ellipse_mask(h, w, xc, yc, major_axis_length, minor_axis_length, orientation):
    Y, X = np.ogrid[:h, :w]
    cos_orientation = np.cos(orientation)
    sin_orientation = np.sin(orientation)
    x_diff = X - xc
    y_diff = Y - yc

    # Increase masking ellipse slightly larger than the ellipse fit
    ellipse_mask = ((x_diff * cos_orientation + y_diff * sin_orientation) ** 2 / (1.2 * major_axis_length) ** 2 + \
                    (x_diff * sin_orientation - y_diff * cos_orientation) ** 2 / (1.2 * minor_axis_length) ** 2) <= 1

    return ellipse_mask


def save_images_and_stats(directory_path, output_directory):
    files = [f for f in os.listdir(directory_path) if f.endswith('.h5')]

    # Dictionary to store sigma values in mm for each screen
    sigma_values = {}

    for file in files:
        file_path = os.path.join(directory_path, file)

        # Extract the screen name
        screen = file.split('_')[0]

        # Extract the position from the filename
        position = file.split('_')[1]

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
                result = fit_ellipse(filtered_image, mask, center, radius)

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

                # Conversion to millimeters
                pixels, mm = pixel_to_length.get(screen, (1, 1))  # Default to (1, 1) if screen not in dictionary
                avg_major_axis_length_mm = (avg_major_axis_length / pixels) * mm
                avg_minor_axis_length_mm = (avg_minor_axis_length / pixels) * mm
                std_major_axis_length_mm = (std_major_axis_length / pixels) * mm
                std_minor_axis_length_mm = (std_minor_axis_length / pixels) * mm

                # Prepare the image for plotting
                plot_image = np.where(mask, filtered_image, 0)

                # Create the ellipse mask to ignore regions outside the ellipse
                ellipse_mask = create_ellipse_mask(h, w, avg_xc, avg_yc, 1.2*avg_major_axis_length, 1.2*avg_minor_axis_length,
                                                   avg_orientation)

                # Create the main plot with the circular mask boundary and fitted ellipse
                fig, axes = plt.subplots(2, 2, figsize=(15, 15))
                ax_main = axes[0, 0]
                ax_x_projection = axes[0, 1]
                ax_y_projection = axes[1, 0]
                ax_fits = axes[1, 1]

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

                # Apply the ellipse mask to the filtered image
                masked_filtered_image = np.where(ellipse_mask, filtered_image, 0)

                # Plot x intensity projection
                x_projection = np.sum(masked_filtered_image, axis=0)
                x = np.arange(len(x_projection))
                ax_x_projection.plot(x, x_projection, label='X Projection')

                # Fit Gaussian to x projection
                popt_x, _ = curve_fit(gaussian, x, x_projection,
                                      p0=[x_projection.max(), len(x_projection) // 2, avg_major_axis_length / 3])
                ax_x_projection.plot(x, gaussian(x, *popt_x), 'r--', label='Gaussian Fit')
                ax_x_projection.set_title('X Intensity Projection')
                ax_x_projection.legend()

                # Plot y intensity projection
                y_projection = np.sum(masked_filtered_image, axis=1)
                y = np.arange(len(y_projection))
                ax_y_projection.plot(y, y_projection, label='Y Projection')

                # Fit Gaussian to y projection
                popt_y, _ = curve_fit(gaussian, y, y_projection,
                                      p0=[y_projection.max(), len(y_projection) // 2, avg_minor_axis_length / 3])
                ax_y_projection.plot(y, gaussian(y, *popt_y), 'r--', label='Gaussian Fit')
                ax_y_projection.set_title('Y Intensity Projection')
                ax_y_projection.legend()

                # Save sigma values in mm for the current screen
                sigma_x_mm = (popt_x[2] / pixels) * mm
                sigma_y_mm = (popt_y[2] / pixels) * mm

                if screen not in sigma_values:
                    sigma_values[screen] = []

                sigma_values[screen].append({
                    'position': position,
                    'sigma_x_mm': sigma_x_mm,
                    'sigma_y_mm': sigma_y_mm
                })

                # Summary plot with fitted Gaussian parameters
                ax_fits.axis('off')
                fit_summary = f'X Fit: A = {popt_x[0]:.2f}, Mean = {popt_x[1]:.2f}, Sigma = {popt_x[2]:.2f} pixels ({sigma_x_mm:.2f} mm)\n' + \
                              f'Y Fit: A = {popt_y[0]:.2f}, Mean = {popt_y[1]:.2f}, Sigma = {popt_y[2]:.2f} pixels ({sigma_y_mm:.2f} mm)'
                ax_fits.text(0.1, 0.5, fit_summary, fontsize=12)

                ax_main.set_title(f'Screen {screen} - Position {position}')
                ax_main.axis('off')

                # Save the plot
                output_path = os.path.join(output_directory, f'{screen}_position_{position}.png')
                plt.savefig(output_path, bbox_inches='tight')
                plt.close()
            else:
                print(f'No suitable regions found for {file}')

    # Save sigma values to a JSON file
    with open(os.path.join(directory_path, '2024_07_18/Scan2/sigma_values.json'), 'w') as json_file:
        json.dump(sigma_values, json_file, indent=4)


# Save images, perform analysis, and save sigma values
save_images_and_stats(directory_path, output_directory)
