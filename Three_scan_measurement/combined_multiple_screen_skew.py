import math
import numpy as np
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
# =====================================================================
# =====================================================================
# LPS plotting
# =====================================================================
# =====================================================================
# LPS plotting
from scipy.interpolate import interp2d
import matplotlib.gridspec as gridspec
from matplotlib.ticker import NullFormatter
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import (inset_axes, InsetPosition, mark_inset)
nullfmt = NullFormatter()         # no labels
import os
import matplotlib.pyplot as plt

import sympy as sym
from sympy import MatrixSymbol, Matrix
from sympy import *
import math
import numpy as np

# Dictionary for pixel to actual length conversion
pixel_to_length = { 'Yag4': (1025 ,50),
                    'Yag6': (935, 47),  # (pixels, mm)
                    'Yag5': (207,25),
                    'Yag7': (1025, 50),
                    'Yag8': (857 ,50) }

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


def save_images_and_stats(directory_path, output_directory,masking=True):
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
                masked_filtered_image = filtered_image
                if masking==True:
                    # Apply the ellipse mask to the filtered image
                    masked_filtered_image = np.where(ellipse_mask, filtered_image, 0)

                # Plot x intensity projection
                x_projection = np.sum(masked_filtered_image, axis=0)
                x = np.arange(len(x_projection))
                ax_x_projection.plot(x, x_projection, label='X Projection')

                # Fit Gaussian to x projection
                popt_x, _ = curve_fit(gaussian, x, x_projection,
                                      p0=[x_projection.max(), len(x_projection) // 2, avg_major_axis_length / 3],maxfev=10000)
                ax_x_projection.plot(x, gaussian(x, *popt_x), 'r--', label='Gaussian Fit')
                ax_x_projection.set_title('X Intensity Projection')
                ax_x_projection.legend()

                # Plot y intensity projection
                y_projection = np.sum(masked_filtered_image, axis=1)
                y = np.arange(len(y_projection))
                ax_y_projection.plot(y, y_projection, label='Y Projection')

                # Fit Gaussian to y projection
                popt_y, _ = curve_fit(gaussian, y, y_projection,
                                      p0=[y_projection.max(), len(y_projection) // 2, avg_minor_axis_length / 3],maxfev=10000)
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




def calculate_q1(d2, S11, S12, dT, S21, S22):
    numerator = -d2 * S11 + S12 - d2 * dT * S21 + dT * S22
    denominator = d2 * dT * S12
    q1_positive = math.sqrt(numerator / denominator)
    q1_negative = -q1_positive
    return q1_positive, q1_negative

def calculate_q2(S12, dT, S22, d2, d3, q1):
    numerator = (S12 + dT * S22)
    denominator = d2 * d3 * (1 + S12 * q1)
    q2 = -numerator / denominator
    return q2

def calculate_q3(q1, q2, d2, S11, S21, dT, d3):
    numerator = q1 + q2 + d2 * S11 * q1 * q2 + S21
    denominator = 1 + ((dT * q1 + d3 * q2) * S11) + (d2 * d3 * q2 * (S21 + q1))
    q3 = -numerator / denominator
    return q3

def calculate_q_values(E, alpha_x,beta_x,alpha_y,beta_y,lq=0.15):
    beta = np.sqrt(beta_x*beta_y)
    alpha = (alpha_x + alpha_y)/2
    print(beta)
    print(alpha)
    # Example usage
    E=E*1e-6 #MeV
    gamma=E/0.511
    d2 = 245e-3 # S1 to S2
    d3 = 382.5e-3 #  S1 to S3
    #d2= .348-0.101
    #d3= .481-0.101
    S11 = alpha  #alpha
    S12 = beta  # beta
    dT = d2+d3 # example value
    S21 = -(1+alpha**2)/beta  # beta
    S22 = -alpha  # example value
    S0=np.zeros((2,2))
    S0[0,0] = S11
    S0[0,1] = S12
    S0[1,0] = S21
    S0[1,1] = S22
    lq = 0.15 #m
    fac= (1/298.)*E
    print ("correlation matrix")
    print (S0)
    print('########Solution 1########\n')
    q1_positive, _ = calculate_q1(d2, S11, S12, dT, S21, S22)
    print("q1:", q1_positive, "Q1:",q1_positive*fac/lq,"current1:",q1_positive*fac/(0.7873*lq))

    q2 = calculate_q2(S12, dT, S22, d2, d3, q1_positive)
    print("q2:", q2,"Q2:",q2*fac/lq,"current2:",q2*fac/(0.7873*lq))

    q3 = calculate_q3(q1_positive, q2, d2, S11, S21, dT, d3)

    print("q3:", q3,"Q3:",q3*fac/(lq), "current3:",q3*fac/(lq*0.7873))

    print('########Solution 2########\n')
    _,q1_negative = calculate_q1(d2, S11, S12, dT, S21, S22)
    print("q1:", q1_negative, "Q1:",q1_negative*fac/lq)

    q22 = calculate_q2(S12, dT, S22, d2, d3, q1_negative)
    print("q2:", q22,"Q2:",q22*fac/lq,"current2:",q22*fac/(0.7873*lq))

    q32 = calculate_q3(q1_negative, q2, d2, S11, S21, dT, d3)
    print("q3:", q32,"Q2:",q32*fac/lq,"current3:",q32*fac/(lq*0.7873))


element_positions={
    # Drift space
    'Yag4' : 11372.85e-3,
    'Yag5' : 16484.6e-3,
    'Yag6' : 14922.5e-3,
    'Yag7' : 17792.7e-3,
    'Yag8' : 18821.4e-3,
    'Skew_1' : 12255.5e-3
}

class Screen:
    def __init__(self, name,position,sigma_x_mm,sigma_y_mm):
        self.name = name
        self.position = position
        self.sigma_x_mm =sigma_x_mm
        self.sigma_y_mm = sigma_y_mm

    def __repr__(self):
        return f"Screen(name={self.name}, files={self.position}, sigmas_mm = {self.sigma_x_mm,self.sigma_y_mm})"



# Load JSON data from a file
date = '2024_07_18'
data_path = os.path.join(date,'Scan2')
# Directory to save extracted images
output_directory = os.path.join(data_path,'Extracted_Images_Gradient_Lineouts')
os.makedirs(output_directory, exist_ok=True)
save_images_and_stats(data_path, output_directory,masking=False)
# Save images, perform analysis, and save sigma values
file_path = os.path.join(data_path, '2024_07_18/Scan2/sigma_values.json')  # replace with your actual file path
with open(file_path, 'r') as file:
    data = json.load(file)

# Create Screen objects from JSON data
screens = []

for files in data:
    name = files
    position = element_positions[name]
    sigma_x_mm = data[files][0]['sigma_x_mm']
    sigma_y_mm = data[files][0]['sigma_y_mm']
    screens.append(Screen(name, position,sigma_x_mm,sigma_y_mm))

positions=[]
sigmas_x_mm=[]
sigmas_y_mm=[]
for screen in screens:
    sigmas_x_mm.append(screen.sigma_x_mm)
    sigmas_y_mm.append(screen.sigma_y_mm)
    positions.append(screen.position)
position0 = 0.01
position_min = np.min(positions)
positions=positions-position_min+position0
print(sigmas_x_mm)
print(sigmas_y_mm)
print(positions)
#plt.scatter(positions,sigmas_x_mm)
#plt.scatter(positions,sigmas_y_mm)
#plt.show()

# Curve fitting
# Define the parabolic function
# Define the Gaussian function

def spot_size_evolution(x, sigma_star, x0, beta_star):
    y = sigma_star*(np.sqrt(1+((x-x0)/(beta_star))**2))
    return y

# YAG positions
sYAG = np.array(positions)
sigx_YAG = np.array(sigmas_x_mm)
sigy_YAG = np.array(sigmas_y_mm)

#Skew positions
skew_1_position = element_positions['Skew_1'] - position_min +position0


parametersx, covariancex = curve_fit(spot_size_evolution, sYAG, sigx_YAG)
fit_sigmax = parametersx[0]
fit_x0 = parametersx[1]
fit_beta_star_x = parametersx[2]

parametersy, covariancey = curve_fit(spot_size_evolution, sYAG, sigy_YAG)
fit_sigmay = parametersy[0]
fit_y0 = parametersy[1]
fit_beta_star_y = parametersy[2]

# =====================================================================
# Fitted curve
sfit = np.linspace(0.0, np.max(positions), 1000)
envx = fit_sigmax*(np.sqrt(1+((sfit-fit_x0)/fit_beta_star_x)**2))
envy = fit_sigmay*(np.sqrt(1+((sfit-fit_y0)/fit_beta_star_y)**2))

if len(screens) > 3:
    # Choose three random points between the screens
    random_points = []
    while len(random_points) < 3:
        point = np.random.uniform(np.min(sYAG), np.max(sYAG))
        is_valid = all(np.abs(point - p) > 1e-2 for p in random_points)  # Ensure not too close to other points
        between_screens = any(min(s1, s2) < point < max(s1, s2) for s1, s2 in zip(sYAG[:-1], sYAG[1:]))
        if is_valid and between_screens:
            random_points.append(point)

    random_points = np.sort(random_points)

random_points=positions
sigma_x_random_points = spot_size_evolution(random_points, *parametersx)
sigma_y_random_points = spot_size_evolution(random_points, *parametersy)

plt.figure()
plt.scatter(positions, sigmas_x_mm, label='Sigma X')
plt.scatter(positions, sigmas_y_mm, label='Sigma Y')
plt.plot(sfit, envx, linestyle='dotted', label='Fitted X')
plt.plot(sfit, envy, linestyle='dotted', label='Fitted Y')
plt.scatter(random_points, sigma_x_random_points, color='red', marker='x', label='Random Points X')
plt.scatter(random_points, sigma_y_random_points, color='green', marker='x', label='Random Points Y')
plt.axvline(skew_1_position, linestyle='dotted', color='black', linewidth=2)
plt.legend()
plt.grid(True)
plt.savefig('multiplescan.png')

# ==========================================================================
# ==========================================================================
# ==========================================================================
# Basic parameters
e   = 1.602e-19   # Electron charge, Coulomb
m   = 9.11e-31    # Electron mass
me  = 0.511e+6    # Electron rest mass (MeV/c)
c   = 299792458   # Speed of Light [m/s]
e0  = 8.85e-12    # Electric permittivity of the free space
mu0 = 4*np.pi*1E-7# Permeability of the free space
mp = 938.272e+6   # proton rest mass (eV/c)
m0 = 511000;
mc2 = m0;
EMASS = mc2;

# Energy and gamma
Ebeam1   = 45.3e6 #; %//initial energy in eV
gamma1   = (Ebeam1+EMASS)/EMASS
beta1    = np.sqrt(1-(1/gamma1**2))
P01      = gamma1*beta1*mc2
pCentral    = P01/EMASS;


# ==========================================================================
# ==========================================================================
# Transfer matrix
# Transverse deflecting cavity: This is horizontal TDC matrix

# Initial guess of the kappa
def Quad(kval, lq):
    """return 4 by 4 matrix of horizontal focusing normal quad"""
    return Matrix([[sym.cos(sym.sqrt(kval)*lq),      (1/sym.sqrt(kval))*sym.sin(sym.sqrt(kval)*lq), 0,     0,  0, 0],
                    [(-sym.sqrt(kval))*sym.sin(sym.sqrt(kval)*lq), sym.cos(sym.sqrt(kval)*lq),       0,     0,  0, 0],
                    [0,      0, sym.cosh(sym.sqrt(kval)*lq),      (1/sym.sqrt(kval))*sym.sinh(sym.sqrt(kval)*lq),  0, 0],
                    [0,      0, (-sym.sqrt(kval))*sym.sinh(sym.sqrt(kval)*lq), sym.cosh(sym.sqrt(kval)*lq),  0, 0],
                    [0,      0, 0,     0,  1, 0],
                    [0,      0, 0,     0,  0, 1]])

def Drift(l):
    return Matrix([[1, l, 0, 0,  0, 0],
                    [0, 1,  0, 0,  0, 0],
                    [0, 0,  1, l, 0, 0],
                    [0, 0,  0, 1,  0, 0],
                    [0, 0,  0, 0,  1, 0],
                    [0, 0,  0, 0,  0, 1]])

def Rotation(phi):
    return        Matrix([[np.cos(phi), 0 , np.sin(phi), 0, 0, 0],
        	             [0, np.cos(phi) , 0,  np.sin(phi), 0, 0],
        	             [-np.sin(phi), 0, np.cos(phi), 0, 0, 0],
        	             [0, -np.sin(phi), 0, np.cos(phi), 0, 0],
                         [0, 0,  0, 0,  1, 0],
                         [0, 0,  0, 0,  0, 1]])

D1 = (Drift(random_points[0]))
D2 = (Drift(random_points[1]))
D3 = (Drift(random_points[2]))

D0 = (Drift(skew_1_position))

# Matrix M
M = Matrix([[D1[0,0]**2, 2*D1[0,0]*D1[0,1], D1[0,1]**2],
            [D2[0,0]**2, 2*D2[0,0]*D2[0,1], D2[0,1]**2],
            [D3[0,0]**2, 2*D3[0,0]*D3[0,1], D3[0,1]**2]])

# Beam size calculation
MM = (M.transpose()*M).inv() * M.transpose()
MM = M.inv()

#sigx_DMA = 4.552377400402452
#sigy_DMA = 4.655794209630551
# ==========================================================================
# ==========================================================================

sig1x = (sigma_x_random_points[0]*1e-3)**2
sig2x = (sigma_x_random_points[1]*1e-3)**2
sig3x = (sigma_x_random_points[2]*1e-3)**2

sig1y = (sigma_y_random_points[0]*1e-3)**2
sig2y = (sigma_y_random_points[1]*1e-3)**2
sig3y = (sigma_y_random_points[2]*1e-3)**2


# ==========================================================================
# RMS beam size calculation
#sig0_11 = MM[0][0]*sig1 + MM[0][1]*sig2 + MM[0][2]*sig3
#sig0_12 = MM[1][0]*sig1 + MM[1][1]*sig2 + MM[1][2]*sig3
#sig0_22 = MM[2][0]*sig1 + MM[2][1]*sig2 + MM[2][2]*sig3
sig0x_11 = MM[0,0]*sig1x + MM[0,1]*sig2x + MM[0,2]*sig3x
sig0x_12 = MM[1,0]*sig1x + MM[1,1]*sig2x + MM[1,2]*sig3x
sig0x_22 = MM[2,0]*sig1x + MM[2,1]*sig2x + MM[2,2]*sig3x
sig0y_11 = MM[0,0]*sig1y + MM[0,1]*sig2y + MM[0,2]*sig3y
sig0y_12 = MM[1,0]*sig1y + MM[1,1]*sig2y + MM[1,2]*sig3y
sig0y_22 = MM[2,0]*sig1y + MM[2,1]*sig2y + MM[2,2]*sig3y


sig0x_11 = np.array(sig0x_11, dtype='float')
sig0x_12 = np.array(sig0x_12, dtype='float')
sig0x_22 = np.array(sig0x_22, dtype='float')
sig0y_11 = np.array(sig0y_11, dtype='float')
sig0y_12 = np.array(sig0y_12, dtype='float')
sig0y_22 = np.array(sig0y_22, dtype='float')

# ==========================================================================
# RMS beam size calculation
sig0x_recon = np.sqrt(sig0x_11)
sig0y_recon = np.sqrt(sig0y_11)

# Emittance
emitx_recon = np.sqrt(sig0x_11*sig0x_22 - (sig0x_12**2))
emity_recon = np.sqrt(sig0y_11*sig0y_22 - (sig0y_12**2))

# Normalized emittance
enx_recon = emitx_recon * pCentral
eny_recon = emity_recon * pCentral

# Twiss Beta function
betax_recon = sig0x_recon**2 / emitx_recon
betay_recon = sig0y_recon**2 / emity_recon

# Twiss Beta function
alphax_recon = -sig0x_12 / emitx_recon
alphay_recon = -sig0y_12 / emity_recon


print('RMSX is ' + repr(sig0x_recon*1e3)+ ' mm.')
print('RMSY is ' + repr(sig0y_recon*1e3)+ ' mm.')
print('========================================')
print('enx is ' + repr(enx_recon*1e6)+ ' mm mrad.')
print('eny is ' + repr(eny_recon*1e6)+ ' mm mrad.')
print('========================================')
print('betax at the initial position is  '+repr(betax_recon)+ ' m.')
print('betay at the initial position is  '+repr(betay_recon)+ ' m.')
print('alphax at the initial position is '+repr(alphax_recon)+ ' .')
print('alphay at the initial position is '+repr(alphay_recon)+ ' .')
print('========================================')

# =====================================================================
# Inverse matrix to calculate beam parameters at YAG780
gammax_recon = (1 + alphax_recon**2) / betax_recon
gammay_recon = (1 + alphay_recon**2) / betay_recon


# Twiss matrix to calculate the Twiss params at the entrance of skew quad
MTx =Matrix([[D0[0,0]**2, -2*D0[0,0]*D0[0,1], D0[0,1]**2],
            [-D0[0,0]*D0[1,0], D0[0,0]*D0[1,1] + D0[0,1]*D0[1,0], -D0[0,1]*D0[1,1]],
            [D0[1,0]**2, -2*D0[1,0]*D0[1,1], D0[1,1]**2]])

MTy =Matrix([[D0[2,2]**2, -2*D0[2,2]*D0[2,3], D0[2,3]**2],
            [-D0[2,2]*D0[3,2], D0[2,2]*D0[3,3] + D0[2,3]*D0[3,2], -D0[2,3]*D0[3,3]],
            [D0[3,2]**2, -2*D0[3,2]*D0[3,3], D0[3,3]**2]])


# Twiss inverse
MTxi = MTx
MTyi = MTy

betaxi = MTxi[0,0]*betax_recon + MTxi[0,1]*alphax_recon + MTxi[0,2]*gammax_recon
alphaxi= MTxi[1,0]*betax_recon + MTxi[1,1]*alphax_recon + MTxi[1,2]*gammax_recon
betayi = MTyi[0,0]*betay_recon + MTyi[0,1]*alphay_recon + MTyi[0,2]*gammay_recon
alphayi= MTyi[1,0]*betay_recon + MTyi[1,1]*alphay_recon + MTyi[1,2]*gammay_recon

betaxi  = np.array(betaxi, dtype='float')
alphaxi = np.array(alphaxi, dtype='float')

betayi  = np.array(betayi, dtype='float')
alphayi = np.array(alphayi, dtype='float')

print('#########Parameters############')
print('z0_x:  '+repr(fit_x0)+ ' m.')
print('z0_y:  '+repr(fit_y0)+ ' m.')
print('beta waist x:'+repr(fit_beta_star_x)+ ' m.')
print('beta waist y: '+repr(fit_beta_star_y)+ ' .')
print('alphay at the entrance of SQ1 is '+repr(alphayi)+ ' .')


print('betax at the entrance of SQ1 is  '+repr(betaxi)+ ' m.')
print('betay at the entrance of SQ1 is  '+repr(betayi)+ ' m.')
print('alphax at the entrance of SQ1 is '+repr(alphaxi)+ ' .')
print('alphay at the entrance of SQ1 is '+repr(alphayi)+ ' .')

calculate_q_values(Ebeam1,alphaxi,betaxi,alphayi,betayi,0.15)