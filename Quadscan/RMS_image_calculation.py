# =====================================================================
# =====================================================================
# LPS plotting
# =====================================================================
# =====================================================================
# LPS plotting
from IPython import get_ipython
get_ipython().magic('reset -sf')


from IPython import get_ipython
get_ipython().magic('reset -sf')

from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp2d
import matplotlib.gridspec as gridspec
from matplotlib.ticker import NullFormatter
from matplotlib import rc
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition,
                                                  mark_inset)
nullfmt = NullFormatter()         # no labels
from scipy.stats import norm


from UIL import get_mask, median_filter, LoadAWA, MouseCrop
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.optimize import curve_fit

import h5py
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter



file_dir = Path('./')

# Step name
step = str('Fig_YAG186')

beam_file = file_dir / '../Case2_Fsol_412.5A/Magnetization_Zone5/EYG7_Beam_Distribution.hdf5'
bkg_file  = file_dir / '../Case2_Fsol_412.5A/Magnetization_Zone5/EYG7_Background.hdf5'
img_only = 0
assert beam_file.exists()
filename = beam_file


# =====================================================================
# =====================================================================
# Import data from hdf5 file
with h5py.File(f"{filename}","r") as f:
    img = f["image"][:]
    charge = f["charge"][:]
    charge2 = f["charge2"][:]
    bpmx = f["bpmx"][:]
    #scope_data = f["scope_data"][:]
    
# =====================================================================
# =====================================================================
# Filter the charge 
set_charge = 1e-9
devcharge = 0.1
filter_charge = np.where((charge > set_charge*(1-devcharge)) & (charge <set_charge*(1+devcharge)))[0]

# =====================================================================
# =====================================================================
# Image check for cutting out
H = img[0]
H = np.flipud(H)
#H1 = H / max(map(max, H))
#H1[H1 < 0] = 0
H1 = H

nbinx = len(H1[0,:])
nbiny = len(H1[:,0])

# =====================================================================
# =====================================================================
# Calibration of the camera resolution

# Pixel calibration
DYG4_pixel   = 4.0000000000000000e-05
YAG689_pixel = 3.9232781168265036e-05
YAG1022_pixel= 2.0622685185185184e-05
EYG7_pixel   = 4.065040650406504e-05
YAG780_pixel = 3.6764705882352945e-05

pixely = YAG780_pixel * 1e3 # # Pixel number in millimeter
pixelx = pixely  # Pixel number in ps


Xedgesi =  np.linspace(-nbinx/2*pixelx, nbinx/2*pixelx, nbinx)
Yedgesi =  np.linspace(-nbiny/2*pixely, nbiny/2*pixely, nbiny)
Xn, Yn = np.meshgrid(Xedgesi, Yedgesi)


plt.figure()
plt.pcolormesh(H)
plt.set_cmap('viridis')
plt.colorbar()
plt.clim([0,300])

# =====================================================================
# =====================================================================
# Check the image
#for i in range(len(filter_charge)):    
#    plt.figure(figsize=(16,10))
#    #plt.imshow(gaussian_filter(img[i][200:1000, 400:1400], 0))
#    plt.imshow(gaussian_filter(img[i], 0))
#    plt.colorbar()
#    #plt.clim(300,10000)
#    plt.show()




# =====================================================================
# =====================================================================
# Define the Gaussian function
def Gauss(x, A, B, C):
    y = A*np.exp(-((x-B)**2/(2*(C)**2)))
    return y

# =====================================================================
# Mesh range for plotting

#x_start = 400
#x_end   = 574

#y_start = 900
#y_end   = 1020
x_start = 380
x_end   = 1030

y_start = 170
y_end   = 1000
# =====================================================================
# For DYG4
x_start = 400
x_end   = 800

y_start = 450
y_end   = 1250
# =====================================================================
# =====================================================================
# For YAG 198
x_start = 200
x_end   = 1000

y_start = 200
y_end   = 900
# =====================================================================
# =====================================================================
# For EYG7
#x_start = 325
#x_end   = 825

#y_start = 400
#y_end   = 850
# =====================================================================
# =====================================================================
# For DMA_1047
#x_start = 275
#x_end   = 975

#y_start = 800
#y_end   = 1500

# =====================================================================
# =====================================================================
# For YAG_186
#x_start = 500
#x_end   = 1000

#y_start = 800
#y_end   = 1500
#x_start = 1425
#x_end   = 2425
#y_start = 1450
#y_end   = 2450

nbinx = y_end - y_start
nbiny = x_end - x_start


Xedges =  np.linspace(-nbinx/2*pixelx, nbinx/2*pixelx, nbinx)
Yedges =  np.linspace(-nbiny/2*pixely, nbiny/2*pixely, nbiny)
Xn, Yn = np.meshgrid(Xedges, Yedges)
sigx = []
sigy = []
rmsx = []
rmsy = []
cenx = []
ceny = []

for i in range(len(filter_charge)):
#for i in range(1):
    # Image
    H = img[i][x_start:x_end, y_start:y_end]
    H = np.flipud(H)
    H1 = H
    H1 = H / max(map(max, H))
    H1[H1 < 0.0] = 0
    
    #plt.figure()
    #clsoplt.imshow(H1)
    
    hist_x = []
    for j in range(nbinx):
        hist_x.append(sum(H1[:,j]))
        
    hist_y = []
    for j in range(nbiny):
        hist_y.append(sum(H1[j,:]))

    # Normalization
    histx = hist_x / max(hist_x)
    histy = hist_y / max(hist_y)

    # Remove backgroud
    histx = histx - min(histx)
    histy = histy - min(histy)

    # Set x and y range
    x_h = np.linspace(-nbinx/2*pixelx, nbinx/2*pixelx, len(histx))
    y_h = np.linspace(-nbiny/2*pixely, nbiny/2*pixely, len(histy))
    
    # =====================================================================
    # =====================================================================
    # Gaussian fitting
    parametersx, covariancex = curve_fit(Gauss, x_h, histx)    
    fit_Ax = parametersx[0]
    fit_Bx = parametersx[1]
    fit_Cx = parametersx[2]
    sigx.append(parametersx[2])
    
    parametersy, covariancey = curve_fit(Gauss, y_h, histy)    
    fit_Ay = parametersy[0]
    fit_By = parametersy[1]
    fit_Cy = parametersy[2]
    sigy.append(parametersy[2])
    
    # Curve fitting comparison
    fitc_x = fit_Ax*np.exp(-((x_h-fit_Bx)/(np.sqrt(2)*fit_Cx))**2)
    fitc_y = fit_Ay*np.exp(-((y_h-fit_By)/(np.sqrt(2)*fit_Cy))**2)
    
    #fitc_xs = fit_Ax*np.exp(-((x_h-fit_Bx)/(np.sqrt(2)*0.2))**2)
    #fitc_ys = fit_Ay*np.exp(-((y_h-fit_By)/(np.sqrt(2)*0.14))**2)
    
    #plt.figure(i)
    #plt.plot(x_h, histx)
    #plt.plot(x_h, fitc_x)
    # =====================================================================
    # RMS size
    # Particle conversion
    import random
    # Random factor for generation
    rand_factor = 1E-2
    # =====================================================================
    # =====================================================================
    # Level summation
    Level_x = histx * 100
    Level_y = histy * 100
    # =====================================================================
    # =====================================================================
    # x axis random generation
    Xfinal = []
    x_gen = []
    for j in range(len(Level_x)):
        
        random_float_listx = []
        # Set a length of the list to 10
        for i in range(round(Level_x[j])):
            # any random float between -1E-2 to 1E-2
            # don't use round() if you need number as it is
            x = round(random.uniform(-rand_factor, rand_factor), 3)
            random_float_listx.append(x)
        
        x_gen.append(x_h[j]*(1+np.array(random_float_listx)))

    # import chain
    from itertools import chain
    Xfinal = list(chain.from_iterable(x_gen))
    
    Yfinal = []
    y_gen = []
    for j in range(len(Level_y)):
        
        random_float_listy = []
        # Set a length of the list to 10
        for i in range(round(Level_y[j])):
            # any random float between -1E-2 to 1E-2
            # don't use round() if you need number as it is
            y = round(random.uniform(-rand_factor, rand_factor), 3)
            random_float_listy.append(y)
        
        y_gen.append(y_h[j]*(1+np.array(random_float_listy)))

    # import chain
    from itertools import chain
    Yfinal = list(chain.from_iterable(y_gen))

    rmsx.append(np.sqrt( np.mean( ((Xfinal - np.mean(Xfinal))**2 ))))
    rmsy.append(np.sqrt( np.mean( ((Yfinal - np.mean(Yfinal))**2 ))))
    
    cenx.append(np.mean(Xfinal))
    ceny.append(np.mean(Yfinal))

# =====================================================================
# =====================================================================
# Final result
sigx = abs(np.array(sigx))
sigy = abs(np.array(sigy))

sigx_mean = sigx.mean()
sigy_mean = sigy.mean()

cenx = abs(np.array(cenx))
ceny = abs(np.array(ceny))

cenx_mean = cenx.mean()
ceny_mean = ceny.mean()


sigx_std  = np.std(np.array(sigx))
sigy_std  = np.std(np.array(sigy))

fwhmy_mean = sigy_mean*2.355
fwhmy_std  = sigy_std*2.355

rmsx = np.array(rmsx)
rmsy = np.array(rmsy)

rmsx_mean = rmsx.mean()
rmsy_mean = rmsy.mean()
rmsx_std  = np.std(np.array(rmsx))
rmsy_std  = np.std(np.array(rmsy))


# Font size
fsize = 25
rc('figure', figsize = (10,8))
rc('axes', grid = False)
rc('lines', linewidth = 2, color = 'r')
rc('font', size = fsize)

# Image and histogram
fig, ax1 = plt.subplots()
plt.pcolormesh(Xn, Yn, H1)
plt.set_cmap('viridis')
#plt.clim([0,0.3])
#plt.plot([1.51, 2.31],[3.41, 3.41], '-', linewidth=2.0, color=(1,1,1))
#plt.plot([1.51, 2.31],[2.88, 2.88], '--', linewidth=2.0, color=(1,1,1))
#plt.text(2.61, 3.31, 'Experiment', color = 'white', fontsize = 17)
#plt.text(2.61, 2.68, 'Gauss fit', color = 'white', fontsize = 17)

#plt.pcolormesh(Xn, Yn, data1)
plt.xlabel('$x$ (mm)', fontsize=fsize)
plt.ylabel('$y$ (mm)', fontsize=fsize), 
cbar = plt.colorbar()
cbar.ax.set_ylabel('Density', fontsize=fsize)

# Create a set of inset Axes: these should fill the bounding box allocated to
# them.
ax2 = plt.axes([0,0,1,1])
# Manually set the position and relative size of the inset axes within ax1
ip1 = InsetPosition(ax1, [0.0, 0.001, 1.0, 0.3])
ax2.set_axes_locator(ip1)

# The data: only display for low temperature in the inset figure.
ax2.plot(x_h, (histx/1)-((histx/1)[0]), '-', linewidth=1.4, color=(1,1,1))
ax2.plot(x_h, fitc_x, '--', linewidth=2, color='yellow')
#ax2.ylim([min(Xn[0,:]), max(Xn[0,:])])

ax2.axis([min(Yn[:,0]), max(Yn[:,0]), 0, 1])
ax2.axis('off')

# Create a set of inset Axes: these should fill the bounding box allocated to
# them.
ax3 = plt.axes([0,0,1,1])
# Manually set the position and relative size of the inset axes within ax1
ip3 = InsetPosition(ax1, [0.001, 0.0, 0.3, 1.0])
ax3.set_axes_locator(ip3)

# The data: only display for low temperature in the inset figure.
ax3.plot((histy/1)-((histy/1)[0]), y_h, '-', linewidth=1.4, color=(1,1,1))
ax3.plot(fitc_y, y_h, '--', linewidth=2, color='yellow')
#ax3.ylim([min(Yn[:,0]), max(Yn[:,0])])

ax3.axis([0, 1, min(Yn[:,0]), max(Yn[:,0])])
ax3.axis('off')
plt.tight_layout()
plt.show()

#plt.savefig(step+'.png', bbox_inches='tight')
