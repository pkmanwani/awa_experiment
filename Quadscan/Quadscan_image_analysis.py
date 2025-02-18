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

ccount = [-150, -130, -110, -90, -69, -50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50, 60, 69, 80, 90, 100]

#p150 to n130, but polarity flipped >> so, it should be n150 to p130
sigx_oct = [2.4282739168148773, 2.2455200529483474, 1.9768628362922278, 1.7581224455862707, 1.5487011516418254, 1.3229489373737382,
            1.2630210200161427, 1.100876064610068, 1.034149577091972, 0.89464953961976, 0.7650400296457475, 0.7130377316714923,
            0.5434300126051961, 0.4732823604698746, 0.3404828781246524, 0.2604588839174492, 0.21184817758655405, 0.2253442720623709,
            0.27307745462063004, 0.3599003423431731, 0.451497166030065]

sigy_oct = [1.05046558973898, 0.8324053196658732, 0.6007605306304739, 0.38429189801097774, 0.2216740300350378, 0.195024095976856,
            0.2643985112331415, 0.3644691480932764, 0.4859707790652298, 0.5847492713038768, 0.6904022340459559, 0.8461444779180513,
            0.9243265868713297, 1.080379059642843, 1.1508993573035093, 1.2732603514099794, 1.3760125970232628, 1.5194932051317664,
            1.626933009700203, 1.723311939330537, 1.8815413214194974]

stdx_oct = [0.08923239166834017, 0.0, 0.07949503194173187, 0.007300787228181949, 0.03709864490500551, 0.024250050646386485,
            0.035262860522906725, 0.028812262218666528, 0.00000, 0.054027861463761026, 0.028093404423126016, 0.0375118993603041,
            0.036397780026447155, 0.01316912978805777, 0.012360917582413308, 0.000000, 0.008986568904885002, 0.014129990708127363,
            0.011578704587581246, 0.011873085573237784, 0.006215935851055741]

stdy_oct = [0.018689988817110766, 0.0, 0.004583158134035353, 0.009135456619712573, 0.014842882127514832, 0.007181973215404691,
            0.006269705794095054, 0.01393920838438542, 0.00000, 0.02628493470663567, 0.02538018365172112, 0.05021357164254611,
            0.04184827373191251, 0.026175965080750718, 0.040541913372725884, 0.000000, 0.03326165716590115, 0.045914124390537624,
            0.046693894167285206, 0.04890436701572715, 0.08346868687330657]

file_dir = Path('./')

# Step name
step = str('n100')

beam_file = file_dir / './Quadscan_230A/n100.hdf5'
bkg_file = file_dir / f"{beam_file.stem}-bkg.hdf5"
img_only = 0
assert beam_file.exists()
filename = beam_file

# =====================================================================
# =====================================================================
# Import data from hdf5 file
with h5py.File(f"{filename}","r") as f:
    img = f["image"][:]
    charge = f["charge"][:]
    bpmx = f["bpmx"][:]
    #scope_data = f["scope_data"][:]
    
# =====================================================================
# =====================================================================
# Filter the charge 
set_charge = 1E-9 
devcharge = 0.03
filter_charge = np.where((charge > set_charge*(1-devcharge)) & (charge <set_charge*(1+devcharge)))[0]

# =====================================================================
# =====================================================================
# Image check for cutting out
H = img[0]
H = np.flipud(H)
H1 = H / max(map(max, H))
H1[H1 < 0] = 0

nbinx = len(H1[0,:])
nbiny = len(H1[:,0])

# =====================================================================
# =====================================================================
# Calibration of the camera resolution
cal = 2.0815420560747665e-05

pixely = cal * 1e3 # Pixel number in millimeter
pixelx = cal * 1e3  # Pixel number in ps


Xedgesi =  np.linspace(-nbinx/2*pixelx, nbinx/2*pixelx, nbinx)
Yedgesi =  np.linspace(-nbiny/2*pixely, nbiny/2*pixely, nbiny)
Xn, Yn = np.meshgrid(Xedgesi, Yedgesi)


plt.figure()
plt.pcolormesh(H1)
plt.set_cmap('viridis')

plt.figure()
plt.pcolormesh(Xn,Yn,H1)
plt.set_cmap('viridis')


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

#x_start = 657
#x_end   = 816

#y_start = 820
#y_end   = 1040

x_start = 78
x_end   = 1115

y_start = 250
y_end   = 1600

nbinx = y_end - y_start
nbiny = x_end - x_start


Xedges =  np.linspace(-nbinx/2*pixelx, nbinx/2*pixelx, nbinx)
Yedges =  np.linspace(-nbiny/2*pixely, nbiny/2*pixely, nbiny)
Xn, Yn = np.meshgrid(Xedges, Yedges)
sigx = []
sigy = []
rmsx = []
rmsy = []
    
for i in range(len(filter_charge)):
#for i in range(1):
    # Image
    H = img[i][x_start:x_end, y_start:y_end]
    H = np.flipud(H)
    H1 = H / max(map(max, H))
    H1[H1 < 0] = 0
    
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
    
    

# =====================================================================
# =====================================================================
# Final result
sigx = abs(np.array(sigx))
sigy = abs(np.array(sigy))

sigx_mean = sigx.mean()
sigy_mean = sigy.mean()
sigx_std  = np.std(np.array(sigx))
sigy_std  = np.std(np.array(sigy))

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
#plt.plot([1.51, 2.31],[3.41, 3.41], '-', linewidth=2.0, color=(1,1,1))
#plt.plot([1.51, 2.31],[2.88, 2.88], '--', linewidth=2.0, color=(1,1,1))
#plt.text(2.61, 3.31, 'Experiment', color = 'white', fontsize = 17)
#plt.text(2.61, 2.68, 'Gauss fit', color = 'white', fontsize = 17)

#plt.pcolormesh(Xn, Yn, data1)
plt.xlabel('$x$ (mm)', fontsize=fsize)
plt.ylabel('$y$ (mm)', fontsize=fsize), 
cbar = plt.colorbar()
cbar.ax.set_ylabel('Normalized density', fontsize=fsize)

# Create a set of inset Axes: these should fill the bounding box allocated to
# them.
ax2 = plt.axes([0,0,1,1])
# Manually set the position and relative size of the inset axes within ax1
ip1 = InsetPosition(ax1, [0.0, 0.001, 1.0, 0.3])
ax2.set_axes_locator(ip1)

# The data: only display for low temperature in the inset figure.
ax2.plot(x_h, (histx/1)-((histx/1)[0]), '-', linewidth=1.4, color=(1,1,1))
ax2.plot(x_h, fitc_x, '--', linewidth=1.4, color=(1,1,1))
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
ax3.plot(fitc_y, y_h, '--', linewidth=1.4, color=(1,1,1))
#ax3.ylim([min(Yn[:,0]), max(Yn[:,0])])

ax3.axis([0, 1, min(Yn[:,0]), max(Yn[:,0])])
ax3.axis('off')
plt.tight_layout()
plt.show()

plt.savefig(step+'.png', bbox_inches='tight')

# =====================================================================
# =====================================================================
# RMS area
import seaborn as sns 

rmsx_area = np.trapz(histx)
b = 0
rmsx_68p = 0.682*rmsx_area

def masking():
    rmsx_area = np.trapz(histx)
    rmsx_68p = 0.682*rmsx_area
    # Tolerance setting
    tolerance = 1e-0
    b = 0
    while rmsx_area <= rmsx_68p:
        # =====================================================================
        # =====================================================================
        b = b - 1e-1*b
        print("The root estimate of tolerance T and mask shape f(x) are " + str(rmsx_area) + " and " + str(b))
        # x and y values for the trapezoid rule
        # Here, xt is actually the range for vertical axis
        # =====================================================================
        # =====================================================================\
        
        rmsx_area = np.trapz(histx[b:len(histx)-b])
        # =====================================================================
        # =====================================================================

masking.Tout = rmsx_area
masking.bout = b
print("The root estimate of tolerance T and mask shape f(x) are " + str(rmsx_area) + " and " + str(b))

# =====================================================================
# =====================================================================
# RMS area
from scipy import stats

# Font size
fsize = 25
rc('figure', figsize = (10,5))
rc('axes', grid = False)
rc('lines', linewidth = 2, color = 'r')
rc('font', size = fsize)

low_rmsx = x_h[75]
high_rmsx = x_h[len(x_h)-75]

colors = ['c', 'r', 'b', 'g', ]
colors = colors + list(reversed(colors))
plt.figure()
plt.plot(x_h, histx)
px = x_h[np.logical_and(x_h >= low_rmsx, x_h <= high_rmsx)]
plt.fill_between(px)
