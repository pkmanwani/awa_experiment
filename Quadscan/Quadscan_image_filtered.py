# =====================================================================
# =====================================================================
# LPS plotting
# =====================================================================
# =====================================================================
# LPS plotting
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
step = str('p90')

beam_file = file_dir / './Quadscan_230A/p90.hdf5'
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
# Image check
#plt.figure()
#plt.imshow(img[0])




# =====================================================================
# =====================================================================
# Filter the charge 
set_charge = 1E-9 
devcharge = 0.03
filter_charge = np.where((charge > set_charge*(1-devcharge)) & (charge <set_charge*(1+devcharge)))[0]



# =====================================================================
# =====================================================================
# Filtering of the image
img_filtered = np.zeros((len(filter_charge), len(img[0][:,0]), len(img[0][0,:])))


for i in range(len(filter_charge)):
    img_filtered[i] = img[filter_charge[i]]


# =====================================================================
# =====================================================================
# Check the image
for i in range(len(filter_charge)):
    plt.figure(figsize=(16,10))
    plt.imshow(gaussian_filter(img[i], 0))
    plt.colorbar()
    #plt.clim(300,10000)
    plt.show()

# =====================================================================
# =====================================================================
# Crop the image

# =====================================================================
# Mesh range for plotting
x_start = 480
x_end   = 960 

y_start = 640
y_end   = 1180

nbiny = y_end-y_start
nbinx = x_end-x_start


#img_filtered_cropped = np.zeros((len(filter_charge)))
#for i in range(len(filter_charge)):
#    img_filtered_cropped[i] = img_filtered[i][x_start:x_end, y_start:y_end]



#plt.figure()
#plt.imshow(img_filtered_cropped[0])



#plt.close('all')

# =====================================================================
# =====================================================================
# Save image
np.save(step+'.npy', img_filtered)
















