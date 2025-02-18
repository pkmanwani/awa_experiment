# =====================================================================
# =====================================================================
# LPS plotting
# =====================================================================
# =====================================================================
# LPS plotting

# LPS plotting
import h5py
import numpy as np
import matplotlib.pyplot as plt
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
import matplotlib.patches as patches
from scipy.optimize import curve_fit

import h5py
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter


import matplotlib.pyplot as plt
from matplotlib import rc
import scipy as sc
import matplotlib.style
import matplotlib.pyplot as plt
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
emass = m0;
clite = c;

# Energy and gamma
Ebeam1   = 45.3E6 #; %//initial energy in GeV
gamma1   = (Ebeam1+EMASS)/EMASS
beta1    = np.sqrt(1-(1/gamma1**2))
P01      = gamma1*beta1*mc2
pCentral    = P01/EMASS;


# Quadrupole count
Horcount = np.linspace(-450, +10, 11)
Vercount = np.linspace(-250, 450, 11)

# Count to T/m
#count_tm = np.array(count) * 7.5e-3 / (0.893 * 0.75)
horcount_tm = np.array(Horcount) * 8.93e-3
vercount_tm = np.array(Vercount) * 8.93e-3


# T/m to m^-2
kvalh = np.array(horcount_tm) * (1/((beta1 * Ebeam1*1e-9) / 0.299))
kvalv = np.array(vercount_tm) * (1/((beta1 * Ebeam1*1e-9) / 0.299))

# =====================================================================
# =====================================================================
quad_length = 0.12
drift_length = 0.06+0.265+0.255+0.27+0.54

#sigx = [0.3510960111642836, 0.21282696832391765, 0.2578194073324571, 0.4688587084545032, 0.705472424826568, 0.7572517346816042, 0.8863599533147581, 1.0884817300720848, 1.3059218603597031, 1.5267260229108128, 1.7326940044665162, 1.952851126771961, 2.21096225226397]
#sigy = [1.6719495731625083, 1.4926098154695975, 1.225857225644446, 1.0522121152715849, 0.8260596481370701, 0.676410828589374, 0.5735582990376374, 0.35951748468907263, 0.19288360546728966, 0.22026737469107707, 0.380621546024965, 0.5942791386376833,0.8204046594351598 ]
#stdx = [0.012697241168706158, 0.00000000000000, 0.00000000000000, 0.012498087515683532, 0.035456892849381735, 0.027699616162490198, 0.05205711513945887, 0.02926990427348568, 0.022485655833590103, 0.03171697308625488, 0.007453393631018379, 0.0782127348725037, 0]
#stdy = [0.02789170681618991, 0.00000000000000, 0.00000000000000, 0.02473285343150758, 0.04283189184664075, 0.02515626603979119, 0.025334543935150312, 0.01377151758931454, 0.007101631753613931, 0.014985924013615504, 0.009671917891592219, 0.005088348094770723, 0]


#p150 to n130, but polarity flipped >> so, it should be n150 to p130
sigx = [3.0826, 2.6727, 2.3681, 2.0351, 1.8029, 1.7370, 1.7456, 1.9395, 2.2844, 2.6555, 3.1333]

sigy = [0.8106, 0.5916, 0.5623, 0.4627, 0.3384, 0.5260, 0.5222, 0.6577, 0.8461, 0.8489, 1.2391]

stdx = [0.04011, 0.01815, 0.01299, 0.00630, 0.00709, 0.00814, 0.01068, 0.02062, 0.01129, 0.006]

stdy = [0.10700, 0.02381, 0.03154, 0.01961, 0.01, 0.01718, 0.01, 0.03, 0.02167, 0.03, 0.057]

# =====================================================================
# =====================================================================
# optimization
def lq(x, a, b, c):
    return (a*(x**2) + b*x + c)


#sigx = np.flipud(np.array(sigx))
#sigy = np.flipud(np.array(sigy))
#stdx = np.flipud(np.array(stdx))
#stdy = np.flipud(np.array(stdy))

# =====================================================================
# =====================================================================

sigx_sqr = np.array(sigx)*1e-3
sigx_sqr = sigx_sqr**2

sigy_sqr = np.array(sigy)*1e-3
sigy_sqr = sigy_sqr**2

stdx_sqr = np.array(stdx)*1e-3
stdx_sqr = stdx_sqr**2

stdy_sqr = np.array(stdy)*1e-3
stdy_sqr = stdy_sqr**2

#start = 4, end = 0 for vertical
#

starth = 1
endh = len(sigx_sqr) - 1

startv = 2
endv = len(sigy_sqr) - 0


sigx_sqr = sigx_sqr[starth:endh]
sigy_sqr = sigy_sqr[startv:endv]
stdx_sqr = stdx_sqr[starth:endh]
stdy_sqr = stdy_sqr[startv:endv]
kvalh = kvalh[starth:endh]
kvalv = kvalv[startv:endv]

parametersx, covariancex = curve_fit(lq, kvalh, sigx_sqr)
ax = parametersx[0]
bx = parametersx[1]
cx = parametersx[2]

# =====================================================================
# =====================================================================
parametersy, covariancey = curve_fit(lq, kvalv, sigy_sqr)
ay = parametersy[0]
by = parametersy[1]
cy = parametersy[2]

# =====================================================================
# =====================================================================
kval_fith = np.linspace(min(kvalh), max(kvalh), 500)
kval_fitv = np.linspace(min(kvalv), max(kvalv), 500)
fitx = (ax*(kval_fith**2) + bx*kval_fith + cx)
fity = (ay*(kval_fitv**2) + by*kval_fitv + cy)

# =====================================================================
# =====================================================================
# Emittance calculation
sq11_x = ax / ((drift_length**2) * (quad_length**2));
sq12_x = (bx - (2*drift_length*quad_length*sq11_x)) / (2*(drift_length**2) * quad_length);
sq21_x = sq12_x;
sq22_x = ( (cx - sq11_x - (2*drift_length*sq12_x)) ) / (drift_length**2);

# Calculation of the geometrical emittance
ex = np.sqrt( ((sq11_x * sq22_x) - (sq12_x**2)));

# Calculation of the normalized emittance
enx = (pCentral*ex);

sq11_y = ay / ((drift_length**2) * (quad_length**2));
sq12_y = (by - (2*drift_length*quad_length*sq11_y)) / (2*(drift_length**2) * quad_length);
sq21_y = sq12_y;
sq22_y = ( (cy - sq11_y - (2*drift_length*sq12_y)) ) / (drift_length**2);

# Calculation of the geometrical emittance
ey = np.sqrt( ((sq11_y * sq22_y) - (sq12_y**2)));

# Calculation of the normalized emittance
eny = (pCentral*ey);


# =====================================================================
# =====================================================================
# Twiss parameters
alpha_x = -sq12_x/ex
beta_x = sq11_x/ex

alpha_y = -sq12_y/ex
beta_y = sq11_y/ex

print('========================================')
print('enx is ' + repr(enx*1e6)+ ' mm mrad.')
print('eny is ' + repr(eny*1e6)+ ' mm mrad.')
print('========================================')
print('betax at the initial position is  '+repr(beta_x)+ ' m.')
print('betay at the initial position is  '+repr(beta_y)+ ' m.')
print('alphax at the initial position is '+repr(alpha_x)+ ' .')
print('alphay at the initial position is '+repr(alpha_y)+ ' .')
print('========================================')

# ==========================================================================
# Figure setting
fonts = 25
# ==========================================================================
# Figure setting
plt.style.use('classic')
rc = {"font.family" : "Arial"}
plt.rcParams.update(rc)
# ==========================================================================
# ==========================================================================
params = {'legend.fontsize': 18,
          'axes.labelsize': 25,
          'axes.titlesize': 25,
          'xtick.labelsize' :25,
          'ytick.labelsize': 25,
          'grid.color': 'k',
          'grid.linestyle': ':',
          'grid.linewidth': 1.5
         }
matplotlib.rcParams.update(params)


#ccode3 = (0.47,0.67,0.23)
ccode1 = [40,122,169]
ccode2 = [120,130,46]
ccode3 = [120,175,59]
ccode4 = [80,80,80]

ccode1 = tuple(np.array(ccode1)/255)
ccode2 = tuple(np.array(ccode2)/255)
ccode3 = tuple(np.array(ccode3)/255)
ccode4 = tuple(np.array(ccode4)/255)

# Linewidth
line_width = 2.3


fig, ax1 = plt.subplots()
fig.patch.set_facecolor('white')
fig.set_size_inches(10,9)
# ==========================================================================
# ax1 plot
p1 = ax1.scatter(kvalh, np.array(sigx_sqr)*1e6, s=40)
p2 = ax1.errorbar(kvalh, np.array(sigx_sqr)*1e6, (stdx_sqr)*1e6, linewidth=0.5, linestyle='--', label=r'$\sigma_{x}^{2},~\mathrm{Measurement}$')
p3 = ax1.plot(kval_fith, np.array(fitx)*1e6, '-',  linewidth=line_width, label=r'$\sigma_{x}^{2},~\mathrm{Curve~fitting}$')
#p1[-1][0].set_linestyle('--')
# ==========================================================================
# ==========================================================================
color = ccode1
ax1.set_xlabel(r'$\mathrm{k~(m^{-2})}$')
ax1.set_ylabel(r'$\sigma_{x}^{2}~\mathrm{(mm^{2})}$', color=color)
ax1.tick_params(axis='y', labelcolor=color)
#ax1.axis([-8, 10, 0, 6])
ax1.legend(loc='best')
ax1.grid()
plt.tight_layout()
plt.show()
#plt.savefig('quadscan_x.png', bbox_inches='tight')



#plt.errorbar(X, Y, Z, label='data')
fig, ax1 = plt.subplots()
fig.patch.set_facecolor('white')
fig.set_size_inches(10,9)
# ==========================================================================
# ax1 plot
p1 = ax1.scatter(kvalv, np.array(sigy_sqr)*1e6, s=40)
p2 = ax1.errorbar(kvalv, np.array(sigy_sqr)*1e6, np.array(stdy_sqr)*1e6, linewidth=0.5, linestyle='--', label=r'$\sigma_{y}^{2},~\mathrm{Measurement}$')
p3 = ax1.plot(kval_fitv, np.array(fity)*1e6, '-',  linewidth=line_width, label=r'$\sigma_{y}^{2},~\mathrm{Curve~fitting}$')
#p1[-1][0].set_linestyle('--')
# ==========================================================================
# ==========================================================================
color = ccode1
ax1.set_xlabel(r'$\mathrm{k~(m^{-2})}$')
ax1.set_ylabel(r'$\sigma_{y}^{2}~\mathrm{(mm^{2})}$', color=color)
ax1.tick_params(axis='y', labelcolor=color)
#ax1.axis([-6, 1, 0, 1])
ax1.legend(loc='upper left'   )
ax1.grid()
plt.tight_layout()
plt.show()








