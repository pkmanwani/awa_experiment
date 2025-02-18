import json
# =====================================================================
# =====================================================================
# LPS plotting
# =====================================================================
# =====================================================================
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
#file_path = os.path.join(data_path, 'sigma_values.json')  # replace with your actual file path
file_path = os.path.join(data_path, 'sigma_values.json')
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
from scipy.optimize import curve_fit
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
        is_valid = all(np.abs(point - p) > 1 for p in random_points)  # Ensure not too close to other points
        between_screens = any(min(s1, s2) < point < max(s1, s2) for s1, s2 in zip(sYAG[:-1], sYAG[1:]))
        if is_valid and between_screens:
            random_points.append(point)

    random_points = np.sort(random_points)
else:
    random_points=positions

sigma_x_random_points = spot_size_evolution(random_points, *parametersx)
sigma_y_random_points = spot_size_evolution(random_points, *parametersy)
print(sigma_x_random_points)
print(sigma_y_random_points)
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
print('Random points:',random_points)
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
print('Sigmas',sigma_x_random_points)
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
print(pCentral)
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

