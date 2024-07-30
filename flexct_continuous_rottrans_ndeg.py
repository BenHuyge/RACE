# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 11:02:17 2024

@author: Ben
"""

import astra
import flexraytools as flex
import numpy as np
from matplotlib import pyplot as plt
import pylops

from autograd.extend import primitive, defvjp
from autograd import numpy as anp
from autograd import grad
from autograd.scipy.special import logsumexp

# package for solvers (pip install git+https://gitlab.com/Visionlab-ASTRA/imsolve.git)
from imsolve import barzilai_borwein

from skimage.metrics import mean_squared_error as MSE

import pickle


##############################################################################
# Functions
#############################################################################

def callback_func(iteration, current_solution):
    # make a global variabla for RMSE to which to write the data
    global nmse
    
    img = np.reshape(current_solution, np.shape(phantom))
    
    if iteration == 0:
        nmse = []
        nmse.append(MSE(phantom, img)/np.var(img))    # divide by var(img) to get normalised MSE
    else:
        nmse.append(MSE(phantom, img)/np.var(img))

##############################################################################
# Variables
##############################################################################

n_iter = 108    # number of iterations to reconstruct   #60 for 1avg, 70 for 2 and 108 for 4

avg = 4  # average 2 projections together to double the angular blurring

n_cols = 2856

crop_box = np.array([[228,229], [0, n_cols]])

rec_vox = 1200 # size for reconstruction volume


rot_speed = 5.013996292373 # degrees per second
exposure = 0.1999996  # seconds
blur_ang = rot_speed* exposure *avg

# estimated center of rotation for FleXCT scan
COR = 1428.97 

# define the path where the data is located
path_ref = "D:\\Doctoraat_BenHuyge\\Data\\Continuous_acquisition\\20240111_reference_scan_JoS\\"
path_blur = "D:\\Doctoraat_BenHuyge\\Data\\Continuous_acquisition\\20240111_rot_trans_blur_JoS\\"

n_subsample = int(np.ceil((blur_ang*np.pi/180)* (rec_vox/2)))
print('Subsampling factor for reconstruction: ', n_subsample)


##############################################################################
# Load and reconstruct the reference scan
##############################################################################

print('Start reference reconstruction')

n_proj_ref, n_flat_ref, n_dark_ref = flex.parse_field(path_ref + "Acquisition settings XRE.txt", ['total projections',
                                                                                                  'number gain images',
                                                                                                  'number offset images'], int)

# use crop box to only load the central slice
# WARNING: this crop box is different than the crop box in the geometry parsing, which only works for 3D
data_ref = flex.load_data(path_ref + "scan_000000.tif", n_proj_ref, crop_box=crop_box)
flat_ref = flex.load_data(path_ref + "io000000.tif", n_flat_ref, crop_box=crop_box)
dark_ref = flex.load_data(path_ref + "di000000.tif", n_dark_ref, crop_box=crop_box)

data_ref = flex.log_correct(data_ref, flat_ref, dark_ref, astra_order=True)[0, :, :].copy()

proj_geom_ref = flex.geometry_parsing.parse_to_astra_geometry(path_ref + "Acquisition settings XRE.txt",
                                                              COR=COR,
                                                              central_slice=True,
                                                              verbose=True)

# create astra objects for volume and projection data
rec_vol_ref = astra.create_vol_geom(rec_vox, rec_vox)
rec_ref_id = astra.data2d.create("-vol", rec_vol_ref)
sino_ref_id = astra.data2d.create("-sino", proj_geom_ref, data_ref)

# reconstruct using non-negativity constraint
rec_ref = flex.BB(sino_ref_id, rec_ref_id, bounds=(0, np.inf), iterations=200, verbose=True)

# approximate phantom by good reconstruction to calculate RMSE etc.
global phantom
phantom = rec_ref.copy()

print("Reference reconstruction done")


##############################################################################
# Reconstruct the blurred scan (with translation)
##############################################################################

print("Start blurred reconstruction with translation")

# loading from file doesn't work for this dataset (again all zeros due to translation)
n_proj_blur = 360
n_flat_blur = 1
n_dark_blur = 1

data_blur = flex.load_data(path_blur + "scan_000000.tif", n_proj_blur, crop_box=crop_box)
flat_blur = flex.load_data(path_blur + "io000000.tif", n_flat_blur, crop_box=crop_box)
dark_blur = flex.load_data(path_blur + "di000000.tif", n_dark_blur, crop_box=crop_box)

data_blur = flex.log_correct(data_blur, flat_blur, dark_blur, astra_order=True)[0, :, :].copy()

# average projections to increase blur without doing a new scan
data_blur = -logsumexp(-data_blur.reshape(int(n_proj_blur/avg), avg, -1), b=1/avg, axis=1)


# load the fleXCT projection geometries with for the blurred scan
with open('FleXCT_geometries/proj_geom_blur_avg'+str(avg)+'.pkl', 'rb') as f:
    proj_geom_blur = pickle.load(f)


rec_vol_blur = astra.create_vol_geom(rec_vox, rec_vox)
rec_blur_id = astra.data2d.create("-vol", rec_vol_blur)
sino_blur_id = astra.data2d.create("-sino", proj_geom_blur, data_blur)

rec_blur = flex.BB(sino_blur_id, rec_blur_id, bounds=(0, np.inf), iterations=n_iter, verbose=True)

##############################################################################
# Reconstruct the blurred scan with translation with ARTIC
##############################################################################

print("Start ARTIC reconstruction")


# load the fleXCT projection geometries with rotation and translation offsets for ARTIC
with open('FleXCT_geometries/proj_geoms_artic_avg'+str(avg)+'.pkl', 'rb') as f:
    proj_geoms_artic = pickle.load(f)

# average the weights in the projection matrices
rec_vol_artic = astra.create_vol_geom(rec_vox, rec_vox)

for ii in range(n_subsample):
    proj_id = astra.create_projector('cuda', proj_geoms_artic[ii], rec_vol_artic)
    if ii==0:
      W_artic = pylops.LinearOperator(astra.OpTomo(proj_id))
    else:
        W_temp = pylops.LinearOperator(astra.OpTomo(proj_id))
        W_artic = W_artic + W_temp

W_artic = (1/n_subsample) * W_artic


@primitive
def proj_artic(x):
    return W_artic @ x
def vjp_proj_artic(ans, x):
    return lambda v: W_artic.T @ v
defvjp(proj_artic, vjp_proj_artic)

def f_artic(x):
    proj_diff = proj_artic(x).ravel() - data_blur.ravel()
    return 1/2*anp.dot(proj_diff, proj_diff)


x = barzilai_borwein(grad(f_artic), dim=rec_vox**2, bounds=(0,np.inf), max_iter=n_iter, verbose=True, callback=callback_func)
rec_artic = x.reshape((rec_vox, rec_vox))

nmse_artic = nmse.copy()

print('ARTIC reconstruction done')

##############################################################################
# Reconstruct the blurred scan with translation with RACE
##############################################################################

@primitive
def proj_race(x):
    return W_race @ x
def vjp_proj_race(ans, x):
    return lambda v: W_race.T @ v
defvjp(proj_race, vjp_proj_race)

def f_race(x):
    shape = (int(n_proj_blur/avg), n_subsample, -1)
    proj_diff = -logsumexp(-proj_race(x).reshape(shape), b=1/n_subsample, axis=1).ravel() - data_blur.ravel()
    return 1/2*anp.dot(proj_diff, proj_diff)


print("Start RACE reconstruction")

# load the fleXCT subsampled projection geometry
with open('FleXCT_geometries/proj_geom_race_avg'+str(avg)+'.pkl', 'rb') as f:
    proj_geom_race = pickle.load(f)

rec_vol_race = astra.create_vol_geom(rec_vox, rec_vox)
proj_id = astra.create_projector('cuda', proj_geom_race, rec_vol_race)
W_race = pylops.LinearOperator(astra.OpTomo(proj_id))
    
x = barzilai_borwein(grad(f_race), dim=rec_vox**2, bounds=(0,np.inf), max_iter=n_iter, verbose=True, callback=callback_func)
rec_race = x.reshape((rec_vox, rec_vox))

nmse_race = nmse.copy()

print('RACE reconstruction done')


##############################################################################
# Figures
##############################################################################
max_val = max(rec_blur.max(), rec_race.max(), rec_artic.max())

plt.figure();plt.imshow(rec_blur,cmap='grey', vmin=0,vmax=max_val);plt.axis('off');
plt.show()

plt.figure();plt.imshow(rec_artic,cmap='grey', vmin=0,vmax=max_val);plt.axis('off');
plt.show()

plt.figure();plt.imshow(rec_race,cmap='grey', vmin=0,vmax=max_val);plt.axis('off');
plt.show()

