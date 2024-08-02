# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 10:10:19 2024

@author: Ben
"""

# Script to test RACE on datasets acquired with parallel beam geometries, at a synchrotorn facility

# example datasets: https://tomobank.readthedocs.io/en/latest/source/data/docs.data.dynamic.html

# here we use the fuelcell dataset as example

import numpy as np
import matplotlib.pyplot as plt
import h5py
import astra
import pylops

from autograd.extend import primitive, defvjp
from autograd import numpy as anp
from autograd import grad
from autograd.scipy.special import logsumexp

# package for solvers (pip install git+https://gitlab.com/Visionlab-ASTRA/imsolve.git)
from imsolve import barzilai_borwein

from skimage.metrics import mean_squared_error as MSE


##############################################################################
# FUNCTIONS
##############################################################################

def get_W(n_vox, vox_size, n_pix, pix_size, angs, blur_corr = False):
    '''
    Get the projection matrix of a simulation.

    Parameters
    ----------
    n_vox : int
        Number of row or column voxels to reconstruct.
    vox_size : float
        Voxels size in mm.
    n_pix : int
        Number of detector pixels
    pix_size : float
        Pixel size in mm.
    angs : array
        Projection angles.
    blur_corr : boolean
        Blur correction, default False. Only used for reconstructiong the blurred dataset without ARTIC or RACE.
        Should be half the blurring interval, to align the rotation effect in the reconstruction. 
         
    Returns
    -------
    W_op : Linear Operator (pylops)
        System matrix.

     '''
    
    if blur_corr == True:
        angs = angs + (ang_range/n_angs)/2
    
    rec_vol = astra.create_vol_geom(n_vox, n_vox)
    proj_geom = astra.create_proj_geom('parallel', pix_size/vox_size, n_pix, angs)
    proj_id = astra.create_projector('cuda', proj_geom, rec_vol)
    W = astra.OpTomo(proj_id)
    
    # make an operator of W with pylops
    W_op = pylops.LinearOperator(W)
    
    return W_op


def get_W_artic(n_vox, vox_size, n_pix, pix_size, angs, n_subsample):
    '''
    Get the projection matrix used for ARTIC reconstruction. 
    Assumes 180 degree rotation.

    Parameters
    ----------
    n_vox : int
        Number of row or column voxels to reconstruct.
    vox_size : float
        Voxels size in mm.
    n_pix : int
        Number of detector pixels
    pix_size : float
        Pixel size in mm.
    angs : array
        Projection angles.
    n_subsample : int
        Number of angles used to subsample the angle step.

    Returns
    -------
    W_ARTIC : Linear Operator (pylops)
        System matrix for ARTIC.

    '''
    ang_step = angs[1]-angs[0]
    
    # starting angles should be divided over the first angle step (radians)
    start_angs = np.linspace(0, ang_step, n_subsample, endpoint=False)
    
    for ii in range(n_subsample):
        
        angles = angs + start_angs[ii]
        
        if ii==0:
            W_artic = get_W(n_vox, vox_size, n_pix, pix_size, angles)
        else:
            W_temp = get_W(n_vox, vox_size, n_pix, pix_size, angles)
            W_artic = W_artic + W_temp
    
    W_artic = (1/n_subsample) * W_artic
    
    return W_artic


def solve_BB(W, sino, n_iter, callback = None):
    '''
    Use Barzilai-Borwein to reconstruct the image

    Parameters
    ----------
    W : pylops operator
        Projection matrix.
    sino : 2D-array (n_angs x n_pix)
        Sinogram with the projection data.
    n_iter : int
        Number of iterations to use.    
    callback: function
        Use the callback_function or not    

    Returns
    -------
    rec: 2D-array (n_pix x n_pix)
        The reconstructed image.

    '''
    n_angs, n_pix = np.shape(sino)
    _, n_vox_tot = np.shape(W)
    
    n_vox = int(np.sqrt(n_vox_tot))

    @primitive
    def proj(x):
        return W @ x
    def vjp_proj(ans, x):
        return lambda v: W.T @ v
    defvjp(proj, vjp_proj)

    def f(x):
        proj_diff = proj(x) - sino.ravel()
        return 1/2*anp.dot(proj_diff, proj_diff)

    x = barzilai_borwein(grad(f), dim=n_vox**2, bounds=(0,np.inf), max_iter=n_iter, verbose=True, callback=callback)
    rec = x.reshape((n_vox, n_vox))
    
    return rec

def callback_func(iteration, current_solution):
    # make a global variabla for RMSE to which to write the data
    global nmse
    
    # do not divide by 2 here, only for simulations
    img = np.reshape(current_solution, np.shape(phantom))
    
    if iteration == 0:
        nmse = []
        nmse.append(MSE(phantom, img)/np.var(img))    # divide by var(img) to get normalised MSE
    else:
        nmse.append(MSE(phantom, img)/np.var(img))
        
        
        
def get_parallel_geom(angs, n_pix, pix_size, vox_size, cor):
    """
    # to generate a parallel beam geometry with COR integration (does not work properly yet)
    
    """

    # create geometry
    projection_vectors = np.zeros((len(angs), 6))

    # position of source, detector and rotation axis
    source = np.array([-1, 0])*pix_size/vox_size
    det_center = np.array([1, 0])*pix_size/vox_size
    rotation_axis = np.array([0.0, source[1]-(cor-(n_pix/2))])*pix_size/vox_size

    # vector from detector pixel (0,0) to (0,1)
    det_x = np.array([0.0, -1])*pix_size/vox_size

    # rotation matrix about the z-axis
    def rot_matrix(t):
        return np.array([[np.cos(t), -np.sin(t)],
                         [np.sin(t), np.cos(t)]])

    for i, angle in enumerate(angs):
        # rotate source position about the rotation axis
        projection_vectors[i, 0:2] = rot_matrix(angle) @ (source - rotation_axis) + rotation_axis
        projection_vectors[i, 2:4] = rot_matrix(angle) @ (det_center - rotation_axis) + rotation_axis
        
        # rotate detector direction
        projection_vectors[i, 4:6] = rot_matrix(angle) @ det_x

    # astra can only work with voxel size 1. We scale the whole geometry to accommodate this.
    proj_geom = {'DetectorCount': n_pix, 'type': 'parallel_vec', 'Vectors': projection_vectors}
    return proj_geom


##############################################################################
# VARIABLES
##############################################################################

n_iters = 50

path = "D:\\Doctoraat_BenHuyge\\Data\\Continuous_acquisition\\Synchrotron_data\\fuelcell_dryHQ_i1.h5"

ang_range = np.pi# radians

pix_size = 0.00275 # mm

n_vox = 1440
vox_size = 0.00275 # mm

blur_extra = False
avg = 20

ang_offset = 23*np.pi/180 # to get structure horizontal

# center of rotation
cor = 702



##############################################################################
# LOAD THE DATA
##############################################################################

data_file = h5py.File(path,'r')
print(list(data_file.keys()))

exchange = data_file['exchange']
print(list(exchange))

imgs = np.array(exchange['data'])
n_angs, n_rows, n_cols = imgs.shape
dark_fields = np.squeeze(np.array(exchange['data_dark']))
flat_fields = np.array(exchange['data_white'])

# only do central slice reconstruction for test
dark_fields = dark_fields[:, int(n_rows/2), :]
flat_fields = flat_fields[:, int(n_rows/2), :]
imgs = imgs[:, int(n_rows/2), :]

# do flat field correction, calculate average flatfield for test
flat_field = np.mean(flat_fields, axis=0)
dark_field = np.mean(dark_fields, axis=0)

projs = -np.log( (imgs-dark_field)/(flat_field-dark_field) )
projs[projs<0]=0    # remove negative values


# hacky way to get COR correct (702 according to the description on the website). 
# Normally, the COR is taken into account in the prjection geometry, 
# but I have no time left to implement that now
projs = projs[:,:-36]
n_cols = n_cols-36



##############################################################################
# RECONSTRUCT REFERENCE
##############################################################################
n_pix = n_cols

angs = np.linspace(0, ang_range, n_angs, endpoint=False) + ang_offset


# the reference reconstruction is technically also blurred, but negligible

print('Start reference reconstruction')

W_ref = get_W(n_vox, vox_size, n_pix, pix_size, angs, blur_corr=True)

rec_ref = solve_BB(W_ref, projs, n_iters)

# approximate phantom by good reconstruction to estimate NMSE 
global phantom
phantom = rec_ref.copy()


print('reference reconstruction done')


##############################################################################
# RECONSTRUCT BLURRED
##############################################################################


# itroduce extra blurring
if blur_extra == True:
    
    # remove last projection, because odd number gives problems for averaging
    projs = projs[:-1,:]
    print(projs.shape)
    
    projs = -logsumexp(-(projs.ravel()).reshape(int(n_angs/avg), avg, -1), b=1/avg, axis=1)
    n_angs = int(n_angs/avg)
    
    
blur_ang = ang_range/n_angs    # angle over which is integrated

# number of subsample steps for the reconstruction
n_subsample_rec = int(np.ceil(blur_ang* (n_vox/2))) 
print('Subsampling factor for reconstruction: ', n_subsample_rec)


angs_blur = np.linspace(0, ang_range, n_angs, endpoint=False) + ang_offset

print('Start blurred reconstruction')

W_blur = get_W(n_vox, vox_size, n_pix, pix_size, angs_blur, blur_corr=True)

rec_blur = solve_BB(W_blur, projs, n_iters, callback=callback_func)

nmse_blur = nmse.copy()


print('Blurred reconstruction done')

##############################################################################
# RECONSTRUCT ARTIC
##############################################################################

print('Start ARTIC reconstruction')

W_artic = get_W_artic(n_vox, vox_size, n_pix, pix_size, angs_blur, n_subsample_rec)

rec_artic = solve_BB(W_artic, projs, n_iters, callback=callback_func)

nmse_artic = nmse.copy()


print('ARTIC reconstruction done')


##############################################################################
# RECONSTRUCT RACE
##############################################################################

angs_race = np.linspace(0, ang_range, n_angs*n_subsample_rec, endpoint=False) + ang_offset


print('Start RACE reconstruction')

# make a sub-sampled projection matrix
W_corr = get_W(n_vox, vox_size, n_pix, pix_size, angs_race)

@primitive
def proj_corr(x):
    return W_corr @ x
def vjp_proj_corr(ans, x):
    return lambda v: W_corr.T @ v
defvjp(proj_corr, vjp_proj_corr)

def f_corr(x):
    shape = (n_angs, n_subsample_rec, -1)
    proj_diff = -logsumexp(-proj_corr(x).reshape(shape), b=1/n_subsample_rec, axis=1).ravel() - projs.ravel()
    return 1/2*anp.dot(proj_diff, proj_diff)

x = barzilai_borwein(grad(f_corr), dim=n_vox**2, bounds=(0,np.inf), max_iter=n_iters, verbose=True, callback=callback_func)
rec_race = x.reshape((n_vox, n_vox))

nmse_race = nmse.copy()


print('RACE reconstruction done')



##############################################################################
# FIGURES
##############################################################################
plt.rcParams.update({'font.size': 22})
plt.rcParams["figure.figsize"] = (10,10)

max_val = max(rec_ref.max(), rec_blur.max(), rec_race.max(), rec_artic.max())


plt.figure()
plt.imshow(rec_ref, cmap='grey', vmin = 0, vmax=max_val)
plt.title('Reference reconstruction');plt.axis('off');
plt.show()

plt.figure()
plt.imshow(rec_blur, cmap='grey', vmin = 0, vmax=max_val)
plt.title('Blurred reconstruction');plt.axis('off');
plt.show()

plt.figure()
plt.imshow(rec_artic, cmap='grey', vmin = 0, vmax=max_val)
plt.title('ARTIC reconstruction');plt.axis('off');
plt.show()

plt.figure()
plt.imshow(rec_race, cmap='grey', vmin = 0, vmax=max_val)
plt.title('RACE reconstruction');plt.axis('off');
plt.show()

plt.figure()
plt.plot(nmse_blur,'r')
plt.plot(nmse_artic,'b')
plt.plot(nmse_race,'g')
plt.show()


diff_artic = rec_ref-rec_artic
diff_race = rec_ref-rec_race

min_val2 = min(diff_artic.min(), diff_race.min())
max_val2 = max(diff_artic.max(), diff_race.max())

plt.figure()
plt.imshow(diff_artic, cmap='jet', vmin = min_val2, vmax=max_val2);plt.colorbar(fraction = 0.046, pad=0.04)
plt.title('ARTIC difference');plt.axis('off');
plt.show()

plt.figure()
plt.imshow(diff_race, cmap='jet', vmin = min_val2, vmax=max_val2);plt.colorbar(fraction = 0.046, pad=0.04)
plt.title('RACE difference');plt.axis('off');
plt.show()


