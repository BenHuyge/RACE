# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 15:11:08 2023

@author: Ben
"""

import numpy as np
import astra
import pylops
import matplotlib.pyplot as plt
import skimage as ski

from autograd.extend import primitive, defvjp
from autograd import numpy as anp
from autograd import grad

# package for solvers (pip install git+https://gitlab.com/Visionlab-ASTRA/imsolve.git)
from imsolve import barzilai_borwein

from skimage.metrics import mean_squared_error as MSE

from autograd.scipy.special import logsumexp


##############################################################################
# FUNCTIONS
##############################################################################

def get_W(n_vox, vox_size, n_pix, pix_size, n_angs, ang_range, src_obj_dist, obj_det_dist, ang_start=0, blur_corr = False):
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
    n_angs : int
        Number of projection angles.
    ang_range : float
        Angular range over which the angles are equally distributed.
    src_obj_dist : float
        Distance between source and object in mm.
    obj_det_dist : float
        Distance between object and detector in mm.
    ang_start : float
        Starting angle (radians), default 0. Only used for subsampling angles with ARTIC
    blur_corr : boolean
        Blur correction, default False. Only used for reconstructiong the blurred dataset without ARTIC or RACE.
        Should be half the blurring interval, to align the rotation effect in the reconstruction. 
         
    Returns
    -------
    W_op : Linear Operator (pylops)
        System matrix.

     '''
    angs = np.linspace(0, ang_range, n_angs, endpoint=False) + ang_start
    
    if blur_corr == True:
        angs = angs + (ang_range/n_angs)/2
    
    rec_vol = astra.create_vol_geom(n_vox, n_vox)
    proj_geom = astra.create_proj_geom('fanflat', pix_size/vox_size, n_pix, angs, src_obj_dist/vox_size, obj_det_dist/vox_size)
    proj_id = astra.create_projector('cuda', proj_geom, rec_vol)
    W = astra.OpTomo(proj_id)
    
    # make an operator of W with pylops
    W_op = pylops.LinearOperator(W)

    return W_op


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

def get_W_artic(n_vox, vox_size, n_pix, pix_size, n_angs, ang_range, src_obj_dist, obj_det_dist, n_subsample):
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
    n_angs : int
        Number of projection angles.
    ang_range : float
        Angular range over which the angles are equally distributed.
    src_obj_dist : float
        Distance between source and object in mm.
    obj_det_dist : float
        Distance between object and detector in mm.
    n_subsample : int
        Number of angles used to subsample the angle step.

    Returns
    -------
    W_ARTIC : Linear Operator (pylops)
        System matrix for ARTIC.

    '''
    ang_step = ang_range/n_angs
    
    # starting angles should be divided over the first angle step (radians)
    start_angs = np.linspace(0, ang_step, n_subsample, endpoint=False)
    
    for ii in range(n_subsample):
        if ii==0:
            W_artic = get_W(n_vox, vox_size, n_pix, pix_size, n_angs, ang_range, src_obj_dist, obj_det_dist, ang_start=start_angs[ii])
        else:
            W_temp = get_W(n_vox, vox_size, n_pix, pix_size, n_angs, ang_range, src_obj_dist, obj_det_dist, ang_start=start_angs[ii])
            W_artic = W_artic + W_temp
    
    W_artic = (1/n_subsample) * W_artic
    
    return W_artic

      
def callback_func(iteration, current_solution):
    # make a global variabla for NMSE to which to write the data
    global nmse
    
    # divide by 2 to counter weird astra bug of multiplying the intensity of the reconstructed image
    # by 2, because the resolution of the reconstruction grid is lower (1400->700)
    img = np.reshape(current_solution/2, (700,700))
    
    if iteration == 0:
        nmse = []
        nmse.append(MSE(phantom, img)/np.var(img))     # divide by var(img) to get normalised MSE
    else:
        nmse.append(MSE(phantom, img)/np.var(img))


def poisson(sino, photon_count):
    '''
    Add Poisson noise to the projections.

    Parameters
    ----------
    sino : 2D-array
        Sinogram without noise.
    photon_count : int
        Number of detected photons in the flatfield.

    Returns
    -------
    noisy_sino : 2D-array
        Sinogram with added Poisson noise.

    '''
    # convert sino to normalized intensities
    sino_I_norm = np.exp(-sino)
    # add the photon count
    sino_I = photon_count * sino_I_norm
    
    noisy_sino_I = np.random.poisson(sino_I)
    
    # convert back to projection sinogram
    noisy_sino = -np.log(noisy_sino_I/photon_count)
    
    # make sure no negative values
    noisy_sino[noisy_sino<0]=0
    
    return noisy_sino

def upscale(img, factor):
    '''
    Increase the resolution of an image.

    Parameters
    ----------
    img : 2D-array
        Array with the image.
    factor : int
        Factor to increase the resolution.

    Returns
    -------
    img_highres : 2D-array
        The image with increased resolution.

    '''
    rows_lowres, cols_lowres = np.shape(img)
    
    img_highres = np.zeros(np.array(np.shape(img))*factor)
    
    for ii in range(rows_lowres):
        for jj in range(cols_lowres):
            img_highres[ii*2, jj*2] = img[ii, jj]
            img_highres[ii*2+1, jj*2] = img[ii, jj]
            img_highres[ii*2, jj*2+1] = img[ii, jj]
            img_highres[ii*2+1, jj*2+1] = img[ii, jj]
    
    return img_highres

##############################################################################
# Variables
##############################################################################

n_angs = 40         # number of projections
n_iters = 2000      # number of iterations

noise = True       # to add noise or not   
I_0 =  10**5       # photon count for Poisson


obj_det_dist = 250
src_obj_dist = 500         # all distance in mm
M = (src_obj_dist + obj_det_dist)/src_obj_dist

n_pix = 1250      # number of detector pixels
pix_size = 0.15

phantom_name = 'phantom2'

global phantom # make phantom a global variable, so it can easily be used in the callback functions to calculate MSE etc.
phantom = (np.squeeze(ski.io.imread('Phantoms/'+phantom_name+'.png')[:,:,0])/255) * 0.01

# upscale phantom to double resolution before simulation
phantom_sim = upscale(phantom,2)

# use for blurring
n_blur_steps = 1000    # 1000 projections between the true projections to introduce blurring 

n_vox_sim = 1400      # number of row voxels of the phantom (square image)
n_vox_rec = 700      # number of row and column voxel in the reconstruction

vox_size_sim = (pix_size/M)/2
vox_size_rec = (pix_size/M)#vox_size_sim *2  # larger voxels, so grid covers the same area

# correct angular range for fan angle (parker1982)
fan_ang = 2*np.arctan((n_pix/2)*pix_size/(obj_det_dist+src_obj_dist))
ang_range = np.pi + fan_ang

# number of subsample steps for the reconstruction
n_subsample_rec = int(np.ceil((ang_range/n_angs)* (n_vox_rec/2))) 
print('Subsampling factor for reconstruction: ', n_subsample_rec)

#############################################################################
# SIMULATE PROJECTIONS
#############################################################################

print('Start simulation')

n_angs_sim = n_angs*n_blur_steps

angs_sim = np.linspace(0, ang_range, n_angs_sim, endpoint=False)

vol_geom_sim = astra.create_vol_geom(n_vox_sim, n_vox_sim)
proj_geom_sim = astra.create_proj_geom('fanflat', pix_size/vox_size_sim, n_pix, angs_sim, src_obj_dist/vox_size_sim, obj_det_dist/vox_size_sim)
proj_id_sim = astra.create_projector('cuda', proj_geom_sim, vol_geom_sim)
W_sim = astra.OpTomo(proj_id_sim)

# do a forward projection to get the full projection data
projs_sim = W_sim @ phantom_sim.ravel()

sino_sim = projs_sim.reshape(n_angs_sim, -1)

print('Simulation done')


##############################################################################
# RECONSTRUCT BLURRED
##############################################################################

print('Start blurred reconstruction')

# blur the projections by averaging intensities
projs_blur = -logsumexp(-projs_sim.reshape(n_angs, n_blur_steps, -1), b=1/n_blur_steps, axis=1).ravel()
# due to some numerical instability 1e-16 negative numbers can appear in the background, put these to zero
projs_blur[projs_blur<0]=0

sino_blur = projs_blur.reshape(n_angs, -1)
if noise == True:
    # add Poisson noise
    sino_blur = poisson(sino_blur, I_0)
   
W_blur = get_W(n_vox_rec, vox_size_rec, n_pix, pix_size, n_angs, ang_range, src_obj_dist, obj_det_dist, blur_corr=True)

rec_blur = solve_BB(W_blur, sino_blur, n_iters, callback=callback_func)

nmse_blur = nmse.copy()

print('Blurred reconstruction done')

##############################################################################
# RECONSTRUCT ARTIC
##############################################################################

print('Start ARTIC reconstruction')

W_artic = get_W_artic(n_vox_rec, vox_size_rec, n_pix, pix_size, n_angs, ang_range, src_obj_dist, obj_det_dist, n_subsample_rec)

rec_artic = solve_BB(W_artic, sino_blur, n_iters, callback=callback_func)

nmse_artic = nmse.copy()

print('ARTIC reconstruction done')

##############################################################################
# RECONSTRUCT RACE
##############################################################################

print('Start RACE reconstruction')

# make a sub-sampled projection matrix
W_corr = get_W(n_vox_rec, vox_size_rec, n_pix, pix_size, n_angs*n_subsample_rec, ang_range, src_obj_dist, obj_det_dist)

@primitive
def proj_corr(x):
    return W_corr @ x
def vjp_proj_corr(ans, x):
    return lambda v: W_corr.T @ v
defvjp(proj_corr, vjp_proj_corr)

def f_corr(x):
    shape = (n_angs, n_subsample_rec, -1)
    proj_diff = -logsumexp(-proj_corr(x).reshape(shape), b=1/n_subsample_rec, axis=1).ravel() - sino_blur.ravel()
    return 1/2*anp.dot(proj_diff, proj_diff)

x = barzilai_borwein(grad(f_corr), dim=n_vox_rec**2, bounds=(0,np.inf), max_iter=n_iters, verbose=True, callback=callback_func)
rec_race = x.reshape((n_vox_rec, n_vox_rec))

nmse_race = nmse.copy()

print('RACE reconstruction done')


##############################################################################
# counter weird astra bug, where the intensity of the reconstructed image is multiplied by a factor 2
# because the reconstruction has a lower resolution voxel grid than the simulated data
# 1400 vox -> 700 vox

rec_blur = rec_blur/2
rec_artic = rec_artic/2
rec_race = rec_race/2


##############################################################################
# FIGURES
##############################################################################


max_val = max(rec_blur.max(), rec_artic.max(), rec_race.max())

plt.figure();
plt.imshow(rec_blur, cmap='grey', vmin=0, vmax=max_val);plt.axis('off');
plt.show()

plt.figure();plt.imshow(rec_artic, cmap='grey', vmin=0, vmax=max_val);plt.axis('off');
plt.show()

plt.figure();plt.imshow(rec_race, cmap='grey', vmin=0, vmax=max_val);plt.axis('off');
plt.show()



