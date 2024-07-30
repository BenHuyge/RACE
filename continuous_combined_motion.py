# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:39:18 2023

@author: Ben
"""

import numpy as np
import astra
import pylops
import matplotlib.pyplot as plt
import imageio.v3 as iio
from scipy.spatial.transform import Rotation as rot

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
def get_W_sim(n_vox, vox_size, n_pix, pix_size, n_angs, ang_range, src_obj_dist, obj_det_dist, blur_shift_size, blur_ang_size, n_subsample):
    '''
    Generate a projection matrix for a rotation where the object is shifted 
    parallel to the detector by some distance each projection (compounding!).
    Also additional rotation blur is added.

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
    blur_shift_size : float
        The size of the object shift each substep in mm.
    blur_ang_size : float
        The size of the angle rotation each substep in mm.
    n_subsample : int
        Number of subsample steps each angle.

    Returns
    -------
    W_op : pylops operator
        The projection matrix
    W : astra optomo operator
        The projection matrix

    '''
    
    # work in x, y, z coordinates for easy rotations
    # vectors in units of voxel size
    # X: optical axis, Y: detector axis (same as flexraytools)
    src_pos_init = np.array([-src_obj_dist/vox_size, 0, 0])
    det_pos_init = np.array([obj_det_dist/vox_size, 0, 0])
    col_pos_init = np.array([0, pix_size/vox_size, 0])
    
    src_pos_x = np.zeros((n_angs*n_subsample,1)) 
    src_pos_y = np.zeros((n_angs*n_subsample,1)) 
    det_pos_x = np.zeros((n_angs*n_subsample,1)) 
    det_pos_y = np.zeros((n_angs*n_subsample,1)) 
    col_pos_x = np.zeros((n_angs*n_subsample,1)) 
    col_pos_y = np.zeros((n_angs*n_subsample,1)) 
    
    ang_step = ang_range/n_angs
    proj_index = 0
    for ii in range(n_angs):
        for jj in range(n_subsample):
            # calculate the rotation angle for projection ii
            ang = ii*ang_step + jj*blur_ang_size
        
            # rotation the vectors describing the geometry (rotate around z-direction)
            rotate_setup = rot.from_rotvec(np.array([0,0,1])*ang)
        
            src_pos_rot = rotate_setup.apply(src_pos_init)
            det_pos_rot = rotate_setup.apply(det_pos_init)
            col_pos_rot = rotate_setup.apply(col_pos_init)
            
            # shift vector rotates with the setup (stays parallel with detector)
            shift = proj_index*blur_shift_size
            shift_vec = np.array([0, shift/vox_size, 0])
            shift_rot = rotate_setup.apply(shift_vec)
    
            # update the geometry
            src_pos_x[proj_index] = src_pos_rot[0] + shift_rot[0]
            src_pos_y[proj_index] = src_pos_rot[1] + shift_rot[1]

            det_pos_x[proj_index] = det_pos_rot[0] + shift_rot[0]
            det_pos_y[proj_index] = det_pos_rot[1] + shift_rot[1]

            # don't shift col pos, because this one is relative to the shifted detector position
            col_pos_x[proj_index] = col_pos_rot[0] 
            col_pos_y[proj_index] = col_pos_rot[1]
            
            proj_index = proj_index + 1

    vectors = np.concatenate((src_pos_x, src_pos_y, det_pos_x, det_pos_y, col_pos_x, col_pos_y), axis=1)

    rec_vol = astra.create_vol_geom(n_vox, n_vox)
    proj_geom_vec = astra.create_proj_geom('fanflat_vec', n_pix, vectors)
    proj_id = astra.create_projector('cuda', proj_geom_vec, rec_vol)
    W = astra.OpTomo(proj_id)
    
    return W


def get_W_combined(n_vox, vox_size, n_pix, pix_size, n_angs, ang_range, src_obj_dist, obj_det_dist, shift_speed, blur_shifts=0, blur_angs=0, blur_corr=False):
    '''
    Generate a projection matrix of a rotation where the object is shifted 
    parallel to the detector by some distance each projection (compounding!).
    Also additional roation blur is added.

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
    shift_speed : float
        The size of the object shift each angle in mm.
    blur_shifts : 1D-array
        The offset of the object shift in mm during blurring. 
        Used for ARTIC to generate different projection matrices with a
        different position for each subsample step. (if blur_shifts is just a number)
        Used for BBATIC to get large projection matrix with the subsampled
        steps, if blur_shifts is a vector.
    blur_angs : 1D-array
        The angle offset of the object shift in rad during blurring. 
        Used for ARTIC to generate different projection matrices with a
        different angle for each subsample step. (if blur_angs is just a number)
        Used for BBATIC to get large projection matrix with the subsampled
        steps, if blur_angs is a vector.
    blur_corr : boolean
        Blur correction, default False. Only used for reconstructiong the blurred dataset without ARTIC or RACE.
        Should be half the blurring interval, to align the rotation effect in the reconstruction. 

    Returns
    -------
    W_op : pylops operator
        The projection matrix
    '''
    
    n_subsample = np.size(blur_shifts)
    
    # work in x, y, z coordinates for easy rotations
    # vectors in units of voxel size
    # X: optial axis, Y: detector axis (same as flexraytools)
    src_pos_init = np.array([-src_obj_dist/vox_size, 0, 0])
    det_pos_init = np.array([obj_det_dist/vox_size, 0, 0])
    col_pos_init = np.array([0, pix_size/vox_size, 0])
    
    src_pos_x = np.zeros((n_angs*n_subsample,1)) 
    src_pos_y = np.zeros((n_angs*n_subsample,1)) 
    det_pos_x = np.zeros((n_angs*n_subsample,1)) 
    det_pos_y = np.zeros((n_angs*n_subsample,1)) 
    col_pos_x = np.zeros((n_angs*n_subsample,1)) 
    col_pos_y = np.zeros((n_angs*n_subsample,1)) 
    
    ang_step = ang_range/n_angs
    proj_index = 0
    for ii in range(n_angs):
        for jj in range(n_subsample):
            # calculate the rotation angle for projection ii
            
            # n_subsample = 1 for normal or ARTIC reconstruction
            # not 1 for RACE
            
            if n_subsample == 1:
                ang = ii*ang_step + blur_angs
                blur_shift_vec = np.array([0, blur_shifts/vox_size, 0])
            else:
                ang = ii*ang_step + blur_angs[jj]
                blur_shift_vec = np.array([0, blur_shifts[jj]/vox_size, 0])
                
            if blur_corr ==True:
                # correct the angle/position for the blurred reconstruction (do not use for ARTIC/RACE)
                ang = ang + (ang_range/n_angs)/2
                blur_shift_vec = blur_shift_vec + np.array([0,(shift_speed/vox_size)/2,0])
            
            # rotation the vectors describing the geometry (rotate around z-direction)
            rotate_setup = rot.from_rotvec(np.array([0,0,1])*ang)
        
            src_pos_rot = rotate_setup.apply(src_pos_init)
            det_pos_rot = rotate_setup.apply(det_pos_init)
            col_pos_rot = rotate_setup.apply(col_pos_init)
        
            # To bring global shift of object into consideration ....
            obj_pos = ii* shift_speed
            obj_pos_vec = np.array([0, obj_pos/vox_size, 0])
            obj_pos_rot = rotate_setup.apply(obj_pos_vec)
        
            # Add small shift due to blurring
            blur_shift_rot = rotate_setup.apply(blur_shift_vec)
        
            # update the geometry
            src_pos_x[proj_index] = src_pos_rot[0] + obj_pos_rot[0] + blur_shift_rot[0]
            src_pos_y[proj_index] = src_pos_rot[1] + obj_pos_rot[1] + blur_shift_rot[1]

            det_pos_x[proj_index] = det_pos_rot[0] + obj_pos_rot[0] + blur_shift_rot[0]
            det_pos_y[proj_index] = det_pos_rot[1] + obj_pos_rot[1] + blur_shift_rot[1]

            # don't shift col pos, because this one is relative to the shifted detector position
            col_pos_x[proj_index] = col_pos_rot[0] 
            col_pos_y[proj_index] = col_pos_rot[1]
            
            proj_index = proj_index + 1

    vectors = np.concatenate((src_pos_x, src_pos_y, det_pos_x, det_pos_y, col_pos_x, col_pos_y), axis=1)

    rec_vol = astra.create_vol_geom(n_vox, n_vox)
    proj_geom_vec = astra.create_proj_geom('fanflat_vec', n_pix, vectors)
    proj_id = astra.create_projector('cuda', proj_geom_vec, rec_vol)
    W = astra.OpTomo(proj_id)
    
    # make an operator of W with pylops
    W_op = pylops.LinearOperator(W)
    
    return W_op


def get_W_artic_combined(n_vox, vox_size, n_pix, pix_size, n_angs, ang_range, src_obj_dist, obj_det_dist, shift_speed, shift_steps, ang_steps):
    '''
    Combine multiple projection matrices with shifted and rotated object into
    one average projection matrix.

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
    shift_speed : float
         The size of the object shift each angle in mm.
    shift_steps : 1D-array
        The shifts of the setup or object parallel to the detector in mm.
    ang_steps : 1D-array
        The angle steps of the setup or object parallel to the detector in mm.

    Returns
    -------
    W_ARTIC_combined : pylops operator
        Average projection matrix to use for ARTIC-like reconstruction

    '''
    n_subsample = np.size(shift_steps)
    
    for ii in range(n_subsample):
        if ii==0:
            W_artic_combined = get_W_combined(n_vox, vox_size, n_pix, pix_size, n_angs, ang_range, src_obj_dist, obj_det_dist, shift_speed, blur_shifts=shift_steps[ii], blur_angs=ang_steps[ii])
        else:
            W_temp = get_W_combined(n_vox, vox_size, n_pix, pix_size, n_angs, ang_range, src_obj_dist, obj_det_dist, shift_speed, blur_shifts=shift_steps[ii], blur_angs=ang_steps[ii])
            W_artic_combined = W_artic_combined + W_temp
    
    W_artic_combined = (1/n_subsample) * W_artic_combined
    
    return W_artic_combined


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
    
    # divide by 2 to counter weird astra bug of multiplying the intensity of the reconstructed image
    # by 2, because the resolution of the reconstruction grid is lower (1400->700)
    img = np.reshape(current_solution/2, np.shape(phantom))
    
    if iteration == 0:
        nmse = []
        nmse.append(MSE(phantom, img)/np.var(img))    # divide by var(img) to get normalised MSE
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
    
    # if photon count too low, noise can give zero valued pixel, which is a problem for the logarithm
    # not yet solved
    
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

n_angs = 40    # number of projections
# warning, if number of angles is too high, the object might shift out FOV due to translation defined as x amount per angle
n_iters = 500     # number of BB iters

noise = False      # to add noise or not
I_0 =  10**4       # photon count for Poisson

obj_det_dist = 250
src_obj_dist = 500         # all distance in mm
M = (src_obj_dist + obj_det_dist)/src_obj_dist

n_pix = 1250      # number of detector pixels
pix_size = 0.15

phantom_name = 'phantom2'

global phantom
phantom = (np.squeeze(iio.imread('Phantoms/'+phantom_name+'.png')[:,:,0])/255) *0.01

# upscale phantom to double resolution before simulation
phantom_sim = upscale(phantom,2)

# use for blurring
n_blur_steps = 1000    # 1000 projections between the true projections to simulate blurring 

n_vox_sim = 1400      # number of row voxels of the phantom (square image)
n_vox_rec = 700      # number of row and column voxel in the reconstruction

vox_size_sim = (pix_size/M)/2
vox_size_rec = (pix_size/M)

shift_speed = 4*vox_size_sim

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

# size of the shift for 1 blur step
blur_shift_size = shift_speed/n_blur_steps

# size of the angle rotation for 1 blur step (assumes 180deg rotation)
blur_ang_size = (ang_range/n_angs)/n_blur_steps

W_sim = get_W_sim(n_vox_sim, vox_size_sim, n_pix, pix_size, n_angs, ang_range, src_obj_dist, obj_det_dist, blur_shift_size, blur_ang_size, n_blur_steps)

# do a forward projection to get the full projection data
projs_sim = W_sim @ phantom_sim.ravel()
sino_sim = projs_sim.reshape(n_angs*n_blur_steps, -1)

print('Simulation done')


##############################################################################
# RECONSTRUCT BLURRED
##############################################################################

print('Start blurred reconstruction')

# blur projections by averaging intensities
projs_blur = -logsumexp(-projs_sim.reshape(n_angs, n_blur_steps, -1), b=1/n_blur_steps, axis=1).ravel()
# due to some numerical instability 1e-16 negative numbers can appear in the background, put these to zero
projs_blur[projs_blur<0]=0

sino_blur = projs_blur.reshape(n_angs, -1)

if noise == True:
    # add Poisson noise
    sino_blur = poisson(sino_blur, I_0)


W = get_W_combined(n_vox_rec, vox_size_rec, n_pix, pix_size, n_angs, ang_range, src_obj_dist, obj_det_dist, shift_speed, blur_corr=True)
rec_blur = solve_BB(W, sino_blur, n_iters, callback=callback_func)
nmse_blur = nmse.copy()

print('Blurred reconstruction done')

##############################################################################
# RECONSTRUCT ARTIC
##############################################################################
# object shifts between two projection angles
blur_shifts_rec = np.linspace(0, shift_speed, n_subsample_rec, endpoint=False)

# object rotations between two projection angles
blur_angs_rec = np.linspace(0, ang_range/n_angs, n_subsample_rec, endpoint=False)

print('Start ARTIC reconstruction')

W_artic = get_W_artic_combined(n_vox_rec, vox_size_rec, n_pix, pix_size, n_angs, ang_range, src_obj_dist, obj_det_dist, shift_speed, blur_shifts_rec, blur_angs_rec)

rec_artic = solve_BB(W_artic, sino_blur, n_iters, callback=callback_func)

nmse_artic = nmse.copy()

print('ARTIC reconstruction done')

##############################################################################
# RECONSTRUCT RACE
##############################################################################

print('Start RACE reconstruction')

# make a sub-sampled projection matrix (easy because it is just the angle)
W_corr = get_W_combined(n_vox_rec, vox_size_rec, n_pix, pix_size, n_angs, ang_range, src_obj_dist, obj_det_dist, shift_speed, blur_shifts_rec, blur_angs_rec)


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

max_val = max(rec_blur.max(), rec_race.max(), rec_artic.max())

plt.figure();plt.imshow(rec_blur,cmap='grey', vmin=0,vmax=max_val);plt.axis('off');
plt.show()

plt.figure();plt.imshow(rec_artic,cmap='grey', vmin=0,vmax=max_val);plt.axis('off');
plt.show()

plt.figure();plt.imshow(rec_race,cmap='grey', vmin=0,vmax=max_val);plt.axis('off');
plt.show()

