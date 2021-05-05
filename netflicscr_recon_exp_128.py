# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:50:04 2018
@author: mry09 and modified by MarienOchoa
"""
from matplotlib import pyplot as plt
import numpy as np
from os import path
from os import mkdir
import scipy.io as sio
import time
from keras.models import load_model
from scipy.ndimage import gaussian_filter1d
start = time.time()

# %% Read data file
fn_data = 'H:/Computers Marien SYNC/Office Computer/Marien/Meetings/Paper Submissions/NetFLICS CR Paper/GitHub repository/ExRPI128/Meas_TD_EM_RanHad_rpi.mat'
log_dir = 'H:/Computers Marien SYNC/Office Computer/Marien/Meetings/Paper Submissions/NetFLICS CR Paper/GitHub repository/trained_netflicscr_pat200/'
folder = 'RPI128recons/'
mkdir(path.join(log_dir,folder))
log_dir2 = path.join(log_dir+folder)
sample = '1'
f = sio.loadmat(fn_data)
data0 = np.array(f['Meas_TD_EM'])
data0 = np.asfarray(data0[0:16,:,0:200])
data0 = gaussian_filter1d(data0,3.5,0)
data0 = np.multiply(data0,16) # multiplied by factor to match photon count range of training set.

# load model------------------------------------------------------------------
fn_model = path.join(log_dir, 'trained_flim-net.h5')
model = load_model(fn_model)

# model predict-------------------------------------------------------------
imgs_recon_int, imgs_recon_lt = model.predict(data0)
np.save(path.join(log_dir2,'recon_intensity_'+sample),imgs_recon_int)
np.save(path.join(log_dir2,'recon_lifetime_'+sample),imgs_recon_lt)

# save as .mat-----------------------------------------------------------------
sio.savemat(path.join(log_dir2,'recon_intensity_'+sample), {'imgs_recon_int': imgs_recon_int})
sio.savemat(path.join(log_dir2,'recon_lifetime_'+sample), {'imgs_recon_lt': imgs_recon_lt})

end = time.time()
print(end - start)

# %% Display reconstruction
re = 128 
img_montage_int = np.ones((re, re*16))
img_montage_lt = np.ones((re, re*16))

for idx in range(0, 16):
    img_montage_lt[:re, idx*re:(idx+1)*re] = imgs_recon_lt[idx,:,:,0]
    img_montage_int[:re, idx*re:(idx+1)*re] = imgs_recon_int[idx,:,:,0]

plt.figure()
plt.imshow(np.flipud(img_montage_int))
plt.imsave(path.join(log_dir2,'recon_intensity_'+sample+'.jpg'),np.flipud(img_montage_int))
    
plt.figure()
plt.imshow(np.flipud(img_montage_lt))
plt.imsave(path.join(log_dir2,'recon_lifetime_'+sample+'.jpg'),np.flipud(img_montage_lt))