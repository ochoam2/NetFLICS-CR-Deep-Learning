# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:46:18 2017

@author: mry09

Data Generator for NetFLICS and NetFLICS-CR
"""

import numpy as np
import h5py
from os import path

folder_data = 'example_train_data'

class DataGenerator(object):
    'Generates data for Keras'
    
    def __init__(self, pat_list, num_gate = 256, img_size = 128, batch_size = 10, shuffle = True):
        
        self.pat_list = pat_list
        self.num_gate = num_gate
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        
    def generate(self, list_IDs):
        
        while 1:
            # generate order
            indexes = self.__get_exploration_order(list_IDs)
            # generate batch
            imax = int(len(indexes)/self.batch_size)
            for i in range(imax):
                # find list of IDs
                list_IDs_tmp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]
                # generate data
                X, y1, y2 = self.__data_generation(list_IDs_tmp)
                # output
                yield X, [y1,y2]
                # yield X, y1
            
    
    def __get_exploration_order(self, list_IDs):
        
        'Generate order of exploration'
        indexes = np.arange(len(list_IDs))
        if self.shuffle == True:
            np.random.shuffle(indexes)
        return indexes
    
    def __data_generation(self, list_IDs_temp):
        
        # initialization for training data, intensity image, lifetime image
        num_pat = np.size(self.pat_list)
        X = np.empty((self.batch_size, self.num_gate, num_pat))
        y1 = np.empty((self.batch_size, self.img_size, self.img_size, 1))
        y2 = np.empty((self.batch_size, self.img_size, self.img_size, 1))
        
        # generate data
        for i, ID in enumerate(list_IDs_temp):
            sample = 'sample_'+str(ID)+'.mat'
            fn_data = path.join(folder_data, sample)
            f = h5py.File(fn_data, 'r')
            cs_data = f['cs_data'].value
            X[i,:,:] = cs_data[:,self.pat_list]
            y1[i,:,:,0] = np.transpose(f['intensity_image'].value)
            y2[i,:,:,0] = np.transpose(f['lifetime_image'].value)
            f.close()
        
        return X, y1, y2