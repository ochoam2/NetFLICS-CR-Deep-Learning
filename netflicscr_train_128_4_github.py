"""
Created by Ruoyang Yao/ Adapted by Marien Ochoa
Train CNN network for lifetime image reconstruction with high CR
"""

from matplotlib import pyplot as plt
import numpy as np
import math
from os import path
from os import mkdir
from os import getcwd
from class_generator import DataGenerator
from res_blocks import resblock_1D, resblock_1D_BN
from res_blocks import resblock_2D, resblock_2D_BN
import h5py

from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.models import Model
from keras.layers import Input, Reshape, Permute, add, SeparableConv2D
from keras.layers import Conv1D, Conv2D, BatchNormalization
from keras.layers.core import Activation
from keras.optimizers import RMSprop, SGD
from keras.utils import plot_model 
# from keras.models import load_model


# Read and Prepare training data file ======================================================
folder_data = getcwd()
log_dir = path.join(folder_data,'trained_netflicscr_pat200')
mkdir(path.join(folder_data,log_dir))
fn_model = path.join(log_dir, 'trained_flim-net.h5')
fn_model_final = path.join(log_dir, 'trained_flim-net_final.h5')

# sorted pattern list----------------------------------------------------------
fn_data = path.join(folder_data, 'pattern_index_HR_128.mat')
f = h5py.File(fn_data, 'r')
#pattern_whole = f['pat_idx'].value
#pattern_whole = pattern_whole.astype(int)-1

pat_num = 200;
pattern_list = list()
for i in range(pat_num):
#    pattern_list.insert(i,pattern_whole[0][i])
    pattern_list.insert(i,i)
f.close();

# define data generator ==================================================================

train_size = 3900; #32000 in NetFLICS-CR paper
valid_size = 1200; #8000 in NetFLICS-CR paper
batch_size = 10; #20 in NetFLICS-CR paper

params = {'pat_list': pattern_list,
          'num_gate': 256,
          'img_size': 128,
          'batch_size': batch_size,
          'shuffle': True}

data_idx = np.random.permutation(train_size+valid_size)+1;
train_idx = data_idx[0:train_size];
validate_idx = data_idx[train_size:train_size+valid_size];
#train_idx = np.arange(train_size)+1;
#validate_idx = np.arange(train_size,train_size+valid_size)+1;

train_generator = DataGenerator(**params).generate(train_idx);
validate_generator = DataGenerator(**params).generate(validate_idx);

# %% Reconstruction network =============================================================

img_rows, img_cols = 128, 128
cs_data = Input(shape=(256,pat_num,))
#xcs = BatchNormalization(axis=1)(cs_data)

xcs = Conv1D(16384, 1, padding='same', activation=None)(cs_data)
xcs = BatchNormalization()(xcs)
xcs = Activation('relu')(xcs)

# End of Segment 1------------------------------------------------------------

xcs = Permute((2,1))(xcs)

# reconstruct intensity image-------------------------------------------------

xint = Reshape((img_rows,img_cols,256))(xcs)

xint = resblock_2D(256, 3, xint)

xint = Conv2D(64, 1, padding='same', activation=None)(xint) 
xint = Activation('relu')(xint)

xint = Conv2D(32, 1, padding='same', activation=None)(xint)
xint = Activation('relu')(xint)

xint = Conv2D(1, 3, padding='same', activation=None)(xint)
img_intensity = Activation('relu')(xint)


# reconstruct lifetime image--------------------------------------------------

xfit = Conv1D(512, 1, padding='same', activation=None)(xcs)
xfit = BatchNormalization()(xfit)
xfit = Activation('relu')(xfit)
xfit = Reshape((img_rows,img_cols,512))(xfit)

xfit = SeparableConv2D(256,1,padding='same', activation=None)(xfit) 
xfit = Activation('relu')(xfit)

xfit = resblock_2D(256, 1, xfit)

xfit = SeparableConv2D(128, 5, padding='same', activation=None)(xfit)
xfit = Activation('relu')(xfit)

xfit = Conv2D(64, 1, padding='same', activation=None)(xfit)
xfit = Activation('relu')(xfit)

xfit = Conv2D(1, 3, padding='same', activation=None)(xfit)
xfit = Activation('relu')(xfit)

xfit = SeparableConv2D(64, 5, padding='same', activation=None)(xfit)
xfit = Activation('relu')(xfit)

xfit = Conv2D(32, 1, padding='same', activation=None)(xfit)
xfit = Activation('relu')(xfit)

xfit = Conv2D(1, 3, padding='same', activation=None)(xfit)
img_lifetime = Activation('relu')(xfit)

# construct new model---------------------------------------------------------

model = Model(inputs=cs_data, outputs=[img_intensity,img_lifetime])
rmsprop = RMSprop(lr=1e-5)
sgd = SGD(lr=1e-3, momentum=0.9, nesterov=True)

def step_decay(epoch):
   initial_lrate = 1e-5 
   drop = 0.5
   epochs_drop = 10.0
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   return lrate

model.compile(loss='mse',
              optimizer=rmsprop,
              loss_weights=[1,1e5],
              metrics=['mae','mse'])

# print network structure-----------------------------------------------------

layers = model.layers

print('=' * 70)
for idx, layer in enumerate(layers):
    print('Layer {}: type={}'.format(idx, type(layer)))
    print(' '*9 + 'output->{}'.format(layer.output_shape))
    if (type(layer) is Conv1D or
        type(layer) is Conv2D):
        weights_shape = layer.get_weights()[0].shape
        print(' '*9 + 'weights->{}'.format(weights_shape))
print('=' * 70)

# %% Training- Other Parameters

earlyStopping = EarlyStopping(monitor='val_activation_16_mean_absolute_error',
                              patience = 15,
                              verbose = 0,
                              mode='auto')

modelCheckPoint = ModelCheckpoint(fn_model,
                                  monitor='val_activation_16_mean_absolute_error',
                                  save_best_only=True, 
                                  verbose=0)

learningrate = LearningRateScheduler(step_decay)

history = model.fit_generator(train_generator,
                              steps_per_epoch = train_size/batch_size,
                              epochs = 100,
                              validation_data = validate_generator,
                              validation_steps = valid_size/batch_size,
                              shuffle = True,
                              callbacks=[modelCheckPoint, earlyStopping, learningrate])

model.save(fn_model_final)
plot_model(model, show_shapes = True, to_file='model_tsne.png')


# %% save training history and plot =====================================================

train_mae_int = history.history['activation_6_mean_absolute_error']
train_mae_lt = history.history['activation_16_mean_absolute_error']
val_mae_int = history.history['val_activation_6_mean_absolute_error']
val_mae_lt = history.history['val_activation_16_mean_absolute_error']

train_mse_int = history.history['activation_6_mean_squared_error']
train_mse_lt = history.history['activation_16_mean_squared_error']
val_mse_int = history.history['val_activation_6_mean_squared_error']
val_mse_lt = history.history['val_activation_16_mean_squared_error']

# save training history
np.save(path.join(log_dir,'train_mae_intensity'),np.asarray(train_mae_int))
np.save(path.join(log_dir,'train_mae_lifetime'),np.asarray(train_mae_lt))
np.save(path.join(log_dir,'validate_mae_intensity'),np.asarray(val_mae_int))
np.save(path.join(log_dir,'validate_mae_lifetime'),np.asarray(val_mae_lt))

np.save(path.join(log_dir,'train_mse_intensity'),np.asarray(train_mse_int))
np.save(path.join(log_dir,'train_mse_lifetime'),np.asarray(train_mse_lt))
np.save(path.join(log_dir,'validate_mse_intensity'),np.asarray(val_mse_int))
np.save(path.join(log_dir,'validate_mse_lifetime'),np.asarray(val_mse_lt))

eps = range(1, len(train_mae_int)+1)

plt.figure()
plt.title('Training and validation MAE of intensity image')
plt.xlabel('epoch')
ta_int, = plt.plot(eps, train_mae_int)
va_int, = plt.plot(eps, val_mae_int)
plt.legend([ta_int, va_int], ['Train int', 'Val int'])
plt.savefig(path.join(log_dir, 'mae_intensity.png'), dpi=600) 

plt.figure()
plt.title('Training and validation MSE of intensity image')
plt.xlabel('epoch')
ta_int, = plt.plot(eps, train_mse_int)
va_int, = plt.plot(eps, val_mse_int)
plt.legend([ta_int, va_int], ['Train int', 'Val int'])
plt.savefig(path.join(log_dir, 'mse_intensity.png'), dpi=600)

plt.figure()
plt.title('Training and validation MAE of lifetime image')
plt.xlabel('epoch')
ta_lt, = plt.plot(eps, train_mae_lt)
va_lt, = plt.plot(eps, val_mae_lt)
plt.legend([ta_lt, va_lt], ['Train_lifetime', 'Val_lifetime'])
plt.savefig(path.join(log_dir, 'mae_lifetime.png'), dpi=600)

plt.figure()
plt.title('Training and validation MSE of lifetime image')
plt.xlabel('epoch')
ta_lt, = plt.plot(eps, train_mse_lt)
va_lt, = plt.plot(eps, val_mse_lt)
plt.legend([ta_lt, va_lt], ['Train_lifetime', 'Val_lifetime'])
plt.savefig(path.join(log_dir, 'mse_lifetime.png'), dpi=600)

