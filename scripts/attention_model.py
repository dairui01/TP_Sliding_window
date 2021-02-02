import os, sys, random

os.environ['KERAS_BACKEND'] = 'tensorflow'
# os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
from keras.layers import Dense, Flatten, Dropout, Reshape, Input
from keras import regularizers
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical
from keras.optimizers import SGD
from i3d_inception import Inception_Inflated3d, conv3d_bn
from keras.utils import Sequence
from keras.layers import LSTM, Dense, Activation
from keras.layers import TimeDistributed, GaussianNoise, GaussianDropout, Dropout
from keras.layers import Activation, concatenate, Dense, Flatten, Dropout, Reshape, Merge, Input, Add, RepeatVector, Permute
from keras.layers import AveragePooling3D
from multiprocessing import cpu_count
import numpy as np
import glob
from skimage.io import imread
import cv2
import keras
from keras.models import load_model
from keras import backend as K


def inflate_dense_spatial(x):
    a = RepeatVector(8*1024)(x)
    a = Permute((2,1), input_shape=(49,8*1024))(a)
    a = Reshape((8,7,7,1024))(a)
    return a

def inflate_dense_temporal(x):
    a = RepeatVector(49*1024)(x)
    a = Permute((2,1), input_shape=(8,49*1024))(a)
    a = Reshape((8,7,7,1024))(a)
    return a

def attention_reg(weight_mat):
    return 0.00001*K.square((1-K.sum(weight_mat)))
    #return 0.001*K.sum(K.square(weight_mat))

class i3d_modified:
    def __init__(self, weights='rgb_imagenet_and_kinetics'):
        self.model = Inception_Inflated3d(include_top=True, weights=weights)

    def i3d_flattened(self, num_classes=60):
        i3d = Model(inputs=self.model.input, outputs=self.model.get_layer(index=-4).output)
        x = conv3d_bn(i3d.output, num_classes, 1, 1, 1, padding='same', use_bias=True, use_activation_fn=False,
                      use_bn=False, name='Conv3d_6a_1x1')
        num_frames_remaining = int(x.shape[1])
        x = Flatten()(x)
        predictions = Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l2(0.01),
                            activity_regularizer=regularizers.l1(0.01))(x)
        new_model = Model(inputs=i3d.input, outputs=predictions)

        # for layer in i3d.layers:
        #    layer.trainable = False

        return new_model


def build_model_two_pathways(n_neuron, timesteps, data_dim, num_classes, n_dropout, dataset):
    i3d = i3d_modified(weights = 'rgb_imagenet_and_kinetics')
    model_branch = i3d.i3d_flattened(num_classes = num_classes)
    if dataset=='NTU':
        model_branch.load_weights('../models/ntu-cv_pre-trained_rgb_model.hdf5')
    elif dataset=='Smarthomes':
        model_branch.load_weights('../models/smarthomes-cs_pre-trained_rgb_model.hdf5')
    optim = SGD(lr=0.01, momentum=0.9)
    model_branch.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

    print('Build model...')
    model_inputs=[]
    if dataset=='NTU':
        model_lstm = load_model('../models/ntu-cv_pre-trained_skeleton_model.hdf5')
    elif dataset=='Smarthomes':
        model_lstm = load_model('../models/smarthomes-cs_pre-trained_skeleton_model.hdf5')
    model_lstm.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.005, clipnorm=1), metrics=['accuracy'])
    for layer in model_lstm.layers:
        layer.trainable = False
    #model_lstm.load_weights('../models/ntu_pre-trained_skeleton_model.hdf5')
    model_lstm.pop()
    z1 = Dense(256, activation='tanh', name='z1_layer', trainable=True)(model_lstm.get_layer('dropout_1').output)
    z2 = Dense(256, activation='tanh', name='z2_layer', trainable=True)(model_lstm.get_layer('dropout_1').output)
    fc_main1 = Dense(49, kernel_initializer='zeros', bias_initializer='zeros', activation='sigmoid', trainable=True, name='dense_1')(z1)
    atten_mask_spatial = keras.layers.core.Lambda(inflate_dense_spatial, output_shape=(8,7,7,1024))(fc_main1)
    fc_main_2 = Dense(8, kernel_initializer='zeros', bias_initializer='zeros',
                      activation='softmax', trainable=True, name='dense_2')(z2)
    atten_mask_temporal = keras.layers.core.Lambda(inflate_dense_temporal, output_shape=(8, 7, 7, 1024))(fc_main_2)

    model_inputs.append(model_lstm.input)
    model_inputs.append(model_branch.input)

    for l in model_branch.layers:
            l.trainable=True

    multiplied_features1 = keras.layers.Multiply()([atten_mask_spatial, model_branch.get_layer('Mixed_5c').output])

    x = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid', name='global_avg_pool1'+'second')(multiplied_features1)
    x = Dropout(n_dropout)(x)

    multiplied_features2 = keras.layers.Multiply()([atten_mask_temporal, model_branch.get_layer('Mixed_5c').output])

    y = AveragePooling3D((2, 7, 7), strides=(1, 1, 1), padding='valid', name='global_avg_pool2' + 'second')(multiplied_features2)
    y = Dropout(n_dropout)(y)

    agg_features = keras.layers.Concatenate()([x,y])

    agg_features = conv3d_bn(agg_features, num_classes, 1, 1, 1, padding='same', use_bias=True, use_activation_fn=False, use_bn=False, name='Conv3d_6a_1x1'+'second')

    agg_features = Flatten(name='flatten'+'second')(agg_features)
    predictions = Dense(num_classes, activation='softmax', name='softmax'+'second')(agg_features)
    model = Model(inputs=model_inputs, outputs=predictions, name = 'ST_attention')

    for l_m, l_lh in zip(model.layers[-8: -7], model_branch.layers[-5: -4]):
        l_m.set_weights(l_lh.get_weights())
        l_m.trainable = True

    for l_m, l_lh in zip(model.layers[-7: -6], model_branch.layers[-5: -4]):
        l_m.set_weights(l_lh.get_weights())
        l_m.trainable = True

    return model