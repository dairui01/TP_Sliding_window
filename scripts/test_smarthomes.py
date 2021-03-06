from argparse import ArgumentParser
import sys

argv = sys.argv
# Parse arguments
parser = ArgumentParser(description='Test Attention model')

parser.add_argument('-model_name', '--model_name', help='location of model stored in model directory')
#parser.add_argument('dataset', help='Dataset Name NTU/Smarthomes')
args = parser.parse_args(argv[1:])

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
#os.environ['CUDA_VISIBLE_DEVICES'] = "2,3"
from keras.layers import Dense, Flatten, Dropout, Reshape, Input
from keras import regularizers
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical
from keras.optimizers import SGD
from i3d_inception import Inception_Inflated3d, conv3d_bn
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, CSVLogger, Callback
from keras.utils import Sequence, multi_gpu_model

import random
from multiprocessing import cpu_count
import numpy as np
import glob
from skimage.io import imread
import cv2
from attention_model import *


num_classes = 32
batch_size = 4
stack_size = 64
data_dim = 39
n_neuron = 512
n_dropout = 0.5
timesteps = 30
from Smarthome_Loader_CS import *

'''
if args.dataset=='NTU':
    num_classes = 60
    batch_size = 4
    stack_size = 64
    data_dim = 150
    n_neuron = 512
    n_dropout = 0.5
    timesteps = 20
    from NTU_Loader import *
elif args.dataset=='Smarthomes':
    num_classes = 32
    batch_size = 4
    stack_size = 64
    data_dim = 39
    n_neuron = 512
    n_dropout = 0.5
    timesteps = 30
    from Smarthome_Loader_CS import DataGenerator_test
    from Smarthome_labels import *
'''
class CustomModelCheckpoint(Callback):

    def __init__(self, model_parallel, path):

        super(CustomModelCheckpoint, self).__init__()

        self.save_model = model_parallel
        self.path = path
        self.nb_epoch = 0

    def on_epoch_end(self, epoch, logs=None):
        self.nb_epoch += 1
        self.save_model.save(self.path + str(self.nb_epoch) + '.hdf5')

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.1, patience = 5)
optim = SGD(lr = 0.01, momentum = 0.9)
model = build_model_two_pathways(n_neuron, timesteps, data_dim, num_classes, n_dropout, 'Smarthomes')
model.load_weights(args.model_name)
model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
print "model compiled"
'''
parallel_model = multi_gpu_model(model, gpus=4)
parallel_model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
'''

paths = {
        'skeleton': '../data/3d_skeleton/Smarthomes/',
        'cnn': '../data/rgb/Smarthomes/',
        'split_path': '../data/'
    }

test_generator = DataGenerator_test(paths, 'test_Smarthomes', batch_size = batch_size)

np.savetxt('../output_files/pred_results.txt', np.argmax(model.predict_generator(generator=test_generator, use_multiprocessing=True, workers=cpu_count()-2), axis=-1))
#np.savetxt('../output_files/pred_results.txt', parallel_model.predict_generator(generator=test_generator, use_multiprocessing=True, workers=cpu_count()-2))
