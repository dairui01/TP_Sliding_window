import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

from keras.utils import Sequence, to_categorical

import numpy as np
from random import sample, randint, shuffle
import glob
import cv2
import time

class DataGenerator(Sequence):
    def __init__(self, paths, mode, batch_size = 4):
        self.batch_size = batch_size
        self.path_skeleton = paths['skeleton']
        self.path_cnn = paths['cnn']
        self.files = [i.strip() for i in open(paths['split_path'] + mode + '.txt').readlines()]
        self.stack_size = 64
        self.num_classes = 32
        self.stride = 2
        self.step = 30
        self.dim = 39
        self.on_epoch_end()

    def _name_to_int(self,name):
        integer=0
        if name=="Cook.Cleandishes":
            integer=1
        elif name=="Cook.Cleanup":
            integer=2
        elif name=="Cook.Cut":
            integer=3
        elif name=="Cook.Stir":
            integer=4
        elif name=="Cook.Usestove":
            integer=5
        elif name=="Cutbread":
            integer=6
        elif name=="Drink.Frombottle":
            integer=7
        elif name=="Drink.Fromcan":
            integer=8
        elif name=="Drink.Fromcup":
            integer=9
        elif name=="Drink.Fromglass":
            integer=10
        elif name=="Eat.Attable":
            integer=11
        elif name=="Eat.Snack":
            integer=12
        elif name=="Enter":
            integer=13
        elif name=="Getup":
            integer=14
        elif name=="Laydown":
            integer=15
        elif name=="Leave":
            integer=16
        elif name=="Makecoffee.Pourgrains":
            integer=17
        elif name=="Makecoffee.Pourwater":
            integer=18
        elif name=="Maketea.Boilwater":
            integer=19
        elif name=="Maketea.Insertteabag":
            integer=20
        elif name=="Pour.Frombottle":
            integer=21
        elif name=="Pour.Fromcan":
            integer=22
        elif name=="Pour.Fromcup":
            integer=23
        elif name=="Pour.Fromkettle":
            integer=24
        elif name=="Readbook":
            integer=25
        elif name=="Sitdown":
            integer=26
        elif name=="Takepills":
            integer=27
        elif name=="Uselaptop":
            integer=28
        elif name=="Usetablet":
            integer=29
        elif name=="Usetelephone":
            integer=30
        elif name=="Walk":
            integer=31
        elif name=="WatchTV":
            integer=32
        return integer

    def __len__(self):
        return int(len(self.files) / self.batch_size)
    
    def __getitem__(self, idx):
        batch = self.files[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch = [os.path.splitext(i)[0] for i in batch]
        #print batch
        x_data_cnn = self._get_data_cnn(batch)
        x_data_skeleton = self._get_data_skeleton(batch)

        y_data = np.array([self._name_to_int(i.split('_')[0]) for i in batch]) - 1
        y_data = to_categorical(y_data, num_classes = self.num_classes)

        return [x_data_skeleton, x_data_cnn], y_data

    def _get_data_cnn(self, batch):

        x_train = [self._get_video(i, self.path_cnn) for i in batch]
        x_train = np.array(x_train, np.float32)
        x_train /= 127.5
        x_train -= 1

        return x_train

    def _get_video(self, vid_name, dataset_path):
        images = glob.glob(dataset_path + vid_name + "/*")
        images.sort()
        files = []
        if len(images) > (self.stack_size * self.stride):
            start = randint(0, len(images) - self.stack_size * self.stride)
            files.extend([images[i] for i in range(start, (start + self.stack_size * self.stride), self.stride)])
        elif len(images) < self.stack_size:
            files.extend(images)
            while len(files) < self.stack_size:
                files.extend(images)
            files = files[:self.stack_size]
        else:
            start = randint(0, len(images) - self.stack_size)
            files.extend([images[i] for i in range(start, (start + self.stack_size))])
            
        files.sort()
        
        arr = []
        for i in files:
            if os.path.isfile(i):
                arr.append(cv2.resize(cv2.imread(i), (224, 224)))
            else:
                arr.append(arr[-1])

        return arr

    def _get_data_skeleton(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.step, self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        #f = 'Cook_p15_r03_v16_c03'
        for i, ID in enumerate(list_IDs_temp):
            # Store sample

            unpadded_file = np.load(self.path_skeleton + ID + '.npz')['arr_0']

            if len(unpadded_file) > 0:
                f = ID
            if len(unpadded_file) == 0:
                unpadded_file = np.load(self.path_skeleton + f + '.npz')['arr_0']
                list_IDs_temp[i] = f
            origin = unpadded_file[0, 3:6]  # Extract hip of the first frame
            [row, col] = unpadded_file.shape
            origin = np.tile(origin, (row, 13))  # making equal dimension
            unpadded_file = unpadded_file - origin  # translation
            extra_frames = (len(unpadded_file) % self.step)
            l = 0
            if len(unpadded_file) < self.step:
                extra_frames = self.step - len(unpadded_file)
                l = 1
            if extra_frames < (self.step / 2) & l == 0:
                padded_file = unpadded_file[0:len(unpadded_file) - extra_frames, :]
            else:
                [row, col] = unpadded_file.shape
                alpha = int(len(unpadded_file) / self.step) + 1
                req_pad = np.zeros(((alpha * self.step) - row, col))
                padded_file = np.vstack((unpadded_file, req_pad))
            splitted_file = np.split(padded_file, self.step)
            splitted_file = np.asarray(splitted_file)
            row, col, width = splitted_file.shape
            sampled_file = []
            for k in range(0, self.step):
                c = np.random.choice(col, 1)
                sampled_file.append(splitted_file[k, c, :])
            sampled_file = np.asarray(sampled_file)
            X[i,] = np.squeeze(sampled_file)

        return X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.files))
        shuffle(self.files)
        pass


class DataGenerator_test(Sequence):
    def __init__(self, paths, mode, batch_size=4):
        self.batch_size = batch_size
        self.path_skeleton = paths['skeleton']
        self.path_cnn = paths['cnn']
        self.files = [i.strip() for i in open(paths['split_path'] + mode + '.txt').readlines()]
        self.stack_size = 64
        self.num_classes = 32
        self.stride = 2
        self.step = 30
        self.dim = 39
        self.on_epoch_end()

    def _name_to_int(self, name):
        integer = 0
        if name == "Cook.Cleandishes":
            integer = 1
        elif name == "Cook.Cleanup":
            integer = 2
        elif name == "Cook.Cut":
            integer = 3
        elif name == "Cook.Stir":
            integer = 4
        elif name == "Cook.Usestove":
            integer = 5
        elif name == "Cutbread":
            integer = 6
        elif name == "Drink.Frombottle":
            integer = 7
        elif name == "Drink.Fromcan":
            integer = 8
        elif name == "Drink.Fromcup":
            integer = 9
        elif name == "Drink.Fromglass":
            integer = 10
        elif name == "Eat.Attable":
            integer = 11
        elif name == "Eat.Snack":
            integer = 12
        elif name == "Enter":
            integer = 13
        elif name == "Getup":
            integer = 14
        elif name == "Laydown":
            integer = 15
        elif name == "Leave":
            integer = 16
        elif name == "Makecoffee.Pourgrains":
            integer = 17
        elif name == "Makecoffee.Pourwater":
            integer = 18
        elif name == "Maketea.Boilwater":
            integer = 19
        elif name == "Maketea.Insertteabag":
            integer = 20
        elif name == "Pour.Frombottle":
            integer = 21
        elif name == "Pour.Fromcan":
            integer = 22
        elif name == "Pour.Fromcup":
            integer = 23
        elif name == "Pour.Fromkettle":
            integer = 24
        elif name == "Readbook":
            integer = 25
        elif name == "Sitdown":
            integer = 26
        elif name == "Takepills":
            integer = 27
        elif name == "Uselaptop":
            integer = 28
        elif name == "Usetablet":
            integer = 29
        elif name == "Usetelephone":
            integer = 30
        elif name == "Walk":
            integer = 31
        elif name == "WatchTV":
            integer = 32
        return integer

    def __len__(self):
        return int(len(self.files) / self.batch_size)

    def __getitem__(self, idx):
        batch = self.files[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch = [os.path.splitext(i)[0] for i in batch]
        # print batch
        x_data_cnn = self._get_data_cnn(batch)
        x_data_skeleton = self._get_data_skeleton(batch)

        y_data = np.array([self._name_to_int(i.split('_')[0]) for i in batch]) - 1
        y_data = to_categorical(y_data, num_classes=self.num_classes)

        return [x_data_skeleton, x_data_cnn], y_data

    def _get_data_cnn(self, batch):

        x_train = [self._get_video(i, self.path_cnn) for i in batch]
        x_train = np.array(x_train, np.float32)
        x_train /= 127.5
        x_train -= 1

        return x_train

    def _get_video(self, vid_name, dataset_path):
        images = glob.glob(dataset_path + vid_name + "/*")
        images.sort()
        files = []
        if len(images) > (self.stack_size * self.stride):
            start = randint(0, len(images) - self.stack_size * self.stride)
            files.extend([images[i] for i in range(start, (start + self.stack_size * self.stride), self.stride)])
        elif len(images) < self.stack_size:
            files.extend(images)
            while len(files) < self.stack_size:
                files.extend(images)
            files = files[:self.stack_size]
        else:
            start = randint(0, len(images) - self.stack_size)
            files.extend([images[i] for i in range(start, (start + self.stack_size))])

        files.sort()

        arr = []
        for i in files:
            if os.path.isfile(i):
                arr.append(cv2.resize(cv2.imread(i), (224, 224)))
            else:
                arr.append(arr[-1])

        return arr

    def _get_data_skeleton(self, list_IDs_temp):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.step, self.dim))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        f = 'Cook_p15_r03_v16_c03'
        for i, ID in enumerate(list_IDs_temp):
            # Store sample

            unpadded_file = np.load(self.path_skeleton + ID + '.npz')['arr_0']

            if len(unpadded_file) > 0:
                f = ID
            if len(unpadded_file) == 0:
                unpadded_file = np.load(self.path_skeleton + f + '.npz')['arr_0']
                list_IDs_temp[i] = f
            origin = unpadded_file[0, 3:6]  # Extract hip of the first frame
            [row, col] = unpadded_file.shape
            origin = np.tile(origin, (row, 13))  # making equal dimension
            unpadded_file = unpadded_file - origin  # translation
            extra_frames = (len(unpadded_file) % self.step)
            l = 0
            if len(unpadded_file) < self.step:
                extra_frames = self.step - len(unpadded_file)
                l = 1
            if extra_frames < (self.step / 2) & l == 0:
                padded_file = unpadded_file[0:len(unpadded_file) - extra_frames, :]
            else:
                [row, col] = unpadded_file.shape
                alpha = int(len(unpadded_file) / self.step) + 1
                req_pad = np.zeros(((alpha * self.step) - row, col))
                padded_file = np.vstack((unpadded_file, req_pad))
            splitted_file = np.split(padded_file, self.step)
            splitted_file = np.asarray(splitted_file)
            row, col, width = splitted_file.shape
            sampled_file = []
            for k in range(0, self.step):
                c = np.random.choice(col, 1)
                sampled_file.append(splitted_file[k, c, :])
            sampled_file = np.asarray(sampled_file)
            X[i,] = np.squeeze(sampled_file)

        return X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.files))
        #shuffle(self.files)
        pass