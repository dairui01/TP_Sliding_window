from argparse import ArgumentParser
import sys

argv = sys.argv
# Parse arguments
parser = ArgumentParser(description='Test Attention model')

parser.add_argument('-model_name', '--model_name', help='location of model stored in model directory')
parser.add_argument('dataset', help='Dataset Name NTU/Smarthomes')
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
    from Smarthome_Loader_CS_window import DataGenerator_window_test
    from Smarthome_labels import *

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
model = build_model_two_pathways(n_neuron, timesteps, data_dim, num_classes, n_dropout, args.dataset)
#model.load_weights(args.model_name)

model.load_weights('/data/stars/user/rdai/Toyota_deploy/deployment_codes1/models/smarthomes-cs_final_STA_model.hdf5')
model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])

'''
parallel_model = multi_gpu_model(model, gpus=4)
parallel_model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
model.compile(loss = 'categorical_crossentropy', optimizer = optim, metrics = ['accuracy'])
'''

paths = {
        'skeleton': '../data/3d_skeleton/'+args.dataset+'/',
        'cnn': '../data/rgb/'+args.dataset+'/',
        'split_path': '../data/'
    }

test_generator = DataGenerator_window_test(paths, 'test_'+args.dataset, batch_size = batch_size)

#np.savetxt('../output_files/pred_results.txt', np.argmax(model.predict_generator(generator=test_generator, use_multiprocessing=True, workers=cpu_count()-2), axis=-1))


#
print ('start predicting P13T18C03')
prediction_result=model.predict_generator(generator = test_generator, use_multiprocessing=True, workers=cpu_count()-2)
np.save('../output_files/pred_results_window_P13T18C03.txt',prediction_result)



import numpy as np
import pandas as pd

result=np.load('../output_files/pred_results_window_P13T18C03.txt.npy')

pred_result=[]
confident_score=[]
gt_list=[]

def _name_to_int(integer):
    if integer==0:
        label="Background"
    if integer==1:
        label="Cook.Clean_dishes"
    if integer==2:
        label="Cook.Cleanup"
    if integer==3:
        label="Cook.Cut"
    if integer==4:
        label="Cook.Stir"
    if integer==5:
        label="Cook.Usestove"
    if integer==6:
        label="Cutbread"
    if integer==7:
        label="Drink.Frombottle"
    if integer==8:
        label="Drink.Fromcan"
    if integer==9:
        label="Drink.Fromcup"
    if integer==10:
        label="Drink.Fromglass"
    if integer==11:
        label="Eat.Attable"
    if integer==12:
        label="Eat.Snack"
    if integer==13:
        label="Enter"
    if integer==14:
        label="Getup"
    if integer==15:
        label="Laydown"
    if integer==16:
        label="Leave"
    if integer==17:
        label="Makecoffee.Pourgrains"
    if integer==18:
        label="Makecoffee.Pourwater"
    if integer==19:
        label="Maketea.Boilwater"
    if integer==20:
        label="Maketea.Insertteabag"
    if integer==21:
        label="Pour.Frombottle"
    if integer==22:
        label="Pour.Fromcan"
    if integer==23:
        label="Pour.Fromcup"
    if integer==24:
        label="Pour.Fromkettle"
    if integer==25:
        label="Readbook"
    if integer==26:
        label="Sitdown"
    if integer==27:
        label="Takepills"
    if integer==28:
        label="Uselaptop"
    if integer==29:
        label="Usetablet"
    if integer==30:
        label="Usetelephone"
    if integer==31:
        label="Walk"
    if integer==32:
        label="WatchTV"
    return label

for i in range(len(result)):
    #None
    if max(result[i]) <0.4:
        pred_result.extend([_name_to_int(0)])
        confident_score.extend(['NAN'])

    if max(result[i]) >=0.4:
        pred_result.extend([_name_to_int(np.argmax(result[i], axis=-1))])
        confident_score.extend([max(result[i])])

gt=pd.read_csv('/data/stars/user/rdai/Toyota_deploy/deployment_codes1/output_files/P03T18C03_gt.csv')

'''
for i in range (33,17737):
    for k in range(len (gt)):
        if gt['start_frame'][k]<=i<=gt['end_frame'][k]:
            gt_list.extend([gt['event'][k]])
        elif 0 < i < gt['start_frame'][0]:
            gt_list.extend(['Background'])
        elif gt['end_frame'][len (gt)-1]<i<17738:
            gt_list.extend(['Background'])
        elif gt['end_frame'][k]<i<gt['start_frame'][k+1] and k<len (gt):
            gt_list.extend(['Background'])
'''
for k in range(len (gt)):
    if k==0:
        for i in range(32,gt['start_frame'][k]):
            gt_list.extend(['Background'])
    for i in range(gt['start_frame'][k],gt['end_frame'][k]):
        gt_list.extend([gt['event'][k]])
    if k!= len (gt)-1:
        for i in range(gt['end_frame'][k],gt['start_frame'][k+1]):
            gt_list.extend(['Background'])
        if gt['end_frame'][k] > gt['start_frame'][k + 1]:
            print ('overlapping exist!!')
            break
    if k== len (gt)-1:
        for i in range(gt['end_frame'][k], 14172):
            gt_list.extend(['Background'])



#17737
#14172

import glob
import cv2
import os
#src = cv.imread('test.jpg')

images = glob.glob("/data/stars/user/rdai/Toyota_deploy/deployment_codes1/data/rgb/Smarthomes/P03T18C03/*")
images.sort()
files = []
files.extend([images[i] for i in range(33, 14172)])
files.sort()
arr = []
counter=0
for i in files:
    if os.path.isfile(i):
        print counter
        img=cv2.imread(i)
        gt_label=gt_list[counter]
        pred_score=confident_score[counter]
        pred_label= pred_result[counter]
        AddText = img.copy()
        if pred_score != 'NAN':
            pred_score =round(pred_score, 2)

        #cv2.rectangle(AddText, (2, 2), (500, 100), (255, 255, 255),-1)
        if gt_label == pred_label:
            cv2.putText(AddText, 'Pred_label: ' + str(pred_label), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
            cv2.putText(AddText, 'Confident_score: ' + str(pred_score), (20, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7,(0, 255, 0), 1)
        else:
            cv2.putText(AddText, 'Pred_label: ' + str(pred_label), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255),1)
            cv2.putText(AddText, 'Confident_score: ' + str(pred_score), (20, 60), cv2.FONT_HERSHEY_COMPLEX,0.7, (0, 0, 255), 1)
        cv2.putText(AddText, 'Ground_truth: '+str(gt_label), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,0), 1)
        cv2.putText(AddText, 'Frame:  ' + str(counter+ 33), (20, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0),
                    1)
        #cvRectangle(AddText, cvPoint(10, 10), cvPoint(200, 200), cvScalar(0, 0, 255), 3, 4, 0);

        #cv2.imshow("image", AddText)

        counter_zfill="%05d" %counter
        cv2.imwrite("/data/stars/user/rdai/Toyota_deploy/deployment_codes1/output_files/P03T18C03/"+str(counter_zfill)+".jpg", AddText)
        counter = counter +1
        #break




for i in range(len(gt)-1):
    if gt['start_frame'][i+1]< gt['end_frame'][i]:
        print i
        print gt['start_frame'][i+1]
        print gt['end_frame'][i]




def thirdMax(nums):
    """
    :type nums: List[int]
    :rtype: int
    """
    temp = list(set(nums))
    temp.sort(reverse=True)
    if len(temp) <= 2:
        return temp[0]
    else:
        return temp[2]
















#np.savetxt('../output_files/pred_results.txt', parallel_model.predict_generator(generator=test_generator, use_multiprocessing=True, workers=cpu_count()-2))
print ('Predictions computed successfuly')
