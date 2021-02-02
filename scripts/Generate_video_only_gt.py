import numpy as np
import pandas as pd
#import Smarthome_labels
from argparse import ArgumentParser
import sys
import glob
import cv2
import os
from tqdm import tqdm

argv = sys.argv
# Parse arguments

parser = ArgumentParser(description='Generate video')
parser.add_argument('video_name', help='Video name')
args = parser.parse_args(argv[1:])

#gt_list=[]

#images = glob.glob("./frames_original/" +str(args.video_name) + "/*")
images = glob.glob("./frame_original/"+str(args.video_name)+"/*")
images.sort()
length = len(images)

gt=pd.read_csv("./"+str(args.video_name)+".csv")

frame={}

for num_frame in range(length):
    frame[str(num_frame)]=['','','','','']

for k in range(len (gt)):
    if k==0:
        for i in range(0,gt['start_frame'][k]):
            frame[str(i)][0]= 'Background'
    for i in range(gt['start_frame'][k],gt['end_frame'][k]):
        if frame[str(i)][0] == '':
            frame[str(i)][0]= gt['event'][k]
        elif frame[str(i)][1] == '':
            frame[str(i)][1]= gt['event'][k]
        elif frame[str(i)][2] == '':
            frame[str(i)][2]= gt['event'][k]
        elif frame[str(i)][3] == '':
            frame[str(i)][3]= gt['event'][k]
    if k!= len (gt)-1:
        for i in range(gt['end_frame'][k],gt['start_frame'][k+1]):
            if frame[str(i)][0] == '':
                frame[str(i)][0]='Background'

    if k== len (gt)-1:
        for i in range(gt['end_frame'][k], length-1):
            if frame[str(i)][0] == '':
                frame[str(i)][0]='Background'

counter=0

files = []
files.extend([images[i] for i in range(0, len(images))])
files.sort()
arr = []


for i in tqdm(files):
    if os.path.isfile(i):
        #print counter
        img=cv2.imread(i)
        gt_label1=frame[str(counter)][0]
        gt_label2 = frame[str(counter)][1]
        gt_label3 = frame[str(counter)][2]
        gt_label4 = frame[str(counter)][3]
        #pred_score=confident_score[counter]
        #pred_label= pred_result[counter]
        AddText = img.copy()
        #if pred_score != 'NAN':
        #    pred_score =round(pred_score, 2)

        #remove the comment to add a rectangle background of the text
        cv2.rectangle(AddText, (2, 2), (450, 90), (255, 255, 255),-1)
        #
        cv2.putText(AddText, 'gt1: ' + str(gt_label1), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 1)
        cv2.putText(AddText, 'gt2: ' + str(gt_label2), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 1)
        cv2.putText(AddText, 'gt3: ' + str(gt_label3), (20, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 1)
        cv2.putText(AddText, 'gt4: ' + str(gt_label4), (20, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 1)
        counter_zfill="%05d" %counter
        #cv2.imwrite("./output_files/"+str(args.video_name)+"/"+str(counter_zfill)+".jpg", AddText)
        cv2.imwrite("./output_files/"+str(args.video_name)+"/"+str(counter_zfill)+".jpg", AddText)
        counter = counter + 1
        if counter == 20:
            #break
            pass


#np.savetxt('../output_files/pred_results.txt', parallel_model.predict_generator(generator=test_generator, use_multiprocessing=True, workers=cpu_count()-2))
print ('The video output successfully')
