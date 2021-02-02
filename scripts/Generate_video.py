import numpy as np
import pandas as pd
import Smarthome_labels
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

result=np.load('../output_files/detection_'+str(args.video_name)+'.npy')

pred_result=[]
confident_score=[]
gt_list=[]

images = glob.glob("../data/rgb/Smarthomes/" +str(args.video_name) + "/*")
images.sort()
length = len(images) -32

for i in range(len(result)):
    #None
    if max(result[i]) <0.4:
        pred_result.extend([Smarthome_labels._int_to_name(0)])
        confident_score.extend(['NAN'])

    if max(result[i]) >=0.4:
        pred_result.extend([Smarthome_labels._int_to_name(np.argmax(result[i], axis=-1))])
        confident_score.extend([max(result[i])])

gt=pd.read_csv("../data/ground_truth/"+str(args.video_name)+".csv")

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
        for i in range(gt['end_frame'][k], length):
            gt_list.extend(['Background'])


files = []
files.extend([images[i] for i in range(33, length)])
files.sort()
arr = []
counter=0


min_length=min(len(files),len(pred_result),len(gt_list),len(confident_score))

files=files[:min_length]
pred_result=pred_result[:min_length]
gt_list=gt_list[:min_length]
confident_score=confident_score[:min_length]

for i in tqdm(files):
    if os.path.isfile(i):
        #print counter
        img=cv2.imread(i)
        gt_label=gt_list[counter]
        pred_score=confident_score[counter]
        pred_label= pred_result[counter]
        AddText = img.copy()
        if pred_score != 'NAN':
            pred_score =round(pred_score, 2)

        #remove the comment to add a rectangle background of the text
        cv2.rectangle(AddText, (2, 2), (450, 50), (255, 255, 255),-1)
        #
        cv2.putText(AddText, 'Pred_label: ' + str(pred_label), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 1)
        cv2.putText(AddText, 'Confident_score: ' + str(pred_score), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7,(255, 0, 0), 1)
        #change the text color for prediction and confidence score
        '''
        if gt_label == pred_label:
            cv2.putText(AddText, 'Pred_label: ' + str(pred_label), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
            cv2.putText(AddText, 'Confident_score: ' + str(pred_score), (20, 60), cv2.FONT_HERSHEY_COMPLEX, 0.7,(0, 255, 0), 1)
        else:
            cv2.putText(AddText, 'Pred_label: ' + str(pred_label), (20, 40), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
            cv2.putText(AddText, 'Confident_score: ' + str(pred_score), (20, 60), cv2.FONT_HERSHEY_COMPLEX,0.7, (0, 0, 255), 1)
        #ground truth
        #cv2.putText(AddText, 'Ground_truth: '+str(gt_label), (20, 20), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,0,0),  1)
        #frame
        #cv2.putText(AddText, 'Frame:  ' + str(counter+ 33), (20, 80), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 1)

        #show the output in realtime
        #cv2.imshow("image", AddText)
        '''
        counter_zfill="%05d" %counter
        cv2.imwrite("../output_files/"+str(args.video_name)+"/"+str(counter_zfill)+".jpg", AddText)
        counter = counter + 1
        if counter == 20:
            #break
            pass

'''
for i in range(len(gt)-1):
    if gt['start_frame'][i+1]< gt['end_frame'][i]:
        print i
        print gt['start_frame'][i+1]
        print gt['end_frame'][i]
'''

#np.savetxt('../output_files/pred_results.txt', parallel_model.predict_generator(generator=test_generator, use_multiprocessing=True, workers=cpu_count()-2))
print ('The video output successfully')
