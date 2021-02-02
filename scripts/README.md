# Code for Spatio-temporal Attention model
By Srijan Das (srijan.das@inria.fr) & Rui Dai (rui.dai@inria.fr)

Requirements:
1. python 2.7
2. Keras 2.1.5
3. Tensorflow 1.4.1
4. cuda/8.0
5. cudnn/5.1-cuda-8.0

## 1. For clipped video

For help - use python train_attention.py -h or test.py -h

Input - ../data/rgb -video frames
        ../data/3D_skeleton  -3D skeleton data
        ../data/train_${dataset_name}.txt - training file containing training videos and so on for test and validation.

Output - model weights with name weights_${experiment_name} will be created in the script folder storing the trainned models.
         pre-trained models are stored in ../models folder
         prediction results are stored in pred_results.txt within output_files folder 

For training - 

./job.sh epoch_number name_of_experiment dataset_name (NTU/Smarthomes)

For testing - 

python test.py --model_name model_location (full path of model location) dataset_name (NTU/Smarthomes)



## 2.For untrimmed video by using window approach

Original videos, skeletons, annotations and pre-trained model are needed. 

Remarks:
videos should put in ../data/videos
Skeletons should put in ../data/3d_skeleton/Smarthomes/
annotations should put in ../data/ground_truth/
pre-trained models should put in ../models/

Processing:

### Detecting + Generating the video 
```
./Window_Job.sh video_name (ex: ./Window_Job.sh P03T18C03)

```
### Only for Detecting 
```
./Detection.sh video_name (ex: ./Detection.sh P03T18C03)

```

### Only for Generating video 
```
./Generate_video.sh video_name (ex: ./Generate_video.sh P03T18C03)
```