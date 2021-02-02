# TP_Sliding_window
By Rui Dai (rui.dai@inria.fr) & Srijan Das (srijan.das@inria.fr)

Requirements:
1. python 2.7
2. Keras 2.1.5
3. Tensorflow 1.4.1
4. cuda/8.0
5. cudnn/5.1-cuda-8.0

## 1. For untrimmed video by using window approach

Original videos, skeletons, annotations and pre-trained model are needed. 

Remarks:
videos should put in ../data/videos (see the download link in the folder)
Skeletons should put in ../data/3d_skeleton/Smarthomes/
annotations should put in ../data/ground_truth/
pre-trained models should put in ../models/ (see the download link in the folder)

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