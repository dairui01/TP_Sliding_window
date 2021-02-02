#!/usr/bin/env bash
export PATH=/home/sdas/anaconda2/bin:$PATH

echo 'Start processing: '$1

echo 'Generating the video ...'

mkdir -p 'frame_original/'$1

ffmpeg -loglevel panic -i $1'.mp4' -vf scale=640:480 -q:v 1 -r 20 './frame_original/'$1'/%8d.jpg'

ffmpeg -loglevel panic -i P03T18C03.mp4 -vf scale=640:480 -q:v 1 -r 20 ./frame_original/P03T18C03/%8d.jpg

mkdir -p '../output_files/'$1

python ../scripts/Generate_video.py $1

ffmpeg -framerate 20 -pattern_type glob -i '../output_files/'$1'/*.jpg' -c:v libx264 -pix_fmt yuv420p '../Predicted_'$1'.mp4'

ffmpeg -framerate 20 -pattern_type glob -i './output_files/P02T14C03/*.jpg' -c:v libx264 -pix_fmt yuv420p './Predicted_P02T14C03.mp4'

#ffmpeg -framerate 20 -pattern_type glob -i '../output_files/P06T07C01/*.jpg' -c:v libx264 -pix_fmt yuv420p '../Predicted_P06T07C01.mp4'
#ffmpeg -framerate 20 -pattern_type glob -i '../output_files/P16T04C04/*.jpg' -c:v libx264 -pix_fmt yuv420p '../Predicted_P16T04C04.mp4'
#ffmpeg -framerate 20 -pattern_type glob -i '../output_files/P03T18C03/*.jpg' -c:v libx264 -pix_fmt yuv420p '../Predicted_P03T18C03.mp4'
echo 'Job finished!'
