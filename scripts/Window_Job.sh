#!/usr/bin/env bash
#./Job.sh P03T18C03
export PATH=/home/sdas/anaconda2/bin:$PATH

echo 'Start processing: '$1

mkdir -p '../data/rgb/Smarthomes/'$1

ffmpeg -r 20 -i '../data/videos/'$1'.mp4' '../data/rgb/Smarthomes/'$1'/%8d.jpg'

echo 'Detecting...'

python ../scripts/Detection.py Smarthomes $1

echo 'Detection is done!'

echo 'Generating the video ...'

mkdir -p '../output_files/'$1

python ../scripts/Generate_video.py $1

ffmpeg -framerate 20 -pattern_type glob -i '../output_files/'$1'/*.jpg' -c:v libx264 -pix_fmt yuv420p '../Predicted_'$1'.mp4'

#ffmpeg -framerate 20 -pattern_type glob -i '../output_files/P03T18C03/*.jpg' '../Predicted_P03T18C03.mp4'

echo 'Job finished!'