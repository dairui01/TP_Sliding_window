#!/usr/bin/env bash
export PATH=/home/sdas/anaconda2/bin:$PATH

echo 'Start processing: '$1

mkdir -p '../data/rgb/Smarthomes/'$1

ffmpeg -r 20 -i '../data/videos/'$1'.mp4' '../data/rgb/Smarthomes/'$1'/%8d.jpg'

echo 'Detecting...'

python ../scripts/Detection.py Smarthomes $1

echo 'Detection is done!'








