#!/bin/bash

export exec=HW5

rm $exec

/Developer/NVIDIA/CUDA-6.5/bin/nvcc -O3 *.cu *.cpp -I/usr/local/Cellar/opencv/2.4.9/include/opencv -I/usr/local/Cellar/opencv/2.4.9/include -I/Developer/NVIDIA/CUDA-6.5/include -I/Developer/NVIDIA/CUDA-6.5/includ/cuda/thrust  -lopencv_core -lopencv_imgproc -lopencv_highgui -o $exec
