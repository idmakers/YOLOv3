import cv2
import numpy as np
import glob
 
img_array = []
for filename in glob.glob('E:/OBJECT_DECTECT/MOT16/train/MOT16-10/img1/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter('project.mp4',fourcc, 30, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()