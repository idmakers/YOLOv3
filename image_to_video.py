import cv2
import numpy as np
import glob
from tkinter import Tk as tk
from tkinter import filedialog as file
'''
root = tk()
root.floder = file.askdirectory()
floder = str(root.floder)
'''
seq=["E:/OBJECT_DECTECT/MOT16/train/MOT16-02/img1","E:/OBJECT_DECTECT/MOT16/train/MOT16-04/img1",
"E:/OBJECT_DECTECT/MOT16/train/MOT16-05/img1","E:/OBJECT_DECTECT/MOT16/train/MOT16-09/img1",
"E:/OBJECT_DECTECT/MOT16/train/MOT16-10/img1","E:/OBJECT_DECTECT/MOT16/train/MOT16-11/img1",
"E:/OBJECT_DECTECT/MOT16/train/MOT16-11/img1","E:/OBJECT_DECTECT/MOT16/train/MOT16-13/img1"]
img_array = []
for seqinfo in seq:
    for filename in glob.glob(seqinfo +"/*.jpg"):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
        print(filename)




        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        out = cv2.VideoWriter(seqinfo[30:38]+".mp4",fourcc,30, size)
        
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()