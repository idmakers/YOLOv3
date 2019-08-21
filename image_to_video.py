import cv2
import numpy as np
import glob
from tkinter import Tk as tk
from tkinter import filedialog as file

root = tk()
root.floder = file.askdirectory()
floder = str(root.floder)

img_array = []
for filename in glob.glob(floder +"/*.jpg"):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)




fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter(floder[30:38]+".mp4",fourcc, 30, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()