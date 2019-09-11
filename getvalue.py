import argparse
import os
#def opreatefile(input):
'''
parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
parser.add_argument("input", type=str,
                    help="The path of the input image.")
args = parser.parse_args()
'''
f = open('E:/OBJECT_DECTECT/VOCdevkit/VOC2012/ImageSets/Main/filename.txt', 'r')
input_string = f.read()
filename  = input_string.split()
filename.extend("q")
f.close

data ="E:/OBJECT_DECTECT/VOCdevkit/VOC2012/ImageSets/Main/" #data 主目錄
path = [os.path.join(data, '/ImageSets/Main/')]#label dataset

for i  in  range(len(filename)):
    if filename[i] == 'q':
        break
    else:
        file1 = open(data+(filename[i]) , 'r')
        file2 = open(data[:36]+"new_"+filename[i],"w")
        input_string = file1.read()
        name  = input_string.splitlines()
        file1.close

        for i in range (len(name)):
            look =name[i][12:14]
            if name[i][12:14] == " 1":
                file2.write(name[i][0:12]+"\n")
        file2.close()