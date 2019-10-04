# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2
import time

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize

from model import yolov3

 
seq=["E:\OBJECT_DECTECT\yolo3\YOLOv3_TensorFlow\MOT16-02.mp4","E:\OBJECT_DECTECT\yolo3\YOLOv3_TensorFlow\MOT16-04.mp4",
"E:\OBJECT_DECTECT\yolo3\YOLOv3_TensorFlow\MOT16-05.mp4","E:\OBJECT_DECTECT\yolo3\YOLOv3_TensorFlow\MOT16-09.mp4",
"E:\OBJECT_DECTECT\yolo3\YOLOv3_TensorFlow\MOT16-10.mp4","E:\OBJECT_DECTECT\yolo3\YOLOv3_TensorFlow\MOT16-11.mp4",
"E:\OBJECT_DECTECT\yolo3\YOLOv3_TensorFlow\MOT16-11.mp4","E:\OBJECT_DECTECT\yolo3\YOLOv3_TensorFlow\MOT16-13.mp4"]
for seqinfo in seq:
    f= open(seqinfo[42:50]+"yolo.txt")
    parser = argparse.ArgumentParser(description="YOLO-V3 video test procedure.")
    #parser.add_argument("input_video", type=str,default="E:\OBJECT_DECTECT\yolo3\YOLOv3_TensorFlow\MOT16-02.mp4",
                    # help="The path of the input video.")
    parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                        help="The path of the anchor txt file.")
    parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                        help="Resize the input image with `new_size`, size format: [width, height]")
    parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Whether to use the letterbox resize.")
    parser.add_argument("--class_name_path", type=str, default="./data/my_data/VOC2012/data.names",
                        help="The path of the class names.")
    parser.add_argument("--restore_path", type=str, default="E:\OBJECT_DECTECT\yolo3\YOLOv3_TensorFlow\checkpoint\VOC2012\VOC2012best_model_Epoch_34_step_6019_mAP_0.5873_loss_9.0562_lr_3e-05",
                        help="The path of the weights to restore.")
    args = parser.parse_args()

    args.anchors = parse_anchors(args.anchor_path)
    args.classes = read_class_names(args.class_name_path)
    args.num_class = len(args.classes)

    color_table = get_color_table(args.num_class)

    vid = cv2.VideoCapture("E:\OBJECT_DECTECT\yolo3\YOLOv3_TensorFlow\MOT16-02.mp4")
    video_frame_cnt = int(vid.get(7))
    video_width = int(vid.get(3))
    video_height = int(vid.get(4))
    video_fps = int(vid.get(5))
    '''
    #if args.save_video:
    #    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #   videoWriter = cv2.VideoWriter('video_result.mp4', fourcc, video_fps, (video_width, video_height))
    '''
    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
        yolo_model = yolov3(args.num_class, args.anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)

        saver = tf.train.Saver()
        saver.restore(sess, args.restore_path)
        #fp = open("filename.txt", "w")
        for frame in range((video_frame_cnt)):
            ret, img_ori = vid.read()
            if args.letterbox_resize:
                img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
            else:
                height_ori, width_ori = img_ori.shape[:2]
                img = cv2.resize(img_ori, tuple(args.new_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = np.asarray(img, np.float32)
            img = img[np.newaxis, :] / 255.

            start_time = time.time()
            boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
            end_time = time.time()

            # rescale the coordinates to the original image
            if args.letterbox_resize:
                boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
                boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio
            else:
                boxes_[:, [0, 2]] *= (width_ori/float(args.new_size[0]))

                boxes_[:, [1, 3]] *= (height_ori/float(args.new_size[1]))


            for i in range(len(boxes_)):
                x0, y0, x1, y1 = boxes_[i]
                plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=color_table[labels_[i]])
            
                w=x1-x0
                h=y1-y0
            # 寫入 This is a testing! 到檔案
                det=[frame,-1,x0,y0,w,h,-1,-1,-1,-1]
            
                print(','.join(map(str,det)),file=f)
        

            # 關閉檔案
            
        f.close()
    
        vid.release()

#python video_test.py ./data/demo_data/video.mp4