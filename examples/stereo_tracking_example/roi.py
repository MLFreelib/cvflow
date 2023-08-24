import argparse
import json
import time
from random import randrange
import configparser
from common.utils import argparser  as ap

import cv2
import numpy as np


def calculate_dist_btw_2_objects(frame, bboxes):
    if bboxes is None:
        return
    new_bboxes = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    roi1, roi2 = bboxes
    left = roi1 if roi1[0]<roi2[0] else roi2
    right = roi1 if roi1[0]>roi2[0] else roi2
    x, y, w, h = [int(v) for v in left]
    left_object = gray[y:y + h, x:x + w]
    left_object = cv2.adaptiveThreshold(left_object, 255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    left_object = cv2.morphologyEx(left_object, cv2.MORPH_OPEN, np.ones((3, 3)))
    cv2.rectangle(left_object, (0,0, *left_object.shape[:2]), (0, 0, 0), 1)
    xt, yt, wt, ht = cv2.boundingRect(left_object)
    bbox = (x+xt, y+yt, wt, yt)

    new_bboxes.append(bbox)

    right_p = (x + xt + wt - 1, y + np.argmax(left_object[:, xt + wt - 1]))
    cv2.circle(frame, right_p, 8, (0, 50, 255), -1)

    x, y, w, h = [int(v) for v in right]
    right_object = gray[y:y + h, x:x + w]

    right_object = cv2.adaptiveThreshold(right_object, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    right_object = cv2.morphologyEx(right_object, cv2.MORPH_OPEN, np.ones((5, 5)))
    cv2.rectangle(left_object, (0, 0, *right_object.shape[:2]), (0, 0, 0), 1)

    xt, yt, wt, ht  = cv2.boundingRect(right_object)  # Replaced code
    bbox = (x + xt, y + yt, wt, yt)
    # cv2.rectangle(frame, bbox, (0, 255, 0), 2)
    new_bboxes.append(bbox)

    left_p = (x + xt, y + np.argmax(right_object[:, xt]))
    cv2.circle(frame, left_p, 8, (0, 50, 255), -1)

    cv2.line(frame, left_p, right_p, (0, 50, 255), 2)

    dist = left_p[0]-right_p[0]
    return  dist


def save_config(conf_name, bboxes):
    print(conf_name)


    config_object = configparser.ConfigParser()

    with open(conf_name, 'w') as f:
        points = []
        for i in bboxes:
            print(i)
            x_1 = int(i[0])
            x_2 = int(i[2])
            y_1 = int(i[1])
            y_2 = int(i[3])
            points.append({"points":((x_1, y_1), (x_2, y_2)),
                           "color": (randrange(0, 255), randrange(0, 255), randrange(0, 255)),
                           "thickness": 2
                          })
        config_object['bboxes'] = {"values": points}
        config_object.write(f)

def get_roi(vs, n, f):
    # loop over frames from the video stream
    bboxes = []
    while True:
        _, frame = vs.read()

        # check to see if we have reached the end of the stream
        if frame is None:
            break
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            key = cv2.waitKey(1) & 0xFF
            while True:
                key = cv2.waitKey(1) & 0xFF
                if key==ord("n"):
                    box = cv2.selectROI("Frame", frame, fromCenter=False,
                                        showCrosshair=True)
                    (x, y, w, h) = [int(v) for v in box]
                    bboxes.append(box)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.imshow("Frame", frame)
                elif len(bboxes)==n:
                    vs.release()
                    save_config("conf.txt", bboxes)
                    cv2.destroyAllWindows()
                    break
                elif key==ord("t"):
                    break
                elif key == ord("q"):
                    vs.release()
                    save_config(f, bboxes)
                    cv2.destroyAllWindows()
            break

        cv2.imshow("Frame", frame)

if __name__ == '__main__':
    #
    args = vars(ap.parse_args())

    video = cv2.VideoCapture(args["videofile"])
    num = int(args["num"])
    fpath = args["destination"]
    get_roi(video, num, fpath)