from imutils.video import VideoStream
from imutils import resize
import dlib
import cv2
import numpy as np
import torch


class Video:
    def __init__(self, model, tracker, lines, server='', tracking_frames=200, width=None):
        self.vs = VideoStream(server).start()
        self.model = model
        self.lines = lines
        self.input = None
        self.counted = 0
        self.width = width
        self.tracking_frames = tracking_frames

    def set_input(self, input_file):
        self.input = input_file

    def process(self):
        trackers = []
        if self.input:
            self.vs = cv2.VideoCapture(self.input)
            frame_number = 0
            rects = []
            while True:
                if frame_number == 1:

                    cv2.waitKey(30000)
                    break
                frame = self.vs.read()[1]
                cv2.imwrite('street.jpg', frame)
                if self.width:
                    frame = resize(frame, width=self.width)
                if frame_number % self.tracking_frames == 0:
                    trackers = []
                    #print('CORRECT', cresult.pred)
                    #print('BEFORE', frame.shape)
                    result = self.model.forward(frame)
                    #print('MY', self.model.res.pred)
                    boxes = result[0]['boxes']
                    #print('BOXES', boxes)
                    cnt = 1
                    for i in boxes:
                        print(f'BOX {cnt}', i.int().tolist(), result[0]['scores'][cnt-1])
                        cnt += 1
                        tracker = dlib.correlation_tracker()
                        #box_i = np.array(i[:4]).astype('int')
                        #rect = dlib.rectangle(box_i[0], box_i[1], box_i[2], box_i[3])
                        #tracker.start_track(frame, rect)
                        checked = False
                        trackers.append({'tracker': tracker, 'checked': checked})
                        #print(result[0]['boxes'])
                        #frame = result.render()[0]
                        #frame = self.model.res.render()
                        cv2.rectangle(frame, (i).int().tolist(), (255, 0, 0), 5)
                for tracker in trackers:
                    #tracker['tracker'].update(frame)
                    pos = tracker['tracker'].get_position()
                    box = [pos.left(), pos.top(), pos.right(), pos.bottom()]
                    box = np.array(box).astype('int')
                    for i in self.lines:
                        x1 = i[0][0]
                        x2 = i[1][0]
                        y1 = i[0][1]
                        y2 = i[1][1]
                        if min(y1, y2) < pos.bottom() < max(y1, y2):
                            if y2 - y1:
                                x = (pos.bottom() - y1) * (x2 - x1) / (y2 - y1) + x1
                                if min(box[0], box[2]) < x < max(box[0], box[2]):
                                    if frame_number == 0:
                                        tracker['checked'] = True
                                    if not tracker['checked']:
                                        self.counted += 1
                                        tracker['checked'] = True
                    rects.append(box)
                    cv2.rectangle(frame, (box[0], box[1], box[2], box[3]), (255, 0, 0), 5)
                for i in self.lines:
                    cv2.line(frame, i[0], i[1], i[2], i[3])
                cv2.putText(frame, f'OBJECTS: {self.counted}', (10, 100),
                            cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 0), 3)

                cv2.imshow("FRAME", frame)
                frame_number += 1
                frame_number %= self.tracking_frames
                cv2.waitKey(1)
