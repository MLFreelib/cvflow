import cv2
import numpy as np

import argparse
class StereoCalibration(object):
    def __init__(self, save_path):
        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS +
                         cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.criteria_cal = (cv2.TERM_CRITERIA_EPS +
                             cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        self.objp = np.zeros((9 * 7, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:9, 0:7].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        self.objpoints = []  # 3d point in real world space
        self.imgpoints_l = []  # 2d points in image plane.
        self.imgpoints_r = []  # 2d points in image plane.
        self.fpath = save_path


    def preprocessing(self, frame):

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        v = cv2.equalizeHist(v)

        hsv = cv2.merge([h, s, v])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        return gray

    def read_images(self, v1, v2):
        counter = 0
        img_shape = None
        while True:

            _, img_l = v1.read()
            _, img_r = v2.read()
            # check to see if we have reached the end of the stream
            if img_l  is None or img_r is None:
                v1.release()
                v2.release()
                cv2.destroyAllWindows()
            cv2.imshow('left', img_l)
            cv2.imshow('right', img_r)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                break
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                v1.release()
                v2.release()
                cv2.destroyAllWindows()
                break

            try:
                img_l = (v1.read()[1])
                img_r = (v2.read()[1])
            except:
                break

            counter += 1

            gray_l = self.preprocessing(img_l)
            gray_r = self.preprocessing(img_r)
            img_shape = gray_l.shape[::-1]

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 7), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 7), None)

            # If found, add object points, image points (after refining them)

            if ret_l is True and ret_r is True:
                self.objpoints.append(self.objp)


                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, (9, 7),
                                                  corners_l, ret_l)



                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, (9, 7),
                                                  corners_r, ret_r)

            for _ in range(5):
                v1.read()
                v2.read()

            cv2.imshow('left', img_l)
            cv2.imshow('right', img_r)


        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_shape, None, None)

        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_shape, None, None)

        self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        flags |= cv2.CALIB_ZERO_TANGENT_DIST


        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)


        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                             ('dist2', d2), ('rvecs1', self.r1),
                             ('rvecs2', self.r2), ('R', R), ('T', T),
                             ('E', E), ('F', F)])
        self.save_config(self.fpath, camera_model)
    def save_config(self, conf_name, params):
        conf_header = '''[calib_parameters]
    values = [
        '''
        with open(conf_name, 'w') as f:
            f.writelines(conf_header)
            for param in params:
                f.write(f"\t'{{ {param}:{params[param]} }}',\n")
            f.write('\t]')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v1", "--video1", type=str,
                    help="path to  first input video file")
    ap.add_argument("-v2", "--video2", type=str,
                    help="path to  first input video file")
    ap.add_argument('-f','--save_path', type=str, default='conf.txt', help='Path to save bboxes')
    args = vars(ap.parse_args())

    v1 = cv2.VideoCapture(args["video1"])
    v2 = cv2.VideoCapture(args["video2"])
    fpath = args["save_path"]
    calib = StereoCalibration(fpath)
    calib.read_images(v1, v2)
