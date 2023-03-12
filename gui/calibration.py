import cv2
import numpy as np

import argparse
class StereoCalibration(object):
    def __init__(self, v1, v2, fpath):
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

        self.read_images(v1, v2)
        self.fpath = fpath

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
            try:
                img_l = (v1.read()[1])
                img_r = (v2.read()[1])
            except:
                break
            counter += 1
            print(counter)
            img_l_s = img_l.copy()
            img_r_s = img_r.copy()

            gray_l = self.preprocessing(img_l)
            gray_r = self.preprocessing(img_r)
            img_shape = gray_l.shape[::-1]

            # Find the chess board corners
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, (9, 7), None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, (9, 7), None)

            # If found, add object points, image points (after refining them)

            if ret_l is True and ret_r is True:
                self.objpoints.append(self.objp)

                rt = cv2.cornerSubPix(gray_l, corners_l, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_l.append(corners_l)

                # Draw and display the corners
                ret_l = cv2.drawChessboardCorners(img_l, (9, 7),
                                                  corners_l, ret_l)
                cv2.imwrite(f"left/left_{counter}.png", img_l_s)

                rt = cv2.cornerSubPix(gray_r, corners_r, (11, 11),
                                      (-1, -1), self.criteria)
                self.imgpoints_r.append(corners_r)

                # Draw and display the corners
                ret_r = cv2.drawChessboardCorners(img_r, (9, 7),
                                                  corners_r, ret_r)
                cv2.imwrite(f"right/right_{counter}.png", img_r_s)

            for _ in range(20):
                v1.read()
                v2.read()

            cv2.imshow('left', img_l)
            cv2.imshow('right', img_r)
            key = cv2.waitKey(1) & 0xFF


        rt, self.M1, self.d1, self.r1, self.t1 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_l, img_shape, None, None)

        rt, self.M2, self.d2, self.r2, self.t2 = cv2.calibrateCamera(
            self.objpoints, self.imgpoints_r, img_shape, None, None)

        self.stereo_calibrate(img_shape)

    def stereo_calibrate(self, dims):
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        # flags |= cv2.CALIB_FIX_PRINCIPAL_POINT
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS
        flags |= cv2.CALIB_FIX_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
        # flags |= cv2.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5

        stereocalib_criteria = (cv2.TERM_CRITERIA_MAX_ITER +
                                cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        ret, M1, d1, M2, d2, R, T, E, F = cv2.stereoCalibrate(
            self.objpoints, self.imgpoints_l,
            self.imgpoints_r, self.M1, self.d1, self.M2,
            self.d2, dims,
            criteria=stereocalib_criteria, flags=flags)

        print('Intrinsic_mtx_1', M1)
        print('dist_1', d1)
        print('Intrinsic_mtx_2', M2)
        print('dist_2', d2)
        print('R', R)
        print('T', T)
        print('E', E)
        print('F', F)

        print('')

        camera_model = dict([('M1', M1), ('M2', M2), ('dist1', d1),
                             ('dist2', d2), ('rvecs1', self.r1),
                             ('rvecs2', self.r2), ('R', R), ('T', T),
                             ('E', E), ('F', F)])

        save_config(self.fpath, camera_model)
        cv2.destroyAllWindows()
        return camera_model



def save_config(conf_name, params):
    print(conf_name)
    conf_header = '''[calib_parameters]
values = [
    '''
    with open(conf_name, 'w') as f:
        f.writelines(conf_header)
        for param in params:
            print(param)
            f.write(f"\t'{{ {param[0]}:{param[1]} }}',\n")
        f.write('\t]')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-v1", "--video1", type=str,
                    help="path to  first input video file")
    ap.add_argument("-v2", "--video2", type=str,
                    help="path to  first input video file")
    ap.add_argument('-f','--fpath', type=str, default='conf.txt', help='Path to save bboxes')
    args = vars(ap.parse_args())

    v1 = cv2.VideoCapture(args["video1"])
    v2 = cv2.VideoCapture(args["video2"])
    fpath = args["fpath"]
    StereoCalibration(v1, v2, fpath)
#
#
# for _ in range(50):
#     l = (v1.read()[1])[45:-300,:]
#     r = (v2.read()[1])[:-45-300,:]


