import numpy as np
import cv2
import glob
import time

# 이미지 경로 생성
def getIMGPath(path, name, extension): 
    return "{}/{}.{}".format(path, name, extension)


class Calibration:
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane
    opt_mtx = None
    roi = None
    mtx = None
    dist = None
    rvecs = None
    tvecs = None
    # camMtx = {'mtx':0, 'dist':0, 'rvecs':0, 'tvecs':0} # mtx, dist, rvecs, tvecs
    camWidth = 0
    camHeight = 0
    fx = 0
    fy = 0
    opt_fx = 0
    opt_fy = 0

    def __init__(self, max_iter) -> None:
        # termination criteria: (type, max_iter, epsilon)
        # 반복 종료 조건
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, 0.0001)    

    def findCamMtx(self, col, row, checkerImgsPath):
        camMtx = {'ret': None, 'mtx': None, 'dist': None, 'rvecs': None, 'tvecs': None}
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        # np.float32 데이터 타입으로 행렬 생성
        # 각 값이 0. 인 (col-1 * row-1 ) * 3 matrix 생성
        objp = np.zeros(((col-1) * (row-1), 3), np.float32)
        # 행렬 전체 중 각 열의 두번째 값까지를 변경
        # mgrid[0:세로, 0:가로] ---> 세로 * 가로 행렬 생성
        # .T ----> 전치 행렬
        # reshape(x, y, z) ---> 행렬의 차원과 모양을 x*y*z로 바꾼 값 리턴. -1이 들어가면 자동으로 자리 맞춤.
        # 즉 데이터가 20개 있을 경우, (4, -1)이라면 20 = 4 * ? 이므로 -1에는 5가 자동으로 맞춰짐
        objp[:,:2] = np.mgrid[0:(col-1), 0:(row-1)].T.reshape(-1,2)

        scount = 1

        # checkerboard images 파일 리스트로 가져오기
        checkerImgList = glob.glob(checkerImgsPath)
        for checker in checkerImgList:
            img = cv2.imread(checker)
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # checker board corner 찾기
            # ret: 찾으면 True
            ret, corners = cv2.findChessboardCorners(gray, ((col-1), (row-1)), None)
            # If found, add object points, image points (after refining them)
            if ret == True:
                print(f'sample{scount}')
                scount += 1
                self.objpoints.append(objp)
                # 찾은 corner에 대한 값을 보정
                refineCorners = cv2.cornerSubPix(gray,corners,(11,11), (-1,-1), self.criteria)
                self.imgpoints.append(refineCorners)
                '''
                # 코너 그리기 및 표시
                img = cv2.drawChessboardCorners(img,  ((col-1), (row-1)), refineCorners, ret)
                resize_img = cv2.resize(img, dsize=(0, 0),fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
                cv2.imshow('img',resize_img)
                cv2.waitKey(500)
        cv2.destroyAllWindows()
        '''

        #  카메라 메트릭스 구하기    
        camMtx['ret'], camMtx['mtx'], camMtx['dist'], camMtx['rvecs'], camMtx['tvecs'] = cv2.calibrateCamera(self.objpoints, self.imgpoints, gray.shape[::-1], None, None) 
        self.fx = camMtx['mtx'][0][0] # 초점거리 x
        self.fy = camMtx['mtx'][1][1] # 초점거리 y
        self.mtx = camMtx['mtx']
        self.dist = camMtx['dist']
        self.rvecs = camMtx['rvecs']
        self.tvecs = camMtx['tvecs']

    def optimalMtx(self, imgPath):  # camera matrix 개선
        img = cv2.imread(imgPath)
        imgHeight, imgWidth = img.shape[:2]
        # camera matrix 개선
        newCameraMtx, roi=cv2.getOptimalNewCameraMatrix(self.mtx, self.dist, (imgWidth, imgHeight), 1, (imgWidth, imgHeight))
        self.opt_fx = newCameraMtx[0][0]
        self.opt_fy = newCameraMtx[1][1]
        self.opt_mtx = newCameraMtx
        self.roi = roi

    def removeImgDistort(self, imgPath, storePath): # 이미지 왜곡 제거
        img = cv2.imread(imgPath)
        # 왜곡 제거
        dst = cv2.undistort(img, self.mtx, self.dist, None, self.opt_mtx)
        x, y, w, h = self.roi
        dst = dst[y : y + h, x : x + w]
        cv2.imwrite(storePath, dst)
    
    def removeCamDistort(self, camIndex): # 카메라 왜곡 제거
        capture = cv2.VideoCapture(camIndex)
        if capture.isOpened():
            w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            while cv2.waitKey(33) < 0:
                ret, frame = capture.read()
                dst = cv2.undistort(frame, self.mtx, self.dist, None, self.opt_mtx)
                x, y, w, h = self.roi
                dst = dst[y : y + h, x : x + w]
                cv2.imshow("VideoFrame", dst)
                # time.sleep(1)
            capture.release()
            cv2.destroyAllWindows()
    
    def printfd(self): # 초점거리 출력
        print("<<focus distance>>")
        print(f"fx:{self.fx}")
        print(f"fy: {self.fy}")
        print(f"avg: {(self.fy+self.fx)/2}")
        print("<<optimal focus distance>>")
        print(f"opt_fx: {self.opt_fx}")
        print(f"opt_fy: {self.opt_fy}")
        print(f"avg: {(self.opt_fy+self.opt_fx)/2}")
    
    def printDist(self): # 왜곡계수 출력
        print('distort:')
        print(self.dist) 
    
    def printExVector(self): # 회전, 이동 백터 출력
        print('rvecs:')
        print(self.rvecs) 
        print('tvecs:')
        print(self.tvecs)

''' # 카메라 초점거리에 따른 실제 거리 계산 실험용
    def calRealDist(self, realSize, pxSize):
        realSize = np.float64(realSize)
        pxSize = np.float64(pxSize)
        print("By fx----")
        result = self.fx * realSize / pxSize
        print(f">> {result}")
        print("By fy----")
        result = self.fy * realSize / pxSize
        print(f">> {result}")
        print("By opt_fx----")
        result = self.opt_fx * realSize / pxSize
        print(f">> {result}")
        print("By opt_fy----")
        result = self.opt_fy * realSize / pxSize
        print(f">> {result}")
        print("By fx,fy avg----")
        result = ((self.fx + self.fy)/2) * realSize / pxSize
        print(f">> {result}")
        print("By opt fx,fy avg----")
        result = ((self.opt_fx + self.opt_fy)/2) * realSize / pxSize
        print(f">> {result}")
'''

class find3dPoint:
    # 2D-> 3D 변환정보
    transData = {'retaval':None, 'rvec':None, 'R':None, 'tvec':None}

    def __init__(self, mtx, d):
        self.cMtx = mtx
        self.dist = d
        self.fx = mtx[0][0]
        self.fy = mtx[1][1]
        self.cx = mtx[0][2]
        self.cy = mtx[1][2]
        self.height = 0


    def findRdata(self, points_2D, points_3D):
        self.transData['retval'], self.transData['rvec'], self.transData['tvec'] = cv2.solvePnP(points_3D, points_2D, self.cMtx, self.dist, rvec = None, tvec = None, useExtrinsicGuess=None, flags=None)
        self.transData['R'] = cv2.Rodrigues(self.transData['rvec'])[0]
    
    def distance(self, realSize, pxSize): # 실제 거리 계산
        realSize = realSize
        pxSize =pxSize
        # print("By fx,fy avg----")
        result = ((self.fx + self.fy)/2) * realSize / pxSize
        # print(f">> {result}")
        return result

    def worldPoint (self, x, y, realSize, pxSize):
        Rt = self.transData['R'].T

        # 픽셀좌표(x, y) -> 정규좌표 Pc(u, v, 1)
        u = (x - self.cx)/self.fx
        v = (y - self.cy)/self.fy
        Pc = np.array([[u, v, 1]]).T

        # Pc의 월드좌표 Pw = Rt* (Pc - t) : 행렬곱
        a = Rt
        b = (Pc - self.transData['tvec'])
        Pw = np.dot(a,b)
        # print('좌표',Pw)
        # print(Pc)

        # 지면과의 교점을 구해서 z
        c = (-self.transData['tvec'])
        Cw = np.dot(a,c)
        self.height = -self.distance(realSize, pxSize)
        k = self.height - Cw[2] / (Pw[2] - Cw[2])
        # print('k: ', k)
        P = Cw + k*(Pw-Cw)
        # print(P)
        # 월드 좌표 wx, wy, wz
        Wx = P[0]
        Wy = P[1]
        Wz = P[2]
        #print(f'x: {Wx}, y: {Wy}, z: {Wz}')
        return Wx, Wy # x, y 좌표 반환


    def printinputData(self):
        print('mxt: ', self.cMtx)
        print('dist: ', self.dist)
        print('fx: ', self.fx)
        print('fy: ', self.fy)
        print('cx: ', self.cx)
        print('cy: ', self.cy)

    def printTransData(self):
        print("<<Translate Data>>")
        print(f"ret: {self.transData['retval']}")
        print(f"rvec: {self.transData['rvec']}")
        print(f"rvec: {self.transData['R']}")
        print(f"tvec: {self.transData['tvec']}")