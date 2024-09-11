 
import numpy as np
import time
import cv2
import cv2.aruco as aruco
 
#相机内参
import yaml
 
file_path = ("./标定文件.yaml")
# file_path = ("/home/pi/PycharmProjects/pythonProject/opencv_test/变焦相机标定参数.yaml")
###加载文件路径###
 
with open(file_path, "r") as file:
    parameter = yaml.load(file.read(), Loader=yaml.Loader)
    mtx = parameter['camera_matrix']
    dist = parameter['dist_coeff']
    camera_u = parameter['camera_u']
    camera_v = parameter['camera_v']
    mtx = np.array(mtx)
    dist = np.array(dist)
 
 
#windows
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)
 
#num = 0
while True:
    start = time.time()
    ret, frame = cap.read()
    # operations on the frame come here
 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_1000)
    parameters =  aruco.DetectorParameters_create()
 
 
    #lists of ids and the corners beloning to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray,
                                                          aruco_dict,
                                                          parameters=parameters)
 
 
    if ids is not None:
 
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, mtx, dist)
        # Estimate pose of each marker and return the values rvet and tvec---different
        # from camera coeficcients
        (rvec-tvec).any() # get rid of that nasty numpy value array error
 
#        aruco.drawAxis(frame, mtx, dist, rvec, tvec, 0.1) #Draw Axis
#        aruco.drawDetectedMarkers(frame, corners) #Draw A square around the markers
 
        for i in range(rvec.shape[0]):
            cv2.drawFrameAxes(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.03)
            aruco.drawDetectedMarkers(frame, corners)
        #显示ID，rvec,tvec, 旋转向量和平移向量
        cv2.putText(frame, "Id: " + str(ids), (10,40), font, 0.5, (0, 0, 255),1,cv2.LINE_AA)
        cv2.putText(frame, "rvec: " + str(rvec[i, :, :]), (10, 60), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "tvec: " + str(tvec[i, :, :]), (10,80), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
 
 
    else:
 
        cv2.putText(frame, "No Ids", (10,64), font, 1, (0,255,0),2,cv2.LINE_AA)
 
    end = time.time()
    # 计算并显示帧率
    cv2.putText(frame, "rate: " + str(1 / (end-start )), (10, 120), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.imshow("frame",frame)
 
    key = cv2.waitKey(1)
 
    if key == 27:         # 按esc键退出
        print('esc break...')
        cap.release()
        cv2.destroyAllWindows()
        break
 
    if key == ord(' '):   # 按空格键保存
#        num = num + 1
#        filename = "frames_%s.jpg" % num  # 保存一张图像
        filename = str(time.time())[:10] + ".jpg"
        cv2.imwrite(filename, frame)
 