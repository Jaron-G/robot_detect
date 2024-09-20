# detect_aruco_image.py
#   用法
# python detect_aruco_image.py --image images/example_01.png --type DICT_5X5_100
# 导入库
import argparse
import imutils
import cv2
import sys
import numpy as np

# 构造参数解析器并解析参数
ap = argparse.ArgumentParser()
ap.add_argument(
    "-i",
    "--image",
    required=True,
    help="Path to the input image containing the ArUCo tag",
)
ap.add_argument(
    "-t", "--type", type=str, default="DICT_4X4_50", help="Tpe of ArUCo tag to detect"
)
args = vars(ap.parse_args())

# 定义 OpenCV 支持的每个可能的 ArUco 标签的名称
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11,
}

# 从磁盘加载输入图像并调整其大小
print("[INFO] Loading image...")
image = cv2.imread(args["image"])
# image = imutils.resize(image, width=1920)
print(image.shape)
# 验证 OpenCV 是否支持提供的 ArUCo 标签
if ARUCO_DICT.get(args["type"], None) is None:
    print("[INFO] ArUCo tag of '{}' is not supported!".format(args["type"]))
    sys.exit(0)

# 加载 ArUCo 字典，抓取 ArUCo 参数并检测标记
print("[INFO] Detecting '{}' tags...".format(args["type"]))
arucoDict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[args["type"]])
arucoParams = cv2.aruco.DetectorParameters()

arucodetector = cv2.aruco.ArucoDetector(
    dictionary=arucoDict, detectorParams=arucoParams
)
(corners, ids, rejected) = arucodetector.detectMarkers(image)

# cameraMatrix = np.array([913.617065, 0, 960.503906,
#                          0, 913.455261, 550.489502,
#                          0, 0, 1]).reshape(3,3)
cameraMatrix = np.array([762.725, 0, 640.5, 0, 762.725, 640.5, 0, 0, 1]).reshape(3, 3)
dist = np.array([0.0515398, -0.00872068, 0.000730499, 0.000393782, 0.0000648475])
distCoeffs = dist[0:5].reshape(1, 5)
cv2.aruco.drawDetectedMarkers(image, corners, ids, (0, 255, 0))
cv2.imshow("kk", image)


rvecs, tvecs, kkk = cv2.aruco.estimatePoseSingleMarkers(
    corners[0], 50, cameraMatrix, distCoeffs
)

print(rvecs, tvecs)
print(len(rvecs))

for i in range(len(rvecs)):
    cv2.drawFrameAxes(image, cameraMatrix, distCoeffs, rvecs, tvecs, 40)

cv2.imshow("pose", image)
cv2.waitKey(0)

import matplotlib.pyplot as plt

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换代码
plt.imshow(image)
plt.show()
# # 验证至少一个 ArUCo 标记被检测到
# if len(corners) > 0:
#     # 展平 ArUCo ID 列表
#     ids = ids.flatten()
#     # 循环检测到的 ArUCo 标记
#     for (markerCorner, markerID) in zip(corners, ids):
#         # 提取始终按​​以下顺序返回的标记：
#         # TOP-LEFT, TOP-RIGHT, BOTTOM-RIGHT, BOTTOM-LEFT
#         corners = markerCorner.reshape((4, 2))
#         (topLeft, topRight, bottomRight, bottomLeft) = corners
#         # 将每个 (x, y) 坐标对转换为整数
#         topRight = (int(topRight[0]), int(topRight[1]))
#         bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
#         bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
#         topLeft = (int(topLeft[0]), int(topLeft[1]))
#         # 绘制ArUCo检测的边界框
#         cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
#         cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
#         cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
#         cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)
#         # 计算并绘制 ArUCo 标记的中心 (x, y) 坐标
#         cX = int((topLeft[0] + bottomRight[0]) / 2.0)
#         cY = int((topLeft[1] + bottomRight[1]) / 2.0)
#         cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
#         # 在图像上绘制 ArUco 标记 ID
#         cv2.putText(image, str(markerID), (topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#         print("[INFO] ArUco marker ID: {}".format(markerID))
#         # 显示输出图像
#         cv2.imshow("Image", image)
#         cv2.waitKey(0)


import transforms3d as tfs


def rodrigues_rotation(r, theta):
    # n旋转轴[3x1]
    # theta为旋转角度
    # 旋转是过原点的，n是旋转轴
    r = np.array(r).reshape(3, 1)
    rx, ry, rz = r[:, 0]
    M = np.array([[0, -rz, ry], [rz, 0, -rx], [-ry, rx, 0]])
    R = np.zeros([3, 3])
    R[:3, :3] = (
        np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * r @ r.T + np.sin(theta) * M
    )
    return R


def rodrigues_rotation_vec_to_R(v):
    # r旋转向量[3x1]
    theta = np.linalg.norm(v)
    r = np.array(v).reshape(3, 1) / theta
    return rodrigues_rotation(r, theta)


v = np.array([-2.89574422, -0.03344724, 0.3793738])
matrix = rodrigues_rotation_vec_to_R(v)
q = tfs.quaternions.mat2quat(matrix)

print(q)
