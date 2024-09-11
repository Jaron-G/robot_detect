

import numpy as np
import transforms3d as tfs


def rodrigues_rotation(r, theta):
    # n旋转轴[3x1]
    # theta为旋转角度
    # 旋转是过原点的，n是旋转轴
    r = np.array(r).reshape(3, 1)
    rx, ry, rz = r[:, 0]
    M = np.array([
        [0, -rz, ry],
        [rz, 0, -rx],
        [-ry, rx, 0]
    ])
    R = np.zeros([3,3])
    R[:3, :3] = np.cos(theta) * np.eye(3) +        \
                (1 - np.cos(theta)) * r @ r.T +    \
                np.sin(theta) * M
    return R

def rodrigues_rotation_vec_to_R(v):
    # r旋转向量[3x1]
    theta = np.linalg.norm(v)
    print(theta)
    r = np.array(v).reshape(3, 1) / theta
    return rodrigues_rotation(r, theta)

rvec = np.array([[[-2.89574422 ,-0.03344724,  0.3793738 ]]] )
tvec = np.array([[[  25.62165022, -169.08106376, 1140.32568696]]])

print(rvec/np.pi*180)
matrix = rodrigues_rotation_vec_to_R(rvec)
matrix2= np.array([[ 0.96640693, -0.00602963, -0.25694608],
 [ 0.05089413, -0.975439 ,   0.21430946],
 [-0.25192743, -0.22018719 ,-0.94236414]])

q = tfs.quaternions.mat2quat(matrix)
q2 = tfs.quaternions.mat2quat(matrix2)
print(matrix)
print(matrix2)
print(q)
print(q2)