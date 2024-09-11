
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

def r_t_to_homogeneous_matrix(R, T):
    R1 = np.vstack([R, np.array([0, 0, 0])])
    T1 = np.vstack([T, np.array([1])])
    HomoMtr = np.hstack([R1, T1])
    return HomoMtr


rvec = np.array([[[0 ,0,  0 ]]] )
tvec = np.array([[[  25.62165022, -169.08106376, 1140.32568696]]]).reshape(3,1)
r_matrix_g2C = rodrigues_rotation_vec_to_R(rvec)
matrix_g2C = r_t_to_homogeneous_matrix(r_matrix_g2C,tvec)
print("matrix_g2C: ",matrix_g2C)

# 手眼矩阵，将相机与夹爪位置重合
matrix_C2H= np.array([[ 1, 0, 0, 0],[ 0, 1 , 0, 0],[0, 0 ,1 , 0],[0, 0 ,0 , 1]])

# current_pose = ur.get_current_pose()
current_position = np.array([ 100 , 200, 300]).reshape(3,1)
current_rotation = np.array([ 0, 1, 0, 0])
r_matrix_H2B = tfs.quaternions.quat2mat(current_rotation)
print(r_matrix_H2B)
matrix_H2B = r_t_to_homogeneous_matrix(r_matrix_H2B,current_position)
print("matrix_H2B: ",matrix_H2B)

matrix_g2B = matrix_H2B @ matrix_C2H @ matrix_g2C 
print("matrix_g2B: ",matrix_g2B)


rvec = np.identity(3)
print(rvec)