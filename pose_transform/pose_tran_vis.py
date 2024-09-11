# import numpy as np
# import transforms3d as tfs


# def rodrigues_rotation(r, theta):
#     # n旋转轴[3x1]
#     # theta为旋转角度
#     # 旋转是过原点的，n是旋转轴
#     r = np.array(r).reshape(3, 1)
#     rx, ry, rz = r[:, 0]
#     M = np.array([
#         [0, -rz, ry],
#         [rz, 0, -rx],
#         [-ry, rx, 0]
#     ])
#     R = np.zeros([3,3])
#     R[:3, :3] = np.cos(theta) * np.eye(3) +        \
#                 (1 - np.cos(theta)) * r @ r.T +    \
#                 np.sin(theta) * M
#     return R

# def rodrigues_rotation_vec_to_R(v):
#     # r旋转向量[3x1]
#     theta = np.linalg.norm(v)
#     print(theta)
#     r = np.array(v).reshape(3, 1) / theta
#     return rodrigues_rotation(r, theta)

# def r_t_to_homogeneous_matrix(R, T):
#     R1 = np.vstack([R, np.array([0, 0, 0])])
#     T1 = np.vstack([T, np.array([1])])
#     HomoMtr = np.hstack([R1, T1])
#     return HomoMtr


# rvec = np.array([[[-2.22 ,2.22,  0.0008 ]]] )
# tvec = np.array([[[  25.62165022, -169.08106376, 1140.32568696]]]).reshape(3,1)
# r_matrix_g2C = rodrigues_rotation_vec_to_R(rvec)
# matrix_g2C = r_t_to_homogeneous_matrix(r_matrix_g2C,tvec)
# print("matrix_g2C: ",matrix_g2C)

# # # 手眼矩阵，将相机与夹爪位置重合
# # matrix_C2H= np.array([[ 1, 0, 0, 0],[ 0, 1 , 0, 0],[0, 0 ,1 , 0],[0, 0 ,0 , 1]])

# # # current_pose = ur.get_current_pose()
# # current_position = np.array([ 100 , 200, 300]).reshape(3,1)
# # current_rotation = np.array([ 0, 1, 0, 0])
# # r_matrix_H2B = tfs.quaternions.quat2mat(current_rotation)
# # print(r_matrix_H2B)
# # matrix_H2B = r_t_to_homogeneous_matrix(r_matrix_H2B,current_position)
# # print("matrix_H2B: ",matrix_H2B)

# # matrix_g2B = matrix_H2B @ matrix_C2H @ matrix_g2C 
# # print("matrix_g2B: ",matrix_g2B)

# import open3d as o3d
# import copy
 
# # 在原点创建坐标框架网格


# matrix_g2H = np.array([[-9.96459111e-01 ,-7.93998847e-03  ,8.37030277e-02 ,-4.11441033e+01],
#  [-1.63738567e-04 , 9.95712387e-01 , 9.25030586e-02 , 3.50788020e+01],
#  [-8.40786147e-02 , 9.21618102e-02 ,-9.92187980e-01  ,4.68494050e+02],
#  [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])
# matrix_H2B=  np.array([[ 9.99947303e-01 ,-2.65442539e-03 ,-9.91691085e-03  ,7.06854461e+02],
#  [-2.64900833e-03 ,-9.99996335e-01 , 5.59339694e-04 , 1.73254724e+02],
#  [-9.91835923e-03 ,-5.33040239e-04, -9.99950670e-01 , 6.69520192e+02],
#  [ 0.00000000e+00 , 0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]])
# matrix_g2B=  np.array([[-9.95572366e-01 ,-1.14965747e-02 , 9.32925141e-02 , 6.60973398e+02],
#  [ 2.75633795e-03 ,-9.95636155e-01 ,-9.32794198e-02,  1.38547089e+02],
#  [ 9.39577938e-02, -9.26092669e-02,  9.91259530e-01 , 2.01438636e+02],
#  [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])


# cone = o3d.geometry.TriangleMesh.create_cone(radius=20.0,
#                                              height=50.0,
#                                              resolution=20,
#                                              split=1)
# cone.compute_vertex_normals()
# cone.paint_uniform_color([1, 0, 0])



# mesh_0= o3d.geometry.TriangleMesh.create_coordinate_frame( size=100,origin=[0, 0, 0])

# mesh_1 = copy.deepcopy(mesh_0).transform(matrix_g2H)
# cone_1 = copy.deepcopy(cone).transform(matrix_g2H)
# cone_1.paint_uniform_color([0, 1, 0])

# mesh_2 = copy.deepcopy(mesh_1).transform(matrix_H2B)
# cone_2 = copy.deepcopy(cone_1).transform(matrix_H2B)
# cone_2.paint_uniform_color([0, 0, 1])

# # mesh_3 = copy.deepcopy(mesh_0).transform(matrix_g2B)
# # cone_3 = copy.deepcopy(cone).transform(matrix_g2B)
# # cone_3.paint_uniform_color([1, 0, 1])


# mesh_g= o3d.geometry.TriangleMesh.create_coordinate_frame( size=100,origin=[0, 0, 0])
# mesh_g = mesh_g.translate([734,132,25])
# cone_g = o3d.geometry.TriangleMesh.create_cone(radius=20.0,
#                                              height=50.0,
#                                              resolution=20,
#                                              split=1)
# cone_g.compute_vertex_normals()
# cone_g.paint_uniform_color([0, 0, 0])
# cone_g.translate([734,132,25])

# # 打印网格中心坐标
# print(f'Center of mesh 0: {mesh_0.get_center()}')
# print(f'Center of mesh 1: {mesh_1.get_center()}')
# print(f'Center of mesh 2: {mesh_2.get_center()}')
# # print(f'Center of mesh 3: {mesh_3.get_center()}')
# # 可视化
# o3d.visualization.draw_geometries([mesh_0, mesh_1, mesh_2,cone,cone_1,cone_2,mesh_g,cone_g])

# import numpy as np
# import transforms3d as tfs
# def rodrigues_rotation(r, theta):
#     # n旋转轴[3x1]
#     # theta为旋转角度
#     # 旋转是过原点的，n是旋转轴
#     r = np.array(r).reshape(3, 1)
#     rx, ry, rz = r[:, 0]
#     M = np.array([
#         [0, -rz, ry],
#         [rz, 0, -rx],
#         [-ry, rx, 0]
#     ])
#     R = np.zeros([3,3])
#     R[:3, :3] = np.cos(theta) * np.eye(3) +        \
#                 (1 - np.cos(theta)) * r @ r.T +    \
#                 np.sin(theta) * M
#     return R

# def rodrigues_rotation_vec_to_R(v):
#     # r旋转向量[3x1]
#     theta = np.linalg.norm(v)
#     print(theta)
#     r = np.array(v).reshape(3, 1) / theta
#     return rodrigues_rotation(r, theta)

# def r_t_to_homogeneous_matrix(R, T):
#     R1 = np.vstack([R, np.array([0, 0, 0])])
#     T1 = np.vstack([T, np.array([1])])
#     HomoMtr = np.hstack([R1, T1])
#     return HomoMtr


# rvec = np.array([[[0 ,-3.14,  0 ]]] )
# tvec = np.array([[[  0, 0, 0]]]).reshape(3,1)
# r_matrix_g2C = rodrigues_rotation_vec_to_R(rvec)
# matrix_g2C = r_t_to_homogeneous_matrix(r_matrix_g2C,tvec)

# print("matrix_g2C: ",matrix_g2C)

# position = np.array([[  1.    ,       0.    ,       0.   ,       -242.62227074 ],
#  [  0.     ,      1.      ,     0.     ,    136.14071289],
#  [  0.      ,     0.     ,      1.    ,     473.98242631],
#  [  0.      ,     0.    ,       0.    ,       1.        ]])

# rot = np.array([[  0.    ,       1.    ,       0.   ,       0 ],
#  [  1.     ,      0.      ,     0.     ,    0],
#  [  0.      ,     0.     ,      -1.    ,     0],
#  [  0.      ,     0.    ,       0.    ,       1.        ]])


# print("位姿1：",position)

# position2 = matrix_g2C @ position.copy()

# print("位姿2：",position2)

# position3 = rot @ position2 

# print("位姿3：",position3)




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


rvec = np.array([[[-2.22 ,2.22,  0.0008 ]]] )
tvec = np.array([[[  25.62165022, -169.08106376, 1140.32568696]]]).reshape(3,1)
r_matrix_g2C = rodrigues_rotation_vec_to_R(rvec)
matrix_g2C = r_t_to_homogeneous_matrix(r_matrix_g2C,tvec)
print("matrix_g2C: ",matrix_g2C)

# # 手眼矩阵，将相机与夹爪位置重合
# matrix_C2H= np.array([[ 1, 0, 0, 0],[ 0, 1 , 0, 0],[0, 0 ,1 , 0],[0, 0 ,0 , 1]])

# # current_pose = ur.get_current_pose()
# current_position = np.array([ 100 , 200, 300]).reshape(3,1)
# current_rotation = np.array([ 0, 1, 0, 0])
# r_matrix_H2B = tfs.quaternions.quat2mat(current_rotation)
# print(r_matrix_H2B)
# matrix_H2B = r_t_to_homogeneous_matrix(r_matrix_H2B,current_position)
# print("matrix_H2B: ",matrix_H2B)

# matrix_g2B = matrix_H2B @ matrix_C2H @ matrix_g2C 
# print("matrix_g2B: ",matrix_g2B)

import open3d as o3d
import copy
 
# 在原点创建坐标框架网格


matrix_g2H = np.array([[-9.96459111e-01 ,-7.93998847e-03  ,8.37030277e-02 ,-4.11441033e+01],
 [-1.63738567e-04 , 9.95712387e-01 , 9.25030586e-02 , 3.50788020e+01],
 [-8.40786147e-02 , 9.21618102e-02 ,-9.92187980e-01  ,4.68494050e+02],
 [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])
matrix_H2B=  np.array([[ 9.99947303e-01 ,-2.65442539e-03 ,-9.91691085e-03  ,7.06854461e+02],
 [-2.64900833e-03 ,-9.99996335e-01 , 5.59339694e-04 , 1.73254724e+02],
 [-9.91835923e-03 ,-5.33040239e-04, -9.99950670e-01 , 6.69520192e+02],
 [ 0.00000000e+00 , 0.00000000e+00,  0.00000000e+00 , 1.00000000e+00]])
matrix_g2B=  np.array([[-9.95572366e-01 ,-1.14965747e-02 , 9.32925141e-02 , 6.60973398e+02],
 [ 2.75633795e-03 ,-9.95636155e-01 ,-9.32794198e-02,  1.38547089e+02],
 [ 9.39577938e-02, -9.26092669e-02,  9.91259530e-01 , 2.01438636e+02],
 [ 0.00000000e+00 , 0.00000000e+00 , 0.00000000e+00 , 1.00000000e+00]])


matrix_C2H=  np.array([[0.0536729,  0.7236786, -0.6880469,-0.1011*1000],
   [0.0390896,  0.6869892,  0.7256154, 0.09678*1000],
   [0.9977932, -0.0658414,  0.0085844 ,0.07269*1000],
   [0,0,0,1]])
cone = o3d.geometry.TriangleMesh.create_cone(radius=20.0,
                                             height=50.0,
                                             resolution=20,
                                             split=1)
cone.compute_vertex_normals()
cone.paint_uniform_color([1, 0, 0])

mesh_0= o3d.geometry.TriangleMesh.create_coordinate_frame( size=100,origin=[0, 0, 0])
mesh_1 = copy.deepcopy(mesh_0).transform(matrix_C2H)
cone_1 = copy.deepcopy(cone).transform(matrix_C2H)
cone_1.paint_uniform_color([0, 1, 0])


rot1 = np.array([  0.7071068, -0.7071068,  0.0000000,
   0.7071068,  0.7071068,  0.0000000,
   0.0000000,  0.0000000,  1.0000000 ]).reshape(3,3)
print(rot1)

matrix_C2H2=  np.array([[0.7071068,  -0.7071068, 0,-0.1011*1000],
   [0.7071068,  0.7071068,  0, 0.09678*1000],
   [0, 0,  1 ,0.07269*1000],
   [0,0,0,1]])

# cone_2 = copy.deepcopy(cone).rotate(rot1).translate([-0.1011*1000,0.09678*1000,0.07269*1000])
cone_2 = copy.deepcopy(cone).transform(matrix_C2H2)
cone_2.paint_uniform_color([0, 0, 1])
# mesh_2 = copy.deepcopy(mesh_0).rotate(rot1).translate([-0.1011*1000,0.09678*1000,0.07269*1000])
mesh_2 = copy.deepcopy(mesh_0).transform(matrix_C2H2)



# 打印网格中心坐标
print(f'Center of mesh 0: {mesh_0.get_center()}')
print(f'Center of mesh 1: {mesh_1.get_center()}')
# print(f'Center of mesh 3: {mesh_3.get_center()}')
# 可视化
o3d.visualization.draw_geometries([mesh_0, mesh_1,cone,cone_1,])


