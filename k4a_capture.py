import os
import numpy as np
import cv2
import pyk4a
from pyk4a import PyK4A, Config
import open3d as o3d
from datetime import datetime

def save_pointcloud(points, file_name):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float32).reshape(-1, 3))
    o3d.io.write_point_cloud(file_name, pcd)

def save_aligned_images_and_point_cloud(save_dir):
    # 创建 PyK4A 实例
    k4a = PyK4A(
        Config(
            color_resolution=pyk4a.ColorResolution.RES_1080P,
            camera_fps=pyk4a.FPS.FPS_30,
            depth_mode=pyk4a.DepthMode.NFOV_UNBINNED,
            synchronized_images_only=True,
        )
    )
    k4a.start()

    # 读取并对齐 RGB 和深度图像
    while True:
        # 获取 RGB 和深度图像
        if k4a.get_capture():
            capture = k4a.get_capture()
            rgb_image = capture.color
            depth_image = capture.transformed_depth
            point_cloud = capture.transformed_depth_point_cloud

            print("rgb:",rgb_image.shape,'depth:',depth_image.shape)

            # 获取点云数据
            # point_cloud = k4a.generate_point_cloud(transformed_depth_image)

            if capture.transformed_depth is not None:
                depth_colormap = cv2.applyColorMap \
                    (cv2.convertScaleAbs(depth_image, alpha=0.008)
                     , cv2.COLORMAP_JET)
                cv2.imshow('depth_color', depth_colormap)
            if capture.color is not None:
                cv2.imshow("rgb", capture.color)

            key = cv2.waitKey(1)
            if key == ord('t'):
                timeStr = datetime.now().strftime("%Y%m%d_%H%M%S")
                # 保存对齐后的 RGB 和深度图像
                cv2.imwrite(os.path.join(save_dir, 'images/color_'+timeStr+'.png'), rgb_image)
                print('color saved', os.path.join(save_dir, 'images/color_'+timeStr+'.png'))
                cv2.imwrite(os.path.join(save_dir, 'depths/depth_'+timeStr+'.png'), depth_image)
                print('depth saved', os.path.join(save_dir, 'depths/depth_'+timeStr+'.png'))
                # 保存点云数据
                # save_pointcloud(point_cloud, os.path.join(save_dir, 'point_clouds/pc_'+timeStr+'.pcd'))
                # print('point_cloud saved', os.path.join(save_dir, 'point_clouds/pc_'+timeStr+'.pcd'))

            if key == 27:
                cv2.destroyAllWindows()
                break

    # 停止相机和关闭设备
    k4a.stop()

def main():
    # 设置保存路径
    save_directory = "./data"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 创建三个子文件夹
    subfolders = ['images', 'depths', 'point_clouds']
    for folder in subfolders:
        folder_path = os.path.join(save_directory, folder)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    # 运行保存对齐的图像和点云
    save_aligned_images_and_point_cloud(save_directory)

if __name__ == "__main__":
    main()
