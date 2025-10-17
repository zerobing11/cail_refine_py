# main_evaluate.py

import cv2
import numpy as np
import os
import glob
# 从我们的 functions.py 文件中导入所需的函数
from functions_2 import stream_extrinsics, load_lidar_points, project_lidar_to_image, _visualize_and_save_single_image, update_visualizations_for_best_extrinsics

# ==============================================================================
# 参数配置区域
# ==============================================================================
# 1. 包含多个外参(rvec, t)的文件路径
EXTRINSICS_FILE_PATH = "/home/lz/PycharmProjects/resize_image/evaluate/6DOF.txt"

# 2. 存放雷达点云文件的文件夹路径
LIDAR_FOLDER_PATH = "/home/lz/PycharmProjects/resize_image/evaluate/lidar"

# 3. 存放相机图像的文件夹路径
IMAGE_FOLDER_PATH = "/home/lz/PycharmProjects/resize_image/evaluate/mask_img"

# 4. 存放可视化结果图的文件夹路径
VISUALIZATION_OUTPUT_PATH = "/home/lz/PycharmProjects/resize_image/evaluate/visualization_results"

# ==============================================================================

def main():
    """
    主函数：测试多组外参，通过计算误投影点的总数来找出最优外参。
    误投影点定义为：
    1. 强度为50的点（板子）落在了图像的白色区域。
    2. 强度为70的点（背景墙）落在了图像的黑色区域。
    """
    # --- 1. 设置内参 ---
    K = np.array([
        [909.554565, 0.0, 630.406738],
        [0.0, 907.064636, 371.516388],
        [0.0, 0.0, 1.0]
    ])
    dist = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

    # --- 2. 加载数据文件路径 ---
    lidar_files = sorted(
        glob.glob(os.path.join(LIDAR_FOLDER_PATH, "*.pcd")) +
        glob.glob(os.path.join(LIDAR_FOLDER_PATH, "*.txt"))
    )
    image_files = sorted(
        glob.glob(os.path.join(IMAGE_FOLDER_PATH, "*.png")) +
        glob.glob(os.path.join(IMAGE_FOLDER_PATH, "*.jpg")) +
        glob.glob(os.path.join(IMAGE_FOLDER_PATH, "*.jpeg"))
    )

    num_pairs = min(len(lidar_files), len(image_files))
    if num_pairs == 0:
        print("错误：未能找到任何匹配的雷达-图像数据对。请检查文件夹路径。")
        return
    print(f"找到 {num_pairs} 对匹配的雷达和图像文件用于评估。\n")

    # --- 3. 遍历外参数据流，计算得分 ---
    best_extrinsics = None
    min_total_misprojections = float('inf')

    # stream_extrinsics 是一个生成器，循环会逐行读取文件，不会占用大量内存
    for i, extrinsics in enumerate(stream_extrinsics(EXTRINSICS_FILE_PATH)):
        if i % 100 == 0:
            print("\n" + "=" * 70)
            print(f"正在测试第 {i + 1} 组外参...")
            print("=" * 70)

        rvec = extrinsics['rvec']
        tvec = extrinsics['tvec']
        R, _ = cv2.Rodrigues(rvec)

        # 针对当前外参，累加所有图像-点云对的误投影点总数
        current_total_misprojections = 0
        for j in range(num_pairs):
            # 加载点云，现在同时返回点坐标和强度
            points_lidar, intensities = load_lidar_points(lidar_files[j])
            image = cv2.imread(image_files[j], cv2.IMREAD_GRAYSCALE)

            if points_lidar is None or intensities is None or image is None:
                print(
                    f"警告: 跳过数据对 {os.path.basename(lidar_files[j])} 和 {os.path.basename(image_files[j])} 因为数据加载失败。")
                continue

            image_points, depths = project_lidar_to_image(points_lidar, R, tvec, K, dist)
            h, w = image.shape

            for k, p in enumerate(image_points):
                # 只考虑在相机前方且在图像范围内的点
                if depths[k] > 0:
                    px, py = int(p[0]), int(p[1])
                    if 0 <= px < w and 0 <= py < h:
                        intensity = intensities[k]
                        pixel_value = image[py, px]

                        # 检查误投影
                        # 1. 板子点(强度50)落在白色区域(>128)
                        is_board_misprojected = (intensity == 50 and pixel_value > 128)
                        # 2. 背景墙点(强度70)落在黑色区域(<=128)
                        is_wall_misprojected = (intensity == 70 and pixel_value <= 128)

                        if is_board_misprojected or is_wall_misprojected:
                            current_total_misprojections += 1

        # 打印当前外参的结果，方便跟踪
        print(f"  测试完成: 第 {i + 1} 组外参，误投影点数: {current_total_misprojections}")

        # 比较当前外参的总误投影点数与已记录的最小值
        if current_total_misprojections < min_total_misprojections:
            min_total_misprojections = current_total_misprojections
            best_extrinsics = extrinsics
            print(f"  >>> 发现新的最优外参！已更新最小误投影点数: {min_total_misprojections}")

            # # 立即调用可视化更新函数
            # update_visualizations_for_best_extrinsics(
            #     best_extrinsics, lidar_files, image_files,
            #     VISUALIZATION_OUTPUT_PATH, K, dist
            # )

    # --- 4. 输出最终结果 ---
    print("\n" + "=" * 30 + " 最终最优结果 " + "=" * 30)
    if best_extrinsics:
        best_rvec = best_extrinsics['rvec']
        best_tvec = best_extrinsics['tvec'].flatten()
        print(f"已找到最优外参，其对应的误投影点总数最小。")
        print(f"最小误投影点数: {min_total_misprojections}")
        print("最优外参 (rvec_x, rvec_y, rvec_z, t_x, t_y, t_z):")
        print(f"{best_rvec[0]},{best_rvec[1]},{best_rvec[2]},{best_tvec[0]},{best_tvec[1]},{best_tvec[2]}")
    else:
        print("未能找到任何最优外参。请检查数据和参数配置。")


if __name__ == "__main__":
    main()