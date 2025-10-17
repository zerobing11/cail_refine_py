# main_evaluate.py

import cv2
import numpy as np
import os
import glob
import torch
import time

# 从我们的 functions.py 文件中导入所需的函数
from functions_cuda import stream_extrinsics, load_lidar_points, project_lidar_to_image_batch_pytorch, generate_final_visualization, _visualize_and_save_single_image

# ==============================================================================
# 参数配置区域
# ==============================================================================
EXTRINSICS_FILE_PATH = "/home/lz/PycharmProjects/resize_image/R_sample/sampled_poses_6dof.txt"
LIDAR_FOLDER_PATH = "/home/lz/PycharmProjects/resize_image/evaluate/lidar"
IMAGE_FOLDER_PATH = "/home/lz/PycharmProjects/resize_image/evaluate/mask_img"
VISUALIZATION_OUTPUT_PATH = "/home/lz/PycharmProjects/resize_image/evaluate/visualization_results"

# --- 新增GPU相关配置 ---
BATCH_SIZE = 1500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """
    主函数：使用GPU批量处理，高速执行评估和可视化流程。
    """
    print(f"正在使用设备: {DEVICE}")
    if not torch.cuda.is_available():
        print("警告: 未找到CUDA设备, 将使用CPU运行，速度会很慢。")

    # --- 1. 设置相机内参 ---
    K_np = np.array([[909.554565, 0.0, 630.406738],
                     [0.0, 907.064636, 371.516388],
                     [0.0, 0.0, 1.0]], dtype=np.float32)
    K_gpu = torch.from_numpy(K_np).to(DEVICE)

    # --- 2. 加载所有数据文件并预处理 ---
    lidar_files = sorted(glob.glob(os.path.join(LIDAR_FOLDER_PATH, "*.pcd")) + \
                         glob.glob(os.path.join(LIDAR_FOLDER_PATH, "*.txt")))
    image_files = sorted(glob.glob(os.path.join(IMAGE_FOLDER_PATH, "*.png")) + \
                         glob.glob(os.path.join(IMAGE_FOLDER_PATH, "*.jpg")) + \
                         glob.glob(os.path.join(IMAGE_FOLDER_PATH, "*.jpeg")))

    if not lidar_files or not image_files:
        print("错误：未能找到任何雷达或图像文件。")
        return

    print("正在加载并预处理所有点云和图像数据...")
    all_points_gpu = [torch.from_numpy(load_lidar_points(p)[0]).to(DEVICE) for p in lidar_files]
    all_intensities_gpu = [torch.from_numpy(load_lidar_points(p)[1]).to(DEVICE) for p in lidar_files]
    all_images_gpu = [torch.from_numpy(cv2.imread(p, cv2.IMREAD_GRAYSCALE)).to(DEVICE) for p in image_files]

    # --- 3. 运行评估，找到最优外参 ---
    best_extrinsics_info = {'rvec': None, 'tvec': None}
    min_total_misprojections = float('inf')

    extrinsics_generator = stream_extrinsics(EXTRINSICS_FILE_PATH)
    is_running = True
    batch_index = 0
    total_processed = 0
    start_time = time.time()

    while is_running:
        rvecs_batch = []
        tvecs_batch = []

        for _ in range(BATCH_SIZE):
            ext = next(extrinsics_generator, None)
            if ext is None:
                is_running = False
                break
            rvecs_batch.append(ext['rvec'])
            tvecs_batch.append(ext['tvec'])

        if not rvecs_batch:
            break

        current_batch_size = len(rvecs_batch)
        rvecs_gpu = torch.from_numpy(np.array(rvecs_batch)).to(DEVICE)
        tvecs_gpu = torch.from_numpy(np.array(tvecs_batch)).to(DEVICE)

        batch_misprojections = torch.zeros(current_batch_size, device=DEVICE, dtype=torch.long)

        for points_gpu, intensities_gpu, image_gpu in zip(all_points_gpu, all_intensities_gpu, all_images_gpu):
            h, w = image_gpu.shape
            image_points_batch, depths_batch = project_lidar_to_image_batch_pytorch(points_gpu, rvecs_gpu, tvecs_gpu,
                                                                                    K_gpu)

            px = image_points_batch[..., 0].long()
            py = image_points_batch[..., 1].long()
            valid_mask = (depths_batch > 0) & (px >= 0) & (px < w) & (py >= 0) & (py < h)

            px_valid = torch.clamp(px, 0, w - 1)
            py_valid = torch.clamp(py, 0, h - 1)
            pixel_values = image_gpu[py_valid, px_valid]

            board_misprojected = (intensities_gpu == 50) & (pixel_values > 128)
            wall_misprojected = (intensities_gpu == 20) & (pixel_values <= 128)

            misprojection_mask = (board_misprojected | wall_misprojected) & valid_mask
            batch_misprojections += torch.sum(misprojection_mask, dim=1)

        batch_misprojections_cpu = batch_misprojections.cpu().numpy()
        min_in_batch = batch_misprojections_cpu.min()

        if min_in_batch < min_total_misprojections:
            min_total_misprojections = min_in_batch
            best_in_batch_idx = batch_misprojections_cpu.argmin()

            rvec = rvecs_batch[best_in_batch_idx]
            tvec = tvecs_batch[best_in_batch_idx]
            best_extrinsics_info['rvec'] = rvec
            best_extrinsics_info['tvec'] = tvec

            print("\n" + "*"*20 + " 发现新的最优外参 " + "*"*20)
            print(f"新的最小误投影点数: {min_total_misprojections}")
            t_flat = tvec.flatten()
            print("新最优外参 (rvec_x, rvec_y, rvec_z, t_x, t_y, t_z):")
            print(f"{rvec[0]},{rvec[1]},{rvec[2]},{t_flat[0]},{t_flat[1]},{t_flat[2]}")
            print("*"*58)


        total_processed += current_batch_size
        batch_index += 1
        if batch_index % 100 == 0:
            print(f"已处理 {total_processed} 组外参... "
                  f"当前最优误差点: {min_total_misprojections} "
                  f"(耗时: {time.time() - start_time:.2f}s)")

    # --- 4. 输出并可视化最终的最优结果 ---
    print("\n" + "=" * 30 + " 评估完成：最终最优结果 " + "=" * 30)
    if best_extrinsics_info['rvec'] is not None:
        r = best_extrinsics_info['rvec']
        t = best_extrinsics_info['tvec'].flatten()
        print(f"已找到最优外参。")
        print(f"最小误投影点数: {min_total_misprojections}")
        print("最优外参 (rvec_x, rvec_y, rvec_z, t_x, t_y, t_z):")
        print(f"{r[0]},{r[1]},{r[2]},{t[0]},{t[1]},{t[2]}")

        # # 调用最终的可视化函数
        # generate_final_visualization(
        #     best_extrinsics_info, lidar_files, image_files,
        #     VISUALIZATION_OUTPUT_PATH, K_np
        # )
    else:
        print("未能找到任何最优外参。")


if __name__ == "__main__":
    main()