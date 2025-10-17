# functions.py

import cv2
import numpy as np
import open3d as o3d
import os
import torch
import shutil


def stream_extrinsics(filepath):
    """
    使用生成器逐行读取外参文件，以节省内存。
    每次只产出（yield）一组外参。
    文件格式应为每行: rvec_x, rvec_y, rvec_z, t_x, t_y, t_z
    """
    if not os.path.exists(filepath):
        print(f"错误: 外参文件未找到于 '{filepath}'")
        return

    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            # 跳过空行或注释行
            if not line or line.startswith('#'):
                continue

            try:
                parts = [float(p) for p in line.split(',')]
                if len(parts) != 6:
                    print(f"警告: 第 {i + 1} 行格式不正确，需要6个值，已跳过。行内容: '{line}'")
                    continue

                rvec = np.array(parts[0:3], dtype=np.float32)
                tvec = np.array(parts[3:6], dtype=np.float32).reshape(3, 1)

                # 使用 yield 代替 return，每次调用会返回一组数据，并暂停在这里
                yield {'rvec': rvec, 'tvec': tvec}

            except ValueError as e:
                print(f"警告: 解析第 {i + 1} 行时出错: {e}，已跳过。行内容: '{line}'")


def load_lidar_points(filepath):
    """
    从 .pcd 或 .txt 文件加载雷达点云（包含强度信息）。
    .txt 文件格式应为 X, Y, Z, Intensity。
    返回点云坐标和强度值。
    """
    if not os.path.exists(filepath):
        print(f"错误: 点云文件未找到于 '{filepath}'")
        return None, None

    file_extension = os.path.splitext(filepath)[1].lower()
    points = None
    intensities = None
    try:
        if file_extension == '.pcd':
            pcd = o3d.io.read_point_cloud(filepath)
            points = np.asarray(pcd.points)
            if pcd.has_colors():
                intensities = np.asarray(pcd.colors)[:, 0]
            else:
                intensities = np.zeros(len(points))  # 如果没有强度，则默认为0
        elif file_extension == '.txt':
            data = np.loadtxt(filepath, ndmin=2)
            if data.shape[1] < 4:
                raise ValueError("TXT 文件每行至少需要包含 X, Y, Z, Intensity 四个值。")
            points = data[:, :3]
            intensities = data[:, 3]
        else:
            return None, None
        return points.astype(np.float32), intensities.astype(np.float32)
    except Exception as e:
        print(f"错误: 加载或解析点云文件 '{filepath}' 时出错: {e}")
        return None, None


def project_lidar_to_image_batch_pytorch(points_lidar_gpu, rvecs_gpu, tvecs_gpu, K_gpu):
    """
    (新增) 使用PyTorch在GPU上批量将雷达点云投影到图像平面。

    Args:
        points_lidar_gpu (torch.Tensor): 形状为 (N, 3) 的点云张量。
        rvecs_gpu (torch.Tensor): 形状为 (B, 3) 的旋转向量批次。
        tvecs_gpu (torch.Tensor): 形状为 (B, 3, 1) 的平移向量批次。
        K_gpu (torch.Tensor): 形状为 (3, 3) 的相机内参矩阵。

    Returns:
        - image_points_batch (torch.Tensor): 形状为 (B, N, 2) 的投影点。
        - depths_batch (torch.Tensor): 形状为 (B, N) 的深度值。
    """
    B = rvecs_gpu.shape[0]  # Batch size
    N = points_lidar_gpu.shape[0]  # Number of points

    # --- 1. 将rvecs (旋转向量) 批量转换为旋转矩阵 Rs ---
    theta = torch.norm(rvecs_gpu, p=2, dim=1, keepdim=True)
    theta = torch.clamp(theta, min=1e-8)
    rvecs_normalized = rvecs_gpu / theta

    K_cross = torch.zeros((B, 3, 3), device=rvecs_gpu.device)
    K_cross[:, 0, 1] = -rvecs_normalized[:, 2]
    K_cross[:, 0, 2] = rvecs_normalized[:, 1]
    K_cross[:, 1, 0] = rvecs_normalized[:, 2]
    K_cross[:, 1, 2] = -rvecs_normalized[:, 0]
    K_cross[:, 2, 0] = -rvecs_normalized[:, 1]
    K_cross[:, 2, 1] = rvecs_normalized[:, 0]

    cos_theta = torch.cos(theta).unsqueeze(2)
    sin_theta = torch.sin(theta).unsqueeze(2)

    I = torch.eye(3, device=rvecs_gpu.device).expand(B, 3, 3)
    Rs_gpu = I + sin_theta * K_cross + (1 - cos_theta) * torch.bmm(K_cross, K_cross)

    # --- 2. 批量进行坐标系转换和投影 ---
    points_lidar_expanded = points_lidar_gpu.unsqueeze(0).expand(B, N, 3)
    points_camera_batch = torch.bmm(Rs_gpu, points_lidar_expanded.transpose(1, 2)) + tvecs_gpu

    # --- 3. 获取深度值 ---
    depths_batch = points_camera_batch[:, 2, :]

    # --- 4. 批量投影到图像平面 ---
    depths_for_division = torch.clamp(depths_batch, min=1e-8)
    projected_points_batch = torch.matmul(K_gpu, points_camera_batch)
    image_points_batch_xy = projected_points_batch[:, :2, :] / depths_for_division.unsqueeze(1)

    return image_points_batch_xy.transpose(1, 2), depths_batch


# ==============================================================================
# (新增) 可视化功能函数
# ==============================================================================

def generate_final_visualization(best_extrinsics, lidar_paths, image_paths, output_dir, K_matrix):
    """
    使用找到的最优外参，生成并保存最终的可视化结果。
    这个函数在CPU上运行，因为它只在最后执行一次。
    """
    print("\n" + "=" * 25 + " 开始生成最终可视化结果 " + "=" * 25)
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    best_rvec = best_extrinsics['rvec']
    best_tvec = best_extrinsics['tvec']
    best_R, _ = cv2.Rodrigues(best_rvec)
    dist_coeffs = np.zeros(5)  # 假设没有畸变

    for lidar_path, image_path in zip(lidar_paths, image_paths):
        points, intensities = load_lidar_points(lidar_path)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if points is None or image is None:
            continue

        # 使用OpenCV的CPU函数进行单次投影
        image_points, _ = cv2.projectPoints(points, best_R, best_tvec, K_matrix, dist_coeffs)
        image_points = image_points.reshape(-1, 2)

        # 为了获取深度，需要手动计算
        points_camera = best_R @ points.T + best_tvec
        depths = points_camera[2, :]

        # 调用绘图函数
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_filepath = os.path.join(output_dir, f"proj_{base_name}.png")
        _visualize_and_save_single_image(
            image_path, output_filepath, image_points, depths, intensities, image
        )
        print(f"已保存可视化结果到: {output_filepath}")

    print("\n所有可视化图像生成完毕。")


def _visualize_and_save_single_image(image_path, output_path, image_points, depths, intensities, image_gray):
    """
    (辅助函数) 将单张图像的投影点可视化并保存。
    """
    vis_image = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    h, w = image_gray.shape

    COLOR_MISPROJECTION = (0, 0, 255)  # 红色
    COLOR_BOARD_CORRECT = (255, 0, 0)  # 蓝色
    COLOR_WALL_CORRECT = (0, 255, 0)  # 绿色

    for k, p in enumerate(image_points):
        if depths[k] > 0:
            px, py = int(p[0]), int(p[1])
            if 0 <= px < w and 0 <= py < h:
                pixel_value = image_gray[py, px]
                intensity = intensities[k]
                color = None

                is_board_misprojected = (intensity == 50 and pixel_value > 128)
                is_wall_misprojected = (intensity == 70 and pixel_value <= 128)

                if is_board_misprojected or is_wall_misprojected:
                    color = COLOR_MISPROJECTION
                elif intensity == 50:
                    color = COLOR_BOARD_CORRECT
                elif intensity == 70:
                    color = COLOR_WALL_CORRECT

                if color:
                    cv2.circle(vis_image, (px, py), radius=2, color=color, thickness=-1)

    cv2.imwrite(output_path, vis_image)

# ==============================================================================