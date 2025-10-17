# functions.py

import cv2
import numpy as np
import open3d as o3d
import os
import shutil # 引入shutil库，用于清空文件夹


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
            # 注意: open3d 对 .pcd 中自定义的 'intensity' 字段没有标准化的读取方法。
            # 一个常见的做法是将强度信息保存在颜色通道中。
            # 这里的代码假设强度信息可以从颜色中获取，你可能需要根据你的 .pcd 文件格式进行调整。
            pcd = o3d.io.read_point_cloud(filepath)
            points = np.asarray(pcd.points)
            if pcd.has_colors():
                # 假设强度存储在第一个颜色通道，并已做相应处理。
                intensities = np.asarray(pcd.colors)[:, 0]
                print(f"从 .pcd 文件 '{filepath}' 的颜色通道加载了强度信息。")
            else:
                print(f"警告: .pcd 文件 '{filepath}' 不包含可作为强度的颜色信息，无法处理。")
                return None, None
        elif file_extension == '.txt':
            # 假设 .txt 文件格式为: X Y Z Intensity
            data = np.loadtxt(filepath, ndmin=2)  # ndmin=2 确保即使文件只有一行也能正确处理
            if data.shape[1] < 4:
                raise ValueError("TXT 文件每行至少需要包含 X, Y, Z, Intensity 四个值。")
            points = data[:, :3]
            intensities = data[:, 3]
        else:
            print(f"不支持的文件格式: '{file_extension}'")
            return None, None
        return points, intensities
    except Exception as e:
        print(f"错误: 加载或解析点云文件 '{filepath}' 时出错: {e}")
        return None, None


def project_lidar_to_image(points_lidar, R, t, K, dist):
    """
    将雷达点云投影到图像平面，并返回投影点和深度。
    """
    points_lidar = points_lidar.astype(np.float32)

    # 1. 将点云转换到相机坐标系以计算深度
    points_camera = R @ points_lidar.T + t

    # 2. 获取深度值 (即相机坐标系下的Z轴坐标)
    depths = points_camera.T[:, 2]

    # 3. 使用 OpenCV 将三维点投影到图像平面
    rvec, _ = cv2.Rodrigues(R)
    image_points, _ = cv2.projectPoints(points_lidar, rvec, t, K, dist)

    # 4. 返回投影点和深度值
    return image_points.reshape(-1, 2), depths


def update_visualizations_for_best_extrinsics(best_extrinsics, lidar_paths, image_paths, output_dir, K, dist):
    """
    (新功能) 当找到新的最优外参时，清空旧的可视化结果并生成全新的可视化图。
    """
    print(f"  ...正在为新的最优外参生成可视化结果...")

    # 1. 清空或创建可视化输出文件夹
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)  # 删除整个文件夹树
    os.makedirs(output_dir)

    # 2. 使用最优外参重新计算投影并保存
    best_R, _ = cv2.Rodrigues(best_extrinsics['rvec'])
    best_tvec = best_extrinsics['tvec']
    num_pairs = min(len(lidar_paths), len(image_paths))

    for i in range(num_pairs):
        points_lidar, intensities = load_lidar_points(lidar_paths[i])
        image = cv2.imread(image_paths[i], cv2.IMREAD_GRAYSCALE)

        if points_lidar is None or intensities is None or image is None:
            continue

        image_points, depths = project_lidar_to_image(points_lidar, best_R, best_tvec, K, dist)
        h, w = image.shape

        misprojection_indices = set()
        for k, p in enumerate(image_points):
            if depths[k] > 0:
                px, py = int(p[0]), int(p[1])
                if 0 <= px < w and 0 <= py < h:
                    if (intensities[k] == 50 and image[py, px] > 128) or \
                            (intensities[k] == 70 and image[py, px] <= 128):
                        misprojection_indices.add(k)

        base_name = os.path.splitext(os.path.basename(image_paths[i]))[0]
        output_filepath = os.path.join(output_dir, f"proj_{base_name}.png")
        _visualize_and_save_single_image(
            image_paths[i], output_filepath, image_points, depths,
            intensities, misprojection_indices
        )
    print(f"  ...可视化结果已更新并保存至: {output_dir}")


def _visualize_and_save_single_image(image_path, output_path, image_points, depths, intensities, misprojection_indices):
    """
    (辅助函数) 将单张图像的投影点可视化并保存。
    """
    vis_image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
    h, w, _ = vis_image.shape

    COLOR_MISPROJECTION = (0, 0, 255)  # 红色
    COLOR_BOARD_CORRECT = (255, 0, 0)  # 蓝色
    COLOR_WALL_CORRECT = (0, 255, 0)  # 绿色

    for k, p in enumerate(image_points):
        if depths[k] > 0:
            px, py = int(p[0]), int(p[1])
            if 0 <= px < w and 0 <= py < h:
                color = None
                if k in misprojection_indices:
                    color = COLOR_MISPROJECTION
                elif intensities[k] == 50:
                    color = COLOR_BOARD_CORRECT
                elif intensities[k] == 70:
                    color = COLOR_WALL_CORRECT

                if color:
                    cv2.circle(vis_image, (px, py), radius=2, color=color, thickness=-1)

    cv2.imwrite(output_path, vis_image)