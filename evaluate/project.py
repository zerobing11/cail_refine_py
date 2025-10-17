import cv2
import numpy as np
import open3d as o3d
import os
import glob

# ==============================================================================
# 参数配置区域
# ==============================================================================
CALIB_FILE_PATH = "/home/lz/cal_refine_py/Extrinsic.txt"
LIDAR_FOLDER_PATH = "/home/lz/cal_refine_py/evaluate/lidar"
IMAGE_FOLDER_PATH = "/home/lz/cal_refine_py/evaluate/mask_img"
OUTPUT_FOLDER_PATH = "/home/lz/cal_refine_py/evaluate/projection_results"
# ==============================================================================

def load_extrinsic_and_intrinsic(filepath):
    if not os.path.exists(filepath):
        print(f"错误: 标定文件未找到于 '{filepath}'")
        return None, None, None, None

    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]

    if len(lines) < 8:
        print(f"错误: 文件 '{filepath}' 格式不正确，至少需要8行数据。")
        return None, None, None, None

    try:
        R = np.array([list(map(float, line.split())) for line in lines[0:3]])
        t = np.array(list(map(float, lines[3].split()))).reshape(3, 1)
        K = np.array([list(map(float, line.split())) for line in lines[4:7]])
        dist = np.array(list(map(float, lines[7].split())))
        return R, t, K, dist
    except (ValueError, IndexError) as e:
        print(f"错误: 解析文件 '{filepath}' 时出错: {e}")
        return None, None, None, None

def load_lidar_points(filepath):
    if not os.path.exists(filepath):
        print(f"错误: 点云文件未找到于 '{filepath}'")
        return None, None

    file_extension = os.path.splitext(filepath)[1].lower()
    try:
        if file_extension == '.pcd':
            pcd = o3d.io.read_point_cloud(filepath)
            points = np.asarray(pcd.points)
            intensities = np.zeros((points.shape[0],))
        elif file_extension == '.txt':
            data = np.loadtxt(filepath)
            if data.shape[1] < 4:
                raise ValueError("TXT 文件每行需包含 X, Y, Z 和 intensity 四个值。")
            points = data[:, :3]
            intensities = data[:, 3]
        else:
            return None, None
        return points, intensities
    except Exception as e:
        print(f"错误: 加载或解析点云文件 '{filepath}' 时出错: {e}")
        return None, None

def project_lidar_to_image(points_lidar, R, t, K, dist):
    points_lidar = points_lidar.astype(np.float32)
    rvec, _ = cv2.Rodrigues(R)
    image_points, _ = cv2.projectPoints(points_lidar, rvec, t, K, dist)
    points_camera = R @ points_lidar.T + t
    return image_points.reshape(-1, 2), points_camera.T[:, 2]

def draw_projection_on_image(image, image_points, intensities, depths):
    proj_img = image.copy()
    h, w, _ = proj_img.shape
    valid_indices = [
        i for i, p in enumerate(image_points)
        if depths[i] > 0 and 0 <= p[0] < w and 0 <= p[1] < h
    ]

    if not valid_indices:
        print("图像范围内没有有效的投影点。")
        return proj_img

    valid_points = image_points[valid_indices]
    valid_intensities = intensities[valid_indices]

    max_intensity = np.max(valid_intensities)
    min_intensity = np.min(valid_intensities)
    norm_intensities = (valid_intensities - min_intensity) / (max_intensity - min_intensity + 1e-6)

    for i, p in enumerate(valid_points):
        hue = norm_intensities[i] * 120  # 0（红）到120（蓝）
        color_hsv = np.uint8([[[hue, 255, 255]]])
        color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
        pt = (int(p[0]), int(p[1]))
        cv2.circle(proj_img, pt, 2, tuple(map(int, color_bgr)), -1)

    print(f"共 {len(valid_indices)} 个点被成功绘制到图像上。")
    return proj_img

def main():
    print("正在加载标定参数...")
    R, t, K, dist = load_extrinsic_and_intrinsic(CALIB_FILE_PATH)
    if R is None:
        return
    print("标定参数加载成功。")

    if not os.path.exists(OUTPUT_FOLDER_PATH):
        os.makedirs(OUTPUT_FOLDER_PATH)
        print(f"已创建输出文件夹: {OUTPUT_FOLDER_PATH}")

    lidar_files = sorted(
        glob.glob(os.path.join(LIDAR_FOLDER_PATH, "*.pcd")) +
        glob.glob(os.path.join(LIDAR_FOLDER_PATH, "*.txt"))
    )
    image_files = sorted(
        glob.glob(os.path.join(IMAGE_FOLDER_PATH, "*.png")) +
        glob.glob(os.path.join(IMAGE_FOLDER_PATH, "*.jpg")) +
        glob.glob(os.path.join(IMAGE_FOLDER_PATH, "*.jpeg"))
    )

    num_lidar = len(lidar_files)
    num_images = len(image_files)

    if num_lidar == 0 or num_images == 0:
        print("错误: 雷达或图像文件夹为空，无法进行处理。")
        return

    num_pairs = min(num_lidar, num_images)
    if num_lidar != num_images:
        print(f"警告: 雷达文件数量 ({num_lidar}) 与图像文件数量 ({num_images}) 不匹配。")
        print(f"将只处理前 {num_pairs} 对文件。")

    print(f"找到 {num_pairs} 对匹配的雷达和图像文件，开始处理...")

    for i in range(num_pairs):
        lidar_file = lidar_files[i]
        image_file = image_files[i]

        print("-" * 50)
        print(f"正在处理第 {i + 1}/{num_pairs} 对文件:")
        print(f"  雷达: {os.path.basename(lidar_file)}")
        print(f"  图像: {os.path.basename(image_file)}")

        points_lidar, intensities = load_lidar_points(lidar_file)
        image = cv2.imread(image_file)

        if points_lidar is None or image is None:
            print(f"警告: 加载文件失败，已跳过此文件对。")
            continue

        image_points, depths = project_lidar_to_image(points_lidar, R, t, K, dist)
        result_image = draw_projection_on_image(image, image_points, intensities, depths)

        output_filename = f"projection_{i:04d}_{os.path.splitext(os.path.basename(lidar_file))[0]}.png"
        output_filepath = os.path.join(OUTPUT_FOLDER_PATH, output_filename)
        cv2.imwrite(output_filepath, result_image)
        print(f"投影结果已保存至: {output_filepath}")

    print("-" * 50)
    print("所有文件处理完毕！")

if __name__ == "__main__":
    main()
