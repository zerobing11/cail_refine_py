import cv2
import numpy as np
import os
import math

# 设置路径
input_folder = '/home/lz/PycharmProjects/resize_image/evaluate/img'
output_folder = '/home/lz/PycharmProjects/resize_image/evaluate/mask_img'

# 如果输出文件夹不存在则创建
os.makedirs(output_folder, exist_ok=True)

# 计算 Sobel 梯度
def compute_gradient_magnitude(gray):
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y)

# 在半径范围内找最大梯度点
def find_strongest_gradient_point(magnitude, center, radius):
    x, y = center
    h, w = magnitude.shape
    x_min, x_max = max(x - radius, 0), min(x + radius + 1, w)
    y_min, y_max = max(y - radius, 0), min(y + radius + 1, h)
    roi = magnitude[y_min:y_max, x_min:x_max]
    _, _, _, max_loc = cv2.minMaxLoc(roi)
    return (x_min + max_loc[0], y_min + max_loc[1])

# 顺时针排序四角点
def sort_corners_clockwise(pts):
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    return sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

# 获取用户点击的四个点
def get_four_points(image):
    click_points = []
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(click_points) < 4:
            click_points.append((x, y))
            print(f"点击点 {len(click_points)}: ({x}, {y})")

    temp = image.copy()
    cv2.namedWindow("Click 4 corners")
    cv2.setMouseCallback("Click 4 corners", mouse_callback)

    while True:
        show = temp.copy()
        for p in click_points:
            cv2.circle(show, p, 4, (0, 0, 255), -1)
        cv2.imshow("Click 4 corners", show)
        key = cv2.waitKey(1)
        if key == 13 and len(click_points) == 4:  # Enter
            break
        elif key == 27:  # ESC退出
            cv2.destroyAllWindows()
            return None
    cv2.destroyAllWindows()
    return click_points

# 遍历处理文件夹
search_radius = 10
image_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

for idx, filename in enumerate(image_files):
    input_path = os.path.join(input_folder, filename)
    output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + '.png')

    image = cv2.imread(input_path)
    if image is None:
        print(f"无法读取图像: {filename}")
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient_magnitude = compute_gradient_magnitude(gray)

    print(f"\n🖼️ 处理图像：{filename}（{idx+1}/{len(image_files)}）")
    points = get_four_points(image)
    if points is None:
        print("用户中断或未完成点击，跳过此图。")
        continue

    real_points = [find_strongest_gradient_point(gradient_magnitude, p, search_radius) for p in points]
    sorted_corners = sort_corners_clockwise(real_points)

    # 构建掩码图
    mask = np.ones_like(gray, dtype=np.uint8) * 255
    pts = np.array([sorted_corners], dtype=np.int32)
    cv2.fillPoly(mask, pts, 0)

    cv2.imwrite(output_path, mask)
    print(f"✅ 已保存: {output_path}")
