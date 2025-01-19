import math
import numpy as np
import cv2
def compute_homography(src_points, dst_points):
    """
    Tính toán ma trận biến chuyển phối cảnh (homography matrix) từ các điểm tương ứng.
    
    :param src_points: Danh sách các điểm nguồn (trên hình)
    :param dst_points: Danh sách các điểm đích (trên thực tế)
    :return: Ma trận biến chuyển phối cảnh (3x3)
    """
    # Chuyển đổi các điểm thành mảng numpy
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    
    # Tính toán ma trận biến chuyển phối cảnh
    H, status = cv2.findHomography(src_points, dst_points)
    return H
image_pts = [(800, 410), (1125, 410), (1920, 850), (0, 850)]
# M6 is roughly 32 meters wide and 140 meters long there.
world_pts = [(0, 0), (32, 0), (32, 140), (0, 140)] 
H = compute_homography(image_pts, world_pts)
print(H)
# Tính ma trận biến chuyển phối cảnh
