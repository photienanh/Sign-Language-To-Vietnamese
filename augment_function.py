import cv2
import numpy as np
import os
import mediapipe as mp
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import mediapipe as mp
import pandas as pd
import unicodedata
import re
from tqdm import tqdm
import json
import shutil
from datetime import datetime
import logging
import warnings
from scipy.interpolate import interp1d
import random
import math

mp_holistic = mp.solutions.holistic
N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS = 21
N_TOTAL_LANDMARKS = N_UPPER_BODY_POSE_LANDMARKS + N_HAND_LANDMARKS + N_HAND_LANDMARKS

ALL_POSE_CONNECTIONS = list(mp_holistic.POSE_CONNECTIONS)
UPPER_BODY_POSE_CONNECTIONS = []
for connection in ALL_POSE_CONNECTIONS:
    if connection[0] < N_UPPER_BODY_POSE_LANDMARKS and connection[1] < N_UPPER_BODY_POSE_LANDMARKS:
        UPPER_BODY_POSE_CONNECTIONS.append(connection)

# 0.7 đến 1.26
def scale_keypoints_sequence(
    keypoints_sequence,
    scale_factor_range= (0.7, 1.26), # Phạm vi để chọn scale_factor ngẫu nhiên
    num_total_landmarks= N_TOTAL_LANDMARKS,
    num_pose_landmarks_for_center= N_UPPER_BODY_POSE_LANDMARKS - 2,
    normalize_to_01= True # Tham số mới để bật/tắt chuẩn hóa
):
    """
    Áp dụng phép scale 2D cho một chuỗi các frame keypoints.
    Tâm scale cho mỗi frame được tính dựa trên các keypoints được chỉ định trong frame đó.
    Một scale_factor được áp dụng nhất quán cho tất cả các frame trong sequence.
    Sau đó, tùy chọn chuẩn hóa tọa độ x, y của các keypoints hợp lệ về khoảng [0,1].

    Args:
        keypoints_sequence: Danh sách các numpy array, mỗi array là keypoints phẳng hóa của một frame.
                            Chấp nhận các phần tử None.
        scale_factor_range: Tuple (min_factor, max_factor) để chọn ngẫu nhiên scale_factor.
        num_total_landmarks: Tổng số landmarks trong mỗi frame.
        num_pose_landmarks_for_center: Số landmark của pose (từ đầu) để tính tâm scale.
        normalize_to_01: Nếu True, chuẩn hóa tọa độ x, y của các keypoints hợp lệ về [0,1] sau khi scale.

    Returns:
        Một danh sách mới chứa các keypoints đã được xử lý.
    """
    processed_sequence = [] # Đổi tên từ scaled_sequence
    if not keypoints_sequence:
        return processed_sequence

    current_scale_factor = random.uniform(scale_factor_range[0], scale_factor_range[1])

    if current_scale_factor <= 0:
        print("Warning: Scale factor phải dương. Trả về sequence gốc.")
        # Nếu normalize_to_01 là True, chúng ta vẫn có thể muốn chuẩn hóa sequence gốc
        if normalize_to_01:
             temp_sequence = []
             for frame_flat in keypoints_sequence:
                if frame_flat is None:
                    temp_sequence.append(None)
                    continue
                try:
                    points_to_norm = frame_flat.copy().reshape(num_total_landmarks, 3)
                     # Áp dụng chuẩn hóa cho frame này
                    valid_xy_mask_norm = np.any(points_to_norm[:, :2] != 0, axis=1)
                    if np.any(valid_xy_mask_norm):
                        x_coords = points_to_norm[valid_xy_mask_norm, 0]
                        y_coords = points_to_norm[valid_xy_mask_norm, 1]

                        min_x, max_x = np.min(x_coords), np.max(x_coords)
                        min_y, max_y = np.min(y_coords), np.max(y_coords)

                        if (max_x - min_x) > 1e-7:
                            points_to_norm[valid_xy_mask_norm, 0] = (x_coords - min_x) / (max_x - min_x)
                        elif x_coords.size > 0:
                            points_to_norm[valid_xy_mask_norm, 0] = 0.5

                        if (max_y - min_y) > 1e-7:
                            points_to_norm[valid_xy_mask_norm, 1] = (y_coords - min_y) / (max_y - min_y)
                        elif y_coords.size > 0:
                            points_to_norm[valid_xy_mask_norm, 1] = 0.5
                    temp_sequence.append(points_to_norm.flatten())
                except Exception:
                    temp_sequence.append(frame_flat.copy()) # Giữ nguyên nếu có lỗi reshape
             return temp_sequence
        return [kp.copy() if isinstance(kp, np.ndarray) else kp for kp in keypoints_sequence]

    for frame_keypoints_flat in keypoints_sequence:
        if frame_keypoints_flat is None:
            processed_sequence.append(None)
            continue

        if not isinstance(frame_keypoints_flat, np.ndarray) or \
           frame_keypoints_flat.shape != (num_total_landmarks * 3,):
            processed_sequence.append(frame_keypoints_flat.copy())
            continue

        try:
            # current_points_3d sẽ chứa các điểm sau khi scale (hoặc gốc nếu không scale)
            current_points_3d = frame_keypoints_flat.copy().reshape(num_total_landmarks, 3)
        except ValueError:
            processed_sequence.append(frame_keypoints_flat.copy())
            continue

        # --- Tính toán tâm scale (như cũ) ---
        pose_points = current_points_3d[0:N_UPPER_BODY_POSE_LANDMARKS]
        hand_points_start_index = N_UPPER_BODY_POSE_LANDMARKS
        other_points_for_center = current_points_3d[hand_points_start_index:]
        points_for_center_pose_part = pose_points[0:num_pose_landmarks_for_center]
        points_to_calculate_center_list = [points_for_center_pose_part]
        if other_points_for_center.shape[0] > 0:
            points_to_calculate_center_list.append(other_points_for_center)
        
        center_x, center_y = 0.0, 0.0
        can_calculate_center = False

        if points_to_calculate_center_list: # Đảm bảo list không rỗng
            points_to_calculate_center_concat = np.concatenate(points_to_calculate_center_list, axis=0)
            # Chỉ sử dụng các điểm có ít nhất một tọa độ khác 0 để tính tâm
            valid_center_points_mask = np.any(points_to_calculate_center_concat != 0, axis=1)
            valid_center_points = points_to_calculate_center_concat[valid_center_points_mask]

            if valid_center_points.shape[0] > 0:
                center_x = np.median(valid_center_points[:, 0])
                center_y = np.median(valid_center_points[:, 1])
                can_calculate_center = True
            else: # Fallback nếu không có điểm hợp lệ trong selection
                all_valid_points_mask = np.any(current_points_3d != 0, axis=1)
                all_valid_points = current_points_3d[all_valid_points_mask]
                if all_valid_points.shape[0] > 0:
                    center_x = np.median(all_valid_points[:, 0])
                    center_y = np.median(all_valid_points[:, 1])
                    can_calculate_center = True
        # --- Kết thúc tính toán tâm scale ---


        # --- Áp dụng phép Scale ---
        if can_calculate_center:
            # Mask này xác định những điểm sẽ được scale (ban đầu khác 0,0,0)
            all_valid_points_mask_for_scaling = np.any(current_points_3d != 0, axis=1)
            
            if np.any(all_valid_points_mask_for_scaling): # Chỉ scale nếu có điểm hợp lệ
                x_all_valid = current_points_3d[all_valid_points_mask_for_scaling, 0]
                y_all_valid = current_points_3d[all_valid_points_mask_for_scaling, 1]

                x_trans = x_all_valid - center_x
                y_trans = y_all_valid - center_y
                x_scaled = x_trans * current_scale_factor
                y_scaled = y_trans * current_scale_factor
                new_x_all_valid = x_scaled + center_x
                new_y_all_valid = y_scaled + center_y

                current_points_3d[all_valid_points_mask_for_scaling, 0] = new_x_all_valid
                current_points_3d[all_valid_points_mask_for_scaling, 1] = new_y_all_valid
        # Nếu không thể tính tâm (can_calculate_center is False), current_points_3d giữ nguyên giá trị gốc.
        # --- Kết thúc phép Scale ---

        # --- Chuẩn hóa về [0,1] nếu được yêu cầu ---
        if normalize_to_01:
            # Mask cho các điểm có tọa độ x HOẶC y khác không. Z không ảnh hưởng.
            # (0,0,Z) sẽ không được dùng để tính min/max cho X,Y và X,Y của nó vẫn là 0.
            valid_xy_mask_norm = np.any(current_points_3d[:, :2] != 0, axis=1)

            if np.any(valid_xy_mask_norm): # Chỉ chuẩn hóa nếu có điểm có x hoặc y khác 0
                x_coords = current_points_3d[valid_xy_mask_norm, 0]
                y_coords = current_points_3d[valid_xy_mask_norm, 1]

                min_x, max_x = np.min(x_coords), np.max(x_coords)
                min_y, max_y = np.min(y_coords), np.max(y_coords)

                # Chuẩn hóa X
                # Gán cho các điểm trong valid_xy_mask_norm
                if (max_x - min_x) > 1e-7: # epsilon để tránh chia cho 0 do lỗi float
                    current_points_3d[valid_xy_mask_norm, 0] = (x_coords - min_x) / (max_x - min_x)
                elif x_coords.size > 0: # Nếu tất cả x_coords giống nhau (và có điểm)
                    current_points_3d[valid_xy_mask_norm, 0] = 0.5 # Hoặc 0.0, tùy bạn chọn

                # Chuẩn hóa Y
                if (max_y - min_y) > 1e-7:
                    current_points_3d[valid_xy_mask_norm, 1] = (y_coords - min_y) / (max_y - min_y)
                elif y_coords.size > 0: # Nếu tất cả y_coords giống nhau (và có điểm)
                    current_points_3d[valid_xy_mask_norm, 1] = 0.5
            # Nếu không có điểm nào có x hoặc y khác 0 (ví dụ: frame toàn (0,0,z) hoặc (0,0,0)),
            # thì current_points_3d giữ nguyên, tọa độ x, y vẫn là 0.
        # --- Kết thúc chuẩn hóa ---

        processed_frame_flat_output = current_points_3d.flatten()

        if np.isnan(processed_frame_flat_output).any() or np.isinf(processed_frame_flat_output).any():
            processed_sequence.append(frame_keypoints_flat.copy()) # Giữ nguyên frame gốc nếu có lỗi NaN/Inf
        else:
            processed_sequence.append(processed_frame_flat_output)

    return processed_sequence


def rotate_keypoints_sequence(
    keypoints_sequence,
    angle_degrees_range = (-15.0, 15.0), # Phạm vi góc xoay (độ)
 # Hoặc cung cấp một góc cố định
    num_total_landmarks = N_TOTAL_LANDMARKS,
    num_pose_landmarks_for_center= N_UPPER_BODY_POSE_LANDMARKS - 2 # Điểm để tính tâm xoay
):
    """
    Áp dụng phép xoay 2D cho một chuỗi các frame keypoints.
    Tâm xoay cho mỗi frame được tính dựa trên các keypoints được chỉ định trong frame đó.
    Một góc xoay (ngẫu nhiên hoặc cố định) được áp dụng nhất quán cho tất cả các frame trong sequence.

    Args:
        keypoints_sequence: Danh sách các numpy array (keypoints phẳng hóa của frame).
        angle_degrees_range: Tuple (min_angle, max_angle) bằng độ.
        fixed_angle_degrees: Nếu được cung cấp, sử dụng góc này (bằng độ).
        num_total_landmarks: Tổng số landmarks trong mỗi frame.
        num_pose_landmarks_for_center: Số landmark của pose (từ đầu) để tính tâm xoay.

    Returns:
        Một danh sách mới chứa các keypoints đã được xoay.
    """
    rotated_sequence = []
    if not keypoints_sequence:
        return rotated_sequence


    angle_deg = random.uniform(angle_degrees_range[0], angle_degrees_range[1])

    angle_rad = math.radians(angle_deg) # Chuyển sang radian để tính sin/cos
    cos_angle = math.cos(angle_rad)
    sin_angle = math.sin(angle_rad)

    for frame_keypoints_flat in keypoints_sequence:
        if frame_keypoints_flat is None:
            rotated_sequence.append(None)
            continue

        if not isinstance(frame_keypoints_flat, np.ndarray) or \
           frame_keypoints_flat.shape != (num_total_landmarks * 3,):
            rotated_sequence.append(frame_keypoints_flat.copy())
            continue

        try:
            all_points = frame_keypoints_flat.copy().reshape(num_total_landmarks, 3)
        except ValueError:
            rotated_sequence.append(frame_keypoints_flat.copy())
            continue

        # --- Tính toán tâm xoay (tương tự như hàm scale) ---
        pose_points = all_points[0:N_UPPER_BODY_POSE_LANDMARKS]
        hand_points_start_index = N_UPPER_BODY_POSE_LANDMARKS
        other_points_for_center = all_points[hand_points_start_index:]
        points_for_center_pose_part = pose_points[0:num_pose_landmarks_for_center]
        points_to_calculate_center_list = [points_for_center_pose_part]
        if other_points_for_center.shape[0] > 0:
            points_to_calculate_center_list.append(other_points_for_center)

        if not points_to_calculate_center_list: # Nếu list rỗng
             rotated_sequence.append(frame_keypoints_flat.copy())
             continue

        points_to_calculate_center = np.concatenate(points_to_calculate_center_list, axis=0)
        valid_center_points_mask = np.any(points_to_calculate_center != 0, axis=1)
        valid_center_points = points_to_calculate_center[valid_center_points_mask]

        center_x, center_y = 0.0, 0.0
        can_calculate_center = False

        if valid_center_points.shape[0] > 0:
            center_x = np.median(valid_center_points[:, 0])
            center_y = np.median(valid_center_points[:, 1])
            can_calculate_center = True
        else:
            all_valid_points_mask = np.any(all_points != 0, axis=1)
            all_valid_points = all_points[all_valid_points_mask]
            if all_valid_points.shape[0] > 0:
                center_x = np.median(all_valid_points[:, 0])
                center_y = np.median(all_valid_points[:, 1])
                can_calculate_center = True

        if not can_calculate_center:
            rotated_sequence.append(frame_keypoints_flat.copy())
            continue
        # --- Kết thúc tính toán tâm xoay ---

        rotated_all_points = all_points.copy()
        # Chỉ xoay các điểm hợp lệ (khác 0,0,0)
        all_valid_points_mask_for_rotation = np.any(all_points != 0, axis=1)

        # Lấy tọa độ x, y của các điểm hợp lệ cần xoay
        x_original_valid = all_points[all_valid_points_mask_for_rotation, 0]
        y_original_valid = all_points[all_valid_points_mask_for_rotation, 1]

        # 1. Dịch chuyển về gốc (tâm xoay làm gốc)
        x_translated = x_original_valid - center_x
        y_translated = y_original_valid - center_y

        # 2. Áp dụng phép xoay
        x_rotated = x_translated * cos_angle - y_translated * sin_angle
        y_rotated = x_translated * sin_angle + y_translated * cos_angle

        # 3. Dịch chuyển trở lại
        new_x_all_valid = x_rotated + center_x
        new_y_all_valid = y_rotated + center_y

        # Cập nhật tọa độ đã xoay vào mảng
        rotated_all_points[all_valid_points_mask_for_rotation, 0] = new_x_all_valid
        rotated_all_points[all_valid_points_mask_for_rotation, 1] = new_y_all_valid
        # Tọa độ z được giữ nguyên

        rotated_frame_flat_output = rotated_all_points.flatten()

        if np.isnan(rotated_frame_flat_output).any() or np.isinf(rotated_frame_flat_output).any():
            rotated_sequence.append(frame_keypoints_flat.copy())
        else:
            rotated_sequence.append(rotated_frame_flat_output)

    return rotated_sequence

def translate_keypoints_sequence(
    keypoints_sequence,
    translate_x_range = (-0.05, 0.05), # Dịch chuyển tối đa 5% chiều rộng
    translate_y_range = (-0.05, 0.05), # Dịch chuyển tối đa 5% chiều cao
    clip_to_01: bool = True, # Cắt giá trị keypoints về khoảng [0,1] sau khi dịch chuyển
    num_total_landmarks= N_TOTAL_LANDMARKS,
):
    """
    Áp dụng phép dịch chuyển 2D (translation) cho một chuỗi các frame keypoints.
    Một vector dịch chuyển (dx, dy) ngẫu nhiên hoặc cố định được áp dụng nhất quán
    cho tất cả các keypoints hợp lệ trong tất cả các frame của sequence.

    Args:
        keypoints_sequence: Danh sách các numpy array (keypoints phẳng hóa của frame).
        translate_x_range: Tuple (min_dx, max_dx) cho dịch chuyển theo trục x.
        translate_y_range: Tuple (min_dy, max_dy) cho dịch chuyển theo trục y.
        fixed_translation: Nếu được cung cấp, sử dụng (dx, dy) này.
        clip_to_01: Nếu True, cắt các tọa độ x, y về khoảng [0,1] sau khi dịch chuyển.
        num_total_landmarks: Tổng số landmarks trong mỗi frame.

    Returns:
        Một danh sách mới chứa các keypoints đã được dịch chuyển.
    """
    translated_sequence = []
    if not keypoints_sequence:
        return translated_sequence


    dx = random.uniform(translate_x_range[0], translate_x_range[1])
    dy = random.uniform(translate_y_range[0], translate_y_range[1])

    for frame_keypoints_flat in keypoints_sequence:
        if frame_keypoints_flat is None:
            translated_sequence.append(None)
            continue

        if not isinstance(frame_keypoints_flat, np.ndarray) or \
           frame_keypoints_flat.shape != (num_total_landmarks * 3,):
            translated_sequence.append(frame_keypoints_flat.copy())
            continue

        try:
            all_points = frame_keypoints_flat.copy().reshape(num_total_landmarks, 3)
        except ValueError:
            translated_sequence.append(frame_keypoints_flat.copy())
            continue

        translated_all_points = all_points.copy()

        # Chỉ dịch chuyển các điểm hợp lệ (khác 0,0,0)
        # Các điểm (0,0,0) thường đại diện cho keypoints không được phát hiện và nên giữ nguyên.
        valid_points_mask = np.any(all_points != 0, axis=1)

        # Áp dụng dịch chuyển
        translated_all_points[valid_points_mask, 0] += dx  # x_new = x_old + dx
        translated_all_points[valid_points_mask, 1] += dy  # y_new = y_old + dy
        # Tọa độ z được giữ nguyên

        # Cắt giá trị về khoảng [0,1] nếu cần
        if clip_to_01:
            # Chỉ clip các điểm đã được dịch chuyển (valid_points_mask)
            # và chỉ clip tọa độ x, y
            translated_all_points[valid_points_mask, 0] = np.clip(translated_all_points[valid_points_mask, 0], 0.0, 1.0)
            translated_all_points[valid_points_mask, 1] = np.clip(translated_all_points[valid_points_mask, 1], 0.0, 1.0)

        translated_frame_flat_output = translated_all_points.flatten()

        if np.isnan(translated_frame_flat_output).any() or np.isinf(translated_frame_flat_output).any():
            translated_sequence.append(frame_keypoints_flat.copy())
        else:
            translated_sequence.append(translated_frame_flat_output)

    return translated_sequence

def time_stretch_keypoints_sequence(
    keypoints_sequence,
    speed_factor_range= (0.8, 1.2), # 0.8 = chậm hơn 20%, 1.2 = nhanh hơn 20%
):
    """
    Thay đổi tốc độ của một chuỗi keypoints bằng cách lấy mẫu lại các frame.
    Một speed_factor ngẫu nhiên hoặc cố định được áp dụng.

    Args:
        keypoints_sequence: Danh sách các numpy array (keypoints phẳng hóa của frame).
        speed_factor_range: Tuple (min_factor, max_factor). Factor < 1 làm chậm, > 1 làm nhanh.
        fixed_speed_factor: Nếu được cung cấp, sử dụng giá trị này.

    Returns:
        Một danh sách mới chứa các keypoints đã được thay đổi tốc độ.
        Độ dài của sequence trả về có thể khác với sequence đầu vào.
    """
    perturbed_sequence = []
    if not keypoints_sequence or all(kp is None for kp in keypoints_sequence):
        return keypoints_sequence # Trả về rỗng hoặc list of Nones

    # Lọc ra các frame không phải None để xử lý
    valid_frames = [kp for kp in keypoints_sequence if kp is not None]
    if not valid_frames: # Nếu tất cả các frame đều là None
        return keypoints_sequence

    original_num_valid_frames = len(valid_frames)


    current_speed_factor = random.uniform(speed_factor_range[0], speed_factor_range[1])

    if current_speed_factor <= 0:
        print("Warning: Speed factor phải dương. Trả về sequence gốc.")
        return [kp.copy() if isinstance(kp, np.ndarray) else kp for kp in keypoints_sequence]
    if current_speed_factor == 1.0: # Không thay đổi gì
        return [kp.copy() if isinstance(kp, np.ndarray) else kp for kp in keypoints_sequence]


    # Tính toán số frame mới dựa trên speed_factor
    # Nếu speed_factor > 1 (nhanh hơn), num_new_frames < original_num_valid_frames
    # Nếu speed_factor < 1 (chậm hơn), num_new_frames > original_num_valid_frames
    num_new_frames = int(round(original_num_valid_frames / current_speed_factor))

    if num_new_frames == 0: # Tránh trường hợp num_new_frames = 0
        if original_num_valid_frames > 0:
            perturbed_sequence.append(valid_frames[0].copy() if valid_frames[0] is not None else None)
        return perturbed_sequence # Trả về 1 frame nếu có, hoặc rỗng

    # Tạo ra các index để lấy mẫu từ valid_frames
    # np.linspace sẽ tạo ra num_new_frames điểm cách đều nhau từ 0 đến original_num_valid_frames - 1
    original_indices = np.linspace(0, original_num_valid_frames - 1, num_new_frames)
    # Làm tròn và chuyển thành int để dùng làm index
    resampled_indices = np.round(original_indices).astype(int)
    # Đảm bảo index không vượt quá giới hạn (mặc dù linspace nên xử lý điều này)
    resampled_indices = np.clip(resampled_indices, 0, original_num_valid_frames - 1)


    for res_idx in resampled_indices:
        perturbed_sequence.append(valid_frames[res_idx].copy())

    # Xử lý trường hợp keypoints_sequence ban đầu có các frame None xen kẽ:
    # Cách đơn giản nhất là chỉ trả về sequence đã resample từ các frame hợp lệ.
    # Nếu cần giữ lại cấu trúc None, sẽ phức tạp hơn nhiều.
    # Hiện tại, hàm này sẽ trả về một sequence chỉ chứa các frame đã được resample,
    # và các frame None ở đầu/cuối của input sequence gốc sẽ bị mất.
    # Nếu input chỉ toàn None thì đã return ở trên.

    return perturbed_sequence

POSE_LM_LEFT_SHOULDER = 11
POSE_LM_RIGHT_SHOULDER = 12
POSE_LM_LEFT_ELBOW = 13
POSE_LM_RIGHT_ELBOW = 14
POSE_LM_LEFT_WRIST = 15
POSE_LM_RIGHT_WRIST = 16

# --- HÀM GIẢI IK 2D (PHIÊN BẢN CẢI TIẾN) ---
def solve_2_link_ik_2d_v2(p_shoulder_xy: np.ndarray,
                          p_wrist_target_xy: np.ndarray,
                          len_upper_arm: float,
                          len_forearm: float,
                          original_elbow_xy= None,
                          original_wrist_xy= None,
                          prefer_original_bend: bool = True):
    """
    Giải bài toán IK 2D cho một cánh tay 2 đoạn, cố gắng giữ hướng gập khuỷu tay.
    p_shoulder_xy: (x,y) của vai (cố định)
    p_wrist_target_xy: (x,y) của cổ tay mục tiêu
    len_upper_arm: chiều dài đoạn vai-khuỷu
    len_forearm: chiều dài đoạn khuỷu-cổ tay
    original_elbow_xy: tọa độ XY của khuỷu tay gốc.
    original_wrist_xy: tọa độ XY của cổ tay gốc.
    prefer_original_bend: Nếu True, cố gắng giữ hướng gập ban đầu.
    """
    d = np.linalg.norm(p_wrist_target_xy - p_shoulder_xy)
    l1 = max(1e-5, len_upper_arm) # Tránh chiều dài bằng 0
    l2 = max(1e-5, len_forearm)  # Tránh chiều dài bằng 0

    # Trường hợp 1: Cổ tay mục tiêu quá xa (không thể với tới)
    if d > l1 + l2 - 1e-5: # Trừ epsilon nhỏ để ổn định
        # Duỗi thẳng cánh tay dọc theo đường từ vai đến cổ tay mục tiêu
        if d < 1e-9: # Vai và cổ tay mục tiêu trùng, không thể xác định hướng
             return p_shoulder_xy + np.array([l1, 0]) if original_elbow_xy is None else original_elbow_xy.copy()
        vec_sw = (p_wrist_target_xy - p_shoulder_xy) / d
        return p_shoulder_xy + vec_sw * l1

    # Trường hợp 2: Cổ tay mục tiêu quá gần vai
    if d < abs(l1 - l2) + 1e-5: # Cộng epsilon nhỏ
        # print(f"IK Warn: Target too close. d={d:.3f}, l1={l1:.3f}, l2={l2:.3f}")
        # Cánh tay gập lại, khuỷu tay nằm trên đường thẳng nối dài hoặc co ngắn.
        # Để đơn giản, nếu có original_elbow, trả về nó, nếu không, duỗi ra xa nhất có thể
        if original_elbow_xy is not None:
            return original_elbow_xy.copy()
        if d < 1e-9: # Vai và cổ tay mục tiêu trùng
            return p_shoulder_xy + np.array([l1, 0]) # Đặt khuỷu tay theo hướng x
        # Duỗi theo hướng cổ tay mục tiêu (mặc dù gần)
        vec_sw = (p_wrist_target_xy - p_shoulder_xy) / d
        # Nếu l1 > l2 + d (khuỷu tay sẽ ở phía sau vai theo đường SW)
        # Nếu l2 > l1 + d (không thể, vì d > |l1-l2|)
        # Đặt khuỷu tay trên đường SW, cách vai l1
        return p_shoulder_xy + vec_sw * l1


    # Trường hợp 3: Giải bình thường (cổ tay trong tầm với)
    # Sử dụng định lý cos
    # a: khoảng cách từ vai đến điểm chiếu của khuỷu tay lên đoạn vai-cổ tay mục tiêu (SW_target)
    # Đảm bảo d không quá nhỏ
    if d < 1e-9: d = 1e-9 # Tránh chia cho 0

    a = (l1**2 - l2**2 + d**2) / (2 * d)
    h_squared = l1**2 - a**2

    if h_squared < -1e-9: # Âm đáng kể, có lỗi
        # print(f"IK Critical Error: h_squared is significantly negative ({h_squared:.4g}). d={d:.3f}, l1={l1:.3f}, l2={l2:.3f}, a={a:.3f}")
        # Fallback: duỗi thẳng như trường hợp không với tới
        vec_sw = (p_wrist_target_xy - p_shoulder_xy) / d
        return p_shoulder_xy + vec_sw * l1
    h = np.sqrt(max(0, h_squared)) # Đảm bảo không lấy căn của số âm do lỗi làm tròn

    # P2 là điểm trên đoạn SW_target, cách S một đoạn 'a'
    p2_x = p_shoulder_xy[0] + a * (p_wrist_target_xy[0] - p_shoulder_xy[0]) / d
    p2_y = p_shoulder_xy[1] + a * (p_wrist_target_xy[1] - p_shoulder_xy[1]) / d

    # Hai nghiệm cho vị trí khuỷu tay (E_sol1, E_sol2)
    # Vector đơn vị vuông góc với SW_target
    perp_vec_x = -(p_wrist_target_xy[1] - p_shoulder_xy[1]) / d
    perp_vec_y =  (p_wrist_target_xy[0] - p_shoulder_xy[0]) / d

    elbow_sol1_xy = np.array([p2_x + h * perp_vec_x, p2_y + h * perp_vec_y])
    elbow_sol2_xy = np.array([p2_x - h * perp_vec_x, p2_y - h * perp_vec_y])

    # Lựa chọn nghiệm
    if not prefer_original_bend or original_elbow_xy is None or original_wrist_xy is None:
        # Nếu không ưu tiên hướng gập cũ, hoặc không có thông tin cũ,
        # chọn nghiệm gần với khuỷu tay cũ hơn (nếu có), hoặc nghiệm 1 mặc định.
        if original_elbow_xy is not None:
            dist1 = np.linalg.norm(elbow_sol1_xy - original_elbow_xy)
            dist2 = np.linalg.norm(elbow_sol2_xy - original_elbow_xy)
            return elbow_sol1_xy if dist1 <= dist2 else elbow_sol2_xy
        return elbow_sol1_xy # Mặc định chọn nghiệm 1

    # Ưu tiên giữ hướng gập ban đầu
    # Tính "side" của original_elbow_xy so với đường S_orig -> W_orig
    # side = (Wx - Sx)*(Ey - Sy) - (Wy - Sy)*(Ex - Sx)
    # Nếu S_orig, W_orig quá gần nhau, không thể xác định side
    vec_sw_orig = original_wrist_xy - p_shoulder_xy
    if np.linalg.norm(vec_sw_orig) < 1e-5: # Vai và cổ tay gốc trùng nhau
        # Khó xác định hướng gập, chọn nghiệm gần nhất
        dist1 = np.linalg.norm(elbow_sol1_xy - original_elbow_xy)
        dist2 = np.linalg.norm(elbow_sol2_xy - original_elbow_xy)
        return elbow_sol1_xy if dist1 <= dist2 else elbow_sol2_xy

    original_side = (original_wrist_xy[0] - p_shoulder_xy[0]) * (original_elbow_xy[1] - p_shoulder_xy[1]) - \
                    (original_wrist_xy[1] - p_shoulder_xy[1]) * (original_elbow_xy[0] - p_shoulder_xy[0])

    # Tính "side" cho hai nghiệm mới so với đường S_orig -> W_target
    side1 = (p_wrist_target_xy[0] - p_shoulder_xy[0]) * (elbow_sol1_xy[1] - p_shoulder_xy[1]) - \
            (p_wrist_target_xy[1] - p_shoulder_xy[1]) * (elbow_sol1_xy[0] - p_shoulder_xy[0])
    side2 = (p_wrist_target_xy[0] - p_shoulder_xy[0]) * (elbow_sol2_xy[1] - p_shoulder_xy[1]) - \
            (p_wrist_target_xy[1] - p_shoulder_xy[1]) * (elbow_sol2_xy[0] - p_shoulder_xy[0])

    # Chọn nghiệm có "side" cùng dấu với original_side
    # (Nếu original_side gần 0 - cánh tay thẳng - thì ưu tiên nghiệm làm h không quá nhỏ, hoặc nghiệm gần hơn)
    if abs(original_side) < 1e-3: # Cánh tay gốc gần như thẳng
        # Ưu tiên nghiệm có |h| lớn hơn một chút (tránh gập hoàn toàn nếu không cần thiết)
        # Hoặc chọn nghiệm gần nhất với vị trí khuỷu tay cũ
        dist1 = np.linalg.norm(elbow_sol1_xy - original_elbow_xy)
        dist2 = np.linalg.norm(elbow_sol2_xy - original_elbow_xy)
        return elbow_sol1_xy if dist1 <= dist2 else elbow_sol2_xy

    if np.sign(side1) == np.sign(original_side):
        return elbow_sol1_xy
    elif np.sign(side2) == np.sign(original_side):
        return elbow_sol2_xy
    else:
        # Nếu không có nghiệm nào cùng hướng (hiếm), chọn nghiệm gần nhất với khuỷu tay cũ
        dist1 = np.linalg.norm(elbow_sol1_xy - original_elbow_xy)
        dist2 = np.linalg.norm(elbow_sol2_xy - original_elbow_xy)
        # print("IK Warn: No solution matched original bend side. Choosing closest.")
        return elbow_sol1_xy if dist1 <= dist2 else elbow_sol2_xy


# --- HÀM TĂNG CƯỜNG CHÍNH ---
def inter_hand_distance(
    keypoints_sequence,
    total_dx_change_range= (-0.1, 0.1),
    overall_dy_shift_range= (-0.03, 0.03),
    clip_to_01: bool = True,
    num_total_landmarks: int = N_TOTAL_LANDMARKS,
):
    augmented_sequence = []
    if not keypoints_sequence: return augmented_sequence

    current_total_dx_change = random.uniform(total_dx_change_range[0], total_dx_change_range[1])

    current_overall_dy_shift = random.uniform(overall_dy_shift_range[0], overall_dy_shift_range[1])

    for frame_keypoints_flat in keypoints_sequence:
        if frame_keypoints_flat is None:
            augmented_sequence.append(None)
            continue
        if not isinstance(frame_keypoints_flat, np.ndarray) or \
           frame_keypoints_flat.shape != (num_total_landmarks * 3,):
            augmented_sequence.append(frame_keypoints_flat.copy())
            continue
        try:
            all_points_orig = frame_keypoints_flat.copy().reshape(num_total_landmarks, 3)
        except ValueError:
            augmented_sequence.append(frame_keypoints_flat.copy())
            continue

        augmented_points = all_points_orig.copy()

        s_l_orig_xy = all_points_orig[POSE_LM_LEFT_SHOULDER, 0:2].copy()
        e_l_orig_xy = all_points_orig[POSE_LM_LEFT_ELBOW, 0:2].copy()
        w_l_orig_xy = all_points_orig[POSE_LM_LEFT_WRIST, 0:2].copy()

        s_r_orig_xy = all_points_orig[POSE_LM_RIGHT_SHOULDER, 0:2].copy()
        e_r_orig_xy = all_points_orig[POSE_LM_RIGHT_ELBOW, 0:2].copy()
        w_r_orig_xy = all_points_orig[POSE_LM_RIGHT_WRIST, 0:2].copy()

        left_arm_key_points_valid = np.all(s_l_orig_xy!=0) and np.all(e_l_orig_xy!=0) and np.all(w_l_orig_xy!=0)
        right_arm_key_points_valid = np.all(s_r_orig_xy!=0) and np.all(e_r_orig_xy!=0) and np.all(w_r_orig_xy!=0)

        # Nếu một trong các điểm chính của cánh tay không hợp lệ, không thực hiện IK cho cánh tay đó
        # nhưng vẫn có thể dịch chuyển tay kia nếu nó hợp lệ.
        # Hoặc, bỏ qua toàn bộ frame nếu một cánh tay không hợp lệ. Để đơn giản, ta sẽ xử lý từng tay.

        # --- Tính X mục tiêu cho cổ tay ---
        # Chỉ tính nếu cả hai cổ tay ban đầu hợp lệ
        if np.all(w_l_orig_xy != 0) and np.all(w_r_orig_xy != 0):
            current_mid_wrists_x = (w_l_orig_xy[0] + w_r_orig_xy[0]) / 2
            # Đảm bảo tay trái luôn ở bên trái tay phải ban đầu để tính current_dist
            x_left = min(w_l_orig_xy[0], w_r_orig_xy[0])
            x_right = max(w_l_orig_xy[0], w_r_orig_xy[0])
            current_dist_wrists_x = x_right - x_left

            target_dist_wrists_x = current_dist_wrists_x + current_total_dx_change
            if target_dist_wrists_x < 0.01: target_dist_wrists_x = 0.01 # Giữ một khoảng cách tối thiểu nhỏ

            # Giữ nguyên thứ tự trái phải ban đầu
            if w_l_orig_xy[0] <= w_r_orig_xy[0]:
                w_l_target_x = current_mid_wrists_x - target_dist_wrists_x / 2
                w_r_target_x = current_mid_wrists_x + target_dist_wrists_x / 2
            else: # Tay phải ban đầu ở bên trái tay trái (hiếm)
                w_r_target_x = current_mid_wrists_x - target_dist_wrists_x / 2
                w_l_target_x = current_mid_wrists_x + target_dist_wrists_x / 2
        else: # Nếu một trong hai cổ tay không hợp lệ, không thay đổi khoảng cách ngang
            w_l_target_x = w_l_orig_xy[0]
            w_r_target_x = w_r_orig_xy[0]


        # --- Xử lý tay trái ---
        if left_arm_key_points_valid:
            len_l_upper = np.linalg.norm(e_l_orig_xy - s_l_orig_xy)
            len_l_forearm = np.linalg.norm(w_l_orig_xy - e_l_orig_xy)
            # Giữ Y của cổ tay không đổi trong bước IK, sau đó mới áp dụng overall_dy_shift
            w_l_target_xy_for_ik = np.array([w_l_target_x, w_l_orig_xy[1]])

            e_l_new_xy = solve_2_link_ik_2d_v2(s_l_orig_xy, w_l_target_xy_for_ik, len_l_upper, len_l_forearm, e_l_orig_xy, w_l_orig_xy)

            if e_l_new_xy is not None:
                dx_wrist_l = w_l_target_xy_for_ik[0] - w_l_orig_xy[0]
                dy_wrist_l = w_l_target_xy_for_ik[1] - w_l_orig_xy[1] # Sẽ là 0 ở bước này

                augmented_points[POSE_LM_LEFT_ELBOW, 0:2] = e_l_new_xy
                augmented_points[POSE_LM_LEFT_WRIST, 0:2] = w_l_target_xy_for_ik

                idx_lh_start = N_UPPER_BODY_POSE_LANDMARKS
                idx_lh_end = idx_lh_start + N_HAND_LANDMARKS
                left_hand_kps_part = augmented_points[idx_lh_start:idx_lh_end]
                left_hand_valid_mask = np.any(left_hand_kps_part != 0, axis=1)
                if np.any(left_hand_valid_mask):
                    left_hand_kps_part[left_hand_valid_mask, 0] += dx_wrist_l
                    left_hand_kps_part[left_hand_valid_mask, 1] += dy_wrist_l
                augmented_points[idx_lh_start:idx_lh_end] = left_hand_kps_part

        # --- Xử lý tay phải ---
        if right_arm_key_points_valid:
            len_r_upper = np.linalg.norm(e_r_orig_xy - s_r_orig_xy)
            len_r_forearm = np.linalg.norm(w_r_orig_xy - e_r_orig_xy)
            w_r_target_xy_for_ik = np.array([w_r_target_x, w_r_orig_xy[1]])

            e_r_new_xy = solve_2_link_ik_2d_v2(s_r_orig_xy, w_r_target_xy_for_ik, len_r_upper, len_r_forearm, e_r_orig_xy, w_r_orig_xy)

            if e_r_new_xy is not None:
                dx_wrist_r = w_r_target_xy_for_ik[0] - w_r_orig_xy[0]
                dy_wrist_r = w_r_target_xy_for_ik[1] - w_r_orig_xy[1]

                augmented_points[POSE_LM_RIGHT_ELBOW, 0:2] = e_r_new_xy
                augmented_points[POSE_LM_RIGHT_WRIST, 0:2] = w_r_target_xy_for_ik

                idx_rh_start = N_UPPER_BODY_POSE_LANDMARKS + N_HAND_LANDMARKS
                idx_rh_end = idx_rh_start + N_HAND_LANDMARKS
                right_hand_kps_part = augmented_points[idx_rh_start:idx_rh_end]
                right_hand_valid_mask = np.any(right_hand_kps_part != 0, axis=1)
                if np.any(right_hand_valid_mask):
                    right_hand_kps_part[right_hand_valid_mask, 0] += dx_wrist_r
                    right_hand_kps_part[right_hand_valid_mask, 1] += dy_wrist_r
                augmented_points[idx_rh_start:idx_rh_end] = right_hand_kps_part

        # --- Áp dụng dịch chuyển Y chung (overall_dy_shift) cho cánh tay và bàn tay ---
        if abs(current_overall_dy_shift) > 1e-5:
            arm_and_hand_indices = [
                POSE_LM_LEFT_WRIST, POSE_LM_LEFT_ELBOW,
                POSE_LM_RIGHT_WRIST, POSE_LM_RIGHT_ELBOW
            ]
            arm_and_hand_indices.extend(list(range(N_UPPER_BODY_POSE_LANDMARKS, num_total_landmarks)))
            unique_arm_hand_indices = sorted(list(set(arm_and_hand_indices)))

            for idx in unique_arm_hand_indices:
                # Chỉ dịch chuyển nếu điểm đó thuộc một cánh tay hợp lệ đã được xử lý IK (hoặc điểm tay)
                is_left_arm_part = (idx == POSE_LM_LEFT_WRIST or idx == POSE_LM_LEFT_ELBOW)
                is_right_arm_part = (idx == POSE_LM_RIGHT_WRIST or idx == POSE_LM_RIGHT_ELBOW)
                is_left_hand_part = (N_UPPER_BODY_POSE_LANDMARKS <= idx < N_UPPER_BODY_POSE_LANDMARKS + N_HAND_LANDMARKS)
                is_right_hand_part = (N_UPPER_BODY_POSE_LANDMARKS + N_HAND_LANDMARKS <= idx < num_total_landmarks)

                should_shift_y = (is_left_arm_part and left_arm_key_points_valid) or \
                                 (is_right_arm_part and right_arm_key_points_valid) or \
                                 (is_left_hand_part and left_arm_key_points_valid) or \
                                 (is_right_hand_part and right_arm_key_points_valid)


                if should_shift_y and idx < len(augmented_points) and np.any(augmented_points[idx, 0:2] !=0):
                    augmented_points[idx, 1] += current_overall_dy_shift

        # Clip
        if clip_to_01:
            indices_to_clip = list(range(POSE_LM_LEFT_SHOULDER, num_total_landmarks)) # Từ vai trở đi
            for idx in indices_to_clip:
                 if idx < len(augmented_points) and np.any(augmented_points[idx,0:2] !=0): # Chỉ clip điểm XY khác 0
                    augmented_points[idx, 0] = np.clip(augmented_points[idx, 0], 0.0, 1.0)
                    augmented_points[idx, 1] = np.clip(augmented_points[idx, 1], 0.0, 1.0)

        augmented_frame_flat_output = augmented_points.flatten()
        if np.isnan(augmented_frame_flat_output).any() or np.isinf(augmented_frame_flat_output).any():
            augmented_sequence.append(frame_keypoints_flat.copy())
        else:
            augmented_sequence.append(augmented_frame_flat_output)
    return augmented_sequence