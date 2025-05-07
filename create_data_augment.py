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
from augment_function import inter_hand_distance, scale_keypoints_sequence,rotate_keypoints_sequence,translate_keypoints_sequence,time_stretch_keypoints_sequence,solve_2_link_ik_2d_v2
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

mp_holistic = mp.solutions.holistic
N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS = 21
N_TOTAL_LANDMARKS = N_UPPER_BODY_POSE_LANDMARKS + N_HAND_LANDMARKS + N_HAND_LANDMARKS

ALL_POSE_CONNECTIONS = list(mp_holistic.POSE_CONNECTIONS)
UPPER_BODY_POSE_CONNECTIONS = []
for connection in ALL_POSE_CONNECTIONS:
    if connection[0] < N_UPPER_BODY_POSE_LANDMARKS and connection[1] < N_UPPER_BODY_POSE_LANDMARKS:
        UPPER_BODY_POSE_CONNECTIONS.append(connection)

def mediapipe_detection(image, model):
    # Mediapipe dùng RGB, cv2 dùng BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def extract_keypoints(results):
    pose_kps = np.zeros((N_UPPER_BODY_POSE_LANDMARKS, 3))
    left_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))
    right_hand_kps = np.zeros((N_HAND_LANDMARKS, 3))
    if results and results.pose_landmarks:
        for i in range(N_UPPER_BODY_POSE_LANDMARKS):
            if i < len(results.pose_landmarks.landmark):
                res = results.pose_landmarks.landmark[i]
                pose_kps[i] = [res.x, res.y, res.z]
    if results and results.left_hand_landmarks:
        left_hand_kps = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark])
    if results and results.right_hand_landmarks:
        right_hand_kps = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark])
    keypoints = np.concatenate([pose_kps,left_hand_kps, right_hand_kps])
    return keypoints.flatten()


def interpolate_keypoints(keypoints_sequence, target_len = 60):#nội suy chuỗi keypoints về 60 frames
    if len(keypoints_sequence) == 0:
        return None

    original_times = np.linspace(0, 1, len(keypoints_sequence))
    target_times = np.linspace(0, 1, target_len)

    num_features = keypoints_sequence[0].shape[0]
    interpolated_sequence = np.zeros((target_len, num_features))

    for feature_idx in range(num_features):
        feature_values = [frame[feature_idx] for frame in keypoints_sequence]

        interpolator = interp1d(
            original_times, feature_values,
            kind='cubic', #nội suy cubic
            bounds_error=False, #không báo lỗi nếu ngoài phạm vi
            fill_value="extrapolate" #ngoại suy nếu cần
        )
        interpolated_sequence[:, feature_idx] = interpolator(target_times)

    return interpolated_sequence

def sequence_frames(video_path, holistic):
  sequence_frames = []
  cap = cv2.VideoCapture(video_path)
  total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

  step = max(1, total_frames // 100)  # xác định bước nhảy để lấy mẫu frames

  while cap.isOpened():#đọc từng frame từ video
      ret, frame = cap.read()
      if not ret:
          break

      #nếu không phải frame cần lấy mẫu thì bỏ qua
      if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % step != 0:
          continue

      try:
          image, results = mediapipe_detection(frame, holistic)#dùng mediapipe để xác định keypoints
          keypoints = extract_keypoints(results)#trích xuất keypoints từ kết quả

          if keypoints is not None:
              sequence_frames.append(keypoints)

      except Exception as e:
          continue

  cap.release()
  return sequence_frames

def create_action_folder(data_path, action):
    action_path = os.path.join(data_path, action)
    os.makedirs(action_path, exist_ok=True)
    return action_path

class GetTime():
    def __init__(self):
        self.starttime = datetime.now()

    def get_time(self):
        return datetime.now() - self.starttime
    
augmentations = [
    scale_keypoints_sequence,
    rotate_keypoints_sequence,
    translate_keypoints_sequence,
    time_stretch_keypoints_sequence,
    inter_hand_distance
]

def generate_augmented_samples(
    original_sequence,
    augmentation_functions,
    num_samples_to_generate: int,
    max_augs_per_sample: int = 3, # Số lượng phép tăng cường tối đa cho mỗi mẫu
    #target_sequence_length: Optional[int] = None # Nếu cần pad/truncate về độ dài cố định
):
    """
    Tạo ra nhiều mẫu tăng cường bằng cách kết hợp ngẫu nhiên các hàm tăng cường.

    Args:
        original_sequence: Sequence keypoints gốc.
        augmentation_functions: Danh sách các hàm tăng cường để chọn.
        num_samples_to_generate: Số lượng mẫu tăng cường cần tạo.
        max_augs_per_sample: Số lượng phép tăng cường tối đa được áp dụng cho một mẫu.
        target_sequence_length: Độ dài mong muốn của các sequence đầu ra.

    Returns:
        Danh sách các sequence keypoints đã được tăng cường.
    """
    generated_samples = []
    if not original_sequence or not augmentation_functions:
        return generated_samples

    num_available_augs = len(augmentation_functions)

    for i in range(num_samples_to_generate):
        current_sequence = [kp.copy() if isinstance(kp, np.ndarray) else kp for kp in original_sequence] # Bắt đầu với bản sao

        # Chọn số lượng phép tăng cường để áp dụng (từ 1 đến max_augs_per_sample, không quá số hàm có sẵn)
        num_augs_to_apply = random.randint(1, min(max_augs_per_sample, num_available_augs))

        # Chọn ngẫu nhiên các hàm tăng cường (không lặp lại)
        selected_aug_funcs_indices = random.sample(range(num_available_augs), num_augs_to_apply)
        selected_aug_funcs = [augmentation_functions[idx] for idx in selected_aug_funcs_indices]

        # Xáo trộn thứ tự áp dụng (tùy chọn, nhưng thường là tốt)
        random.shuffle(selected_aug_funcs)

        # print(f"Sample {i+1}: Applying {num_augs_to_apply} augmentations: {[f.__name__ for f in selected_aug_funcs]}")

        # Áp dụng tuần tự
        for aug_func in selected_aug_funcs:
            # print(f"  Applying {aug_func.__name__}...")
            current_sequence = aug_func(current_sequence)
            # Kiểm tra nếu sequence trở thành None hoặc rỗng sau một phép aug
            if not current_sequence or all(frame is None for frame in current_sequence):
                # print(f"  Warning: Sequence became invalid after {aug_func.__name__}. Skipping further augs for this sample.")
                break

        if not current_sequence or all(frame is None for frame in current_sequence):
            # print(f"Sample {i+1} resulted in an invalid sequence. Skipping.")
            continue # Bỏ qua mẫu này nếu nó không hợp lệ

        generated_samples.append(current_sequence)

    return generated_samples


DATA_PATH = os.path.join('Data')
DATASET_PATH = os.path.join('Dataset')
LOG_PATH = os.path.join('Logs')

sequence_length = 60

os.makedirs(LOG_PATH, exist_ok=True)
label_file = os.path.join(DATASET_PATH, 'Text', 'final_label.csv')
video_folder = os.path.join(DATASET_PATH, 'Videos')
df = pd.read_csv(label_file)

selected_actions = []

os.makedirs(DATA_PATH, exist_ok=True)
os.makedirs(LOG_PATH, exist_ok=True)



selected_actions = sorted(df['LABEL'].unique())
label_map = { action: idx for idx, action in enumerate(selected_actions) }

label_map_path = os.path.join(LOG_PATH, 'label_map.json')
with open(label_map_path, 'w', encoding='utf-8') as f:
    json.dump(label_map, f, ensure_ascii=False, indent=4)


# lưu các hành động riêng biệt vào một file mapping
#save_action_mapping(selected_actions, LOG_PATH)
print(f"\n Selected {len(df['LABEL'].unique())} actions.")

'''
current_state = {
        'selected_actions': list(selected_actions),  # danh sách các hành động đang xử lý
        'progress': {
            action: 0 for action in selected_actions # dict theo dõi tiến độ của từng hành động
        }
    }
'''


time = GetTime()
print(f"{datetime.now()} Start processing data...")

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    action_position = {action: idx + 1 for idx, action in enumerate(pd.unique(df['LABEL']))}

    for _, row in tqdm(df.iterrows(), total=len(df), desc='Process actions'):
        action = row['LABEL']
        video_file = row['VIDEO']
        label      = label_map[action]

        print()
        #action_ascii = convert_to_ascii(action)
        action_path = create_action_folder(DATA_PATH, action)

        idx = 0
        #sequence_folder = os.path.join(action_path, str(idx))
        #os.makedirs(sequence_folder, exist_ok=True)

        video_path = os.path.join(video_folder, video_file)

        if not os.path.exists(video_path):
            print(f"Video not found: {video_path}")
            continue
        frame_lists = sequence_frames(video_path, holistic)

        augmenteds = generate_augmented_samples(frame_lists,augmentations,1000,5)
        augmenteds.append(frame_lists)

        for aug in augmenteds:
          seq = interpolate_keypoints(aug)
           # Lưu .npz chứa sequence và label
          file_path = os.path.join(action_path, f'{idx}.npz')
          np.savez(
              file_path,
              sequence=seq,
              label=label
          )
          idx += 1

        #current_state['progress'].update({action: idx + 1})
        #save_progress_state(current_state, LOG_PATH)

        print(f"Action {action_position[action]}/{len(df['LABEL'].unique())} : {action} - Time: {time.get_time()}")


print(f"{'-'*50}\n")
print("DATA PROCESSING COMPLETED.")