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

os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'  # Tắt GPU MediaPipe (chỉ sử dụng CPU)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Tắt DNN Optimization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Tắt thông báo tensorflow

logging.getLogger().setLevel(logging.ERROR)  # Tắt tất cả log mặc định
logging.disable(logging.CRITICAL)
logging.basicConfig(level=logging.ERROR) #Chỉ hiển thị thông báo ERROR, tránh spam WARNING, INFO
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class GetTime():
    def __init__(self):
        self.starttime = datetime.now()

    def get_time(self):
        return datetime.now() - self.starttime
    
mp_hands = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    # Mediapipe dùng RGB, cv2 dùng BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False  
    results = model.process(image)  
    image.flags.writeable = True  
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results  

def extract_keypoints(results):
    try: #lưu tọa độ 21 điểm của 1 bàn tay
        left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        keypoints = np.concatenate([left_hand, right_hand])

        if np.isnan(keypoints).any() or np.isinf(keypoints).any():
            return None
        if len(keypoints) != 21*3*2:
            return None 
        return keypoints
    except Exception:
        return None
    
def convert_to_ascii(text):
    text = text.lower()
    text = text.replace('đ', 'd_')
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    text = text.replace('d_', 'đ')
    return text

def create_action_folder(data_path, action):
    action_path = os.path.join(data_path, action)
    os.makedirs(action_path, exist_ok=True)
    return action_path

def save_action_mapping(actions, log_path = 'Logs'):
    os.makedirs(log_path, exist_ok=True)
    mapping_file = os.path.join(log_path, 'action_mapping.json')
    mapping = {
        'action': {convert_to_ascii(action): action for action in actions}
    }

    with open(mapping_file, 'w', encoding='utf-8') as file:
        json.dump(mapping, file, ensure_ascii=False, indent=2)
    print(f'Action mapping saved to {mapping_file}')

def load_action_mapping(log_path = 'Logs'):
    mapping_file = os.path.join(log_path, 'action_mapping.json')
    try:
        with open(mapping_file, 'r') as file:
            mapping = json.load(file)
        return mapping['action']
    except FileNotFoundError:
        print('Action mapping file not found')
        return {}
    except Exception as e:
        print('Error loading action mapping file:', e)
        return {}
    
def get_action_name(action_ascii, mapping=None):
    if mapping is None:
        mapping = load_action_mapping()
    return mapping.get(action_ascii, action_ascii)

def save_progress_state(state_data, log_path='Logs'):
    os.makedirs(log_path, exist_ok=True)
    state_file = os.path.join(log_path, 'progress_state.json')
    with open(state_file, 'w', encoding='utf-8') as f:
        json.dump(state_data, f, ensure_ascii=False, indent=2)

def load_progress_state(log_path='Logs'):
    state_file = os.path.join(log_path, 'progress_state.json')
    try:
        with open(state_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    
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

def process_video_sequence(video_path, holistic, sequence_length=40):
    # mở video và lấy frames
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
    
    if len(sequence_frames) < 3:  #cần ít nhất 3 frames cho nội suy cubic
        return None
        
    try:
        interpolated_sequence = interpolate_keypoints(sequence_frames, sequence_length)
        return interpolated_sequence
    except Exception as e: #bỏ qua video
        print(f"Error while interpolating: {str(e)}")
        return 

def collect_data_from_videos():
    DATA_PATH = os.path.join('Data')
    DATASET_PATH = os.path.join('Dataset')
    LOG_PATH = os.path.join('Logs')

    no_sequences = 10
    sequence_length = 40

    os.makedirs(LOG_PATH, exist_ok=True)
    label_file = os.path.join(DATASET_PATH, 'Text', 'label.csv')
    video_folder = os.path.join(DATASET_PATH, 'Videos')
    df = pd.read_csv(label_file)

    df_filtered = None
    selected_actions = []
    previous_state = load_progress_state(LOG_PATH)

    if previous_state and os.path.exists(DATA_PATH):
        while True:
            print('Previous state found. Continue from previous state? (y/n): ', end='')
            choice = input().strip().lower()
            if choice in ['y', 'n']:
                break
            
        if choice == 'y':
            remaining_actions = [action for action in previous_state['progress'] if previous_state['progress'][action] < no_sequences]
            print("Continuing from previous state...")
            df_filtered = df[df['LABEL'].isin(remaining_actions)]
            current_state = previous_state
            print(f"Remaining actions: {len(remaining_actions)}")

        else:
            if os.path.exists(DATA_PATH):
                shutil.rmtree(DATA_PATH)
            if os.path.exists(LOG_PATH):
                shutil.rmtree(LOG_PATH)
            print("Previous state deleted.")
            previous_state = None

    if df_filtered is None:
        os.makedirs(DATA_PATH, exist_ok=True)
        os.makedirs(LOG_PATH, exist_ok=True)
        selected_actions = np.array(df['LABEL'].unique()) 
        df_filtered = df[df['LABEL'].isin(selected_actions)]
        save_action_mapping(selected_actions, LOG_PATH)
        print(f"\n Selected {len(df['LABEL'].unique())} actions.")
        current_state = {
                    'selected_actions': list(selected_actions),
                    'progress': {
                        action: 0 for action in selected_actions
                    }
                }

    time = GetTime()
    print(f"{datetime.now()} Start processing data...")

    with  mp_hands.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action, group in tqdm(df_filtered.groupby('LABEL'), desc='Process actions'):
            print()
            action_ascii = convert_to_ascii(action)
            action_path = create_action_folder(DATA_PATH, action_ascii)

            action_position = {action: idx + 1 for idx, action in enumerate(sorted(df['LABEL'].unique()))}
            start_sequence = 0
            if previous_state and action in previous_state['progress'] and previous_state['progress'][action] < no_sequences:
                start_sequence = previous_state['progress'][action]

            for sequence in range(start_sequence, no_sequences):
                sequence_folder = os.path.join(action_path, str(sequence))
                os.makedirs(sequence_folder, exist_ok=True)

                video_row = group.sample(1).iloc[0]
                video_path = os.path.join(video_folder, video_row['VIDEO'])

                if not os.path.exists(video_path):
                    print(f"Video not found: {video_path}")
                    continue

                interpolated_sequence = process_video_sequence(video_path, holistic, sequence_length)

                if interpolated_sequence is not None:
                    for frame_idx, keypoint in enumerate(interpolated_sequence):
                        frame_path = os.path.join(sequence_folder, f'{frame_idx}.npy')
                        np.save(frame_path, keypoint)
                
                else:
                    continue

                current_state['progress'].update({action: sequence + 1})
                save_progress_state(current_state, LOG_PATH)

                print(f"Action {action_position[action].__index__()}/{len(df['LABEL'].unique())} : {action} - Sequence: {sequence + 1}/{no_sequences} - Time: {time.get_time()}")


    print(f"{'-'*50}\n")
    print("DATA PROCESSING COMPLETED.")                   

def count_collected_data():
    count = len(next(os.walk('Data'))[1]) if os.path.exists('Data') else 0  
    print(f"Total collected actions: {count}")
    return count

def main():
    print(f"{'-'*50}\n")
    print("Start collecting data...")
    if os.path.exists(os.path.join('Logs', 'progress_state.json')):
        print("Previous state founds")
    collect_data_from_videos()
    count_collected_data()

if __name__ == "__main__":
    main()


