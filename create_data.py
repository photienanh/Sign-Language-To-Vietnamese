import cv2
import numpy as np
import os
import mediapipe as mp
import pandas as pd
import unicodedata
import re
from tqdm import tqdm
from tabulate import tabulate
import json
import shutil
from colorama import init, Fore, Style
from datetime import datetime
import logging
import tensorflow as tf
import warnings
from scipy import interpolate

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Tắt logging TensorFlow
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'  # Tắt GPU MediaPipe (chỉ sử dụng CPU)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Tắt DNN Optimization

logging.basicConfig(level=logging.ERROR) #Chỉ hiển thị thông báo ERROR, tránh spam WARNING, INFO
warnings.filterwarnings('ignore')

class ProgressStart():
    def __init__(self):
        self.starttime = datetime.now()
        self.total_processed = 0
        self.total_success = 0
    
    def update(self, success = False):
        self.total_processed += 1
        if success:
            self.total_success += 1

    def get_time(self):
        return datetime.now() - self.starttime

    def get_success_rate(self):
        return (self.total_success / self.total_processed *100) if self.total_processed > 0 else 0
    
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

def extract_keypoint(results):
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
    text = text.unicodedata(text, 'NFD').encode('ascii', 'ignore').decode('utf-8')
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    text = text.replace('d_', 'đ')
    return text


