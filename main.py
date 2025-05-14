import streamlit as st
import numpy as np
import tensorflow as tf
import tempfile
import os
import cv2
import mediapipe as mp
from scipy.interpolate import interp1d
import time
st.set_page_config(page_title="VSL Prediction", layout="centered")
st.title("DỰ ĐOÁN NGÔN NGỮ KÝ HIỆU")

mp_holistic = mp.solutions.holistic
N_UPPER_BODY_POSE_LANDMARKS = 25
N_HAND_LANDMARKS = 21
N_TOTAL_LANDMARKS = N_UPPER_BODY_POSE_LANDMARKS + N_HAND_LANDMARKS + N_HAND_LANDMARKS

ALL_POSE_CONNECTIONS = list(mp_holistic.POSE_CONNECTIONS)
UPPER_BODY_POSE_CONNECTIONS = []
# ====================
# Load model và label_map
# ====================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('Models/checkpoints/final_model.keras')  # model đã huấn luyện

@st.cache_data
def load_label_map():
    import json
    with open('Logs/label_map.json', 'r', encoding='utf-8') as f:
        label_map = json.load(f)
    inv_label_map = {v: k for k, v in label_map.items()}
    return label_map, inv_label_map

model = load_model()
label_map, inv_label_map = load_label_map()
# ====================
# Hàm xử lý video (placeholder)
# ====================
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

def process_webcam_to_sequence():
    cap = cv2.VideoCapture(0)  # Sử dụng webcam mặc định
    st.write("⏳ Đang chuẩn bị... Bắt đầu trong 1.5 giây...")
    time.sleep(1.5)  # Hiển thị thông báo trong 1.5 giây
    
    # Đọc video từ webcam trong 4 giây
    st.write("🎥 Đang ghi hình trong 4 giây...")
    sequence = []
    start_time = time.time()

    # Khởi tạo Mediapipe Holistic model
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Không thể truy cập webcam")
            break
        elapsed_time = time.time() - start_time
        if elapsed_time > 4:  # Sau 4 giây thì dừng
            break
        # Chuyển đổi frame từ BGR (OpenCV) sang RGB (Mediapipe)
        image, results = mediapipe_detection(frame, holistic)

        # Trích xuất keypoints từ kết quả của Mediapipe
        keypoints = extract_keypoints(results)
        
        # Thêm keypoints vào chuỗi (có thể dừng sau 60 frames hoặc khi người dùng nhấn nút)
        if keypoints is not None:
            sequence.append(keypoints)

        # Hiển thị webcam feed trên Streamlit
        stframe.image(image, channels="BGR", caption="Webcam feed", use_container_width=True)

    cap.release()
    
    return sequence

# Streamlit App

input_mode = st.radio("Chọn nguồn đầu vào:", ["🎞️ Video file", "📷 Webcam"])

sequence = None
holistic =mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
if input_mode == "🎞️ Video file":
    uploaded_file = st.file_uploader("Tải lên video (.mp4, .avi)", type=["mp4", "avi"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        st.video(tmp_path)
        if st.button("🔍 Dự đoán từ video"):
            sequence = sequence_frames(tmp_path, holistic)

elif input_mode == "📷 Webcam":
    st.warning("Nhấn nút bên dưới để bắt đầu ghi hình từ webcam.")
    if st.button("📸 Ghi và dự đoán"):
        sequence = process_webcam_to_sequence()

# Dự đoán
if sequence is not None:
    kp = interpolate_keypoints(sequence)
    result = model.predict(np.expand_dims(kp, axis=0))
    pred_idx = np.argmax(result, axis=1)
    pred_label = [inv_label_map[idx] for idx in pred_idx]
    st.success(f"✅ Nhãn dự đoán: **{pred_label}**")

