# NHẬN DIỆN NGÔN NGỮ KÝ HIỆU VIỆT NAM
Hệ thống nhận dạng ngôn ngữ ký hiệu tiếng Việt sử dụng kỹ thuật học sâu và thị giác máy tính, chuyên biệt cho ngôn ngữ ký hiệu Việt Nam.
## Yêu cầu thiết bị
- Python 3.8+
- TensorFlow 2.x
- Scikit-learn
- MediaPipe
- OpenCV
- Streamlit
- Yêu cầu thiết bị có webcam (nếu muốn sử dụng tính năng nhận diện qua webcam)
## Cài đặt
### 1. Clone và setup
```bash
git clone https://github.com/photienanh/Sign-Language-To-Vietnamese
cd 'Sign Language To Vietnamese'
# Hoặc tải file zip thông qua Github.
```
### 2. Cài đặt Python và thư viện cần thiết
```bash
# Cài đặt Python từ trang web https://www.python.org/.
# Sau đó cài các thư viện cần thiết.
pip install -r requirements.txt
```
## Hướng dẫn sử dụng
### 1. Sử dụng trực tiếp
Sau khi đã cài đặt đầy đủ Python và thư viện, có thể sử dụng trực tiếp bằng lệnh:
```bash
streamlit run main.py
```
### 2.Thiết lập lại từ đầu
Có thể cài đăt lại mô hình từ đầu theo các bước sau.
```bash
# Xóa toàn bộ thiết lập trước đó.
Get-ChildItem -Path "./" -Directory | Remove-Item -Recurse -Force
```
Sau đó crawl data.
```bash
python download_data.py
```

Tiền xử lý dữ liệu.
```bash
python create_data_augment.py
```

Huấn luyện mô hình
```bash
# Run All trên training.ipynb.
# Nên sử dụng trên thiết bị có GPU.
```

Sau khi hoàn thành huấn luyện mô hình, có thể sử dụng mô hình để nhận diện qua giao diện ứng dụng.
```bash
streamlit run main.py
```
## Tính năng
- Tự động tải xuống video.
- Tự động xử lý dữ liệu đưa vào huấn luyện.
- Nhận diện ngôn ngữ ký hiệu thông qua video hoặc webcam.