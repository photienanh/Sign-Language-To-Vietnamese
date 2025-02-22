from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support import expected_conditions
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.by import By
import os
import csv
import requests
import urllib
import zipfile
from tqdm import tqdm
from urllib.parse import urlparse
from concurrent.futures import ThreadPoolExecutor
import threading

BASE_URL = "https://qipedc.moet.gov.vn"
chrome_driver_url = "https://storage.googleapis.com/chrome-for-testing-public/133.0.6943.126/win64/chromedriver-win64.zip"
chrome_driver_path = "chromedriver-win64/chromedriver.exe"
videos_dir = "Data/Videos"
text_dir = "Data/Text"
os.makedirs(videos_dir, exist_ok=True)  
os.makedirs(text_dir, exist_ok=True)
csv_path = os.path.join(text_dir, "label.csv")
csv_lock = threading.Lock()

def download_chrome_driver():
    if not os.path.exists(chrome_driver_path):
        urllib.request.urlretrieve(chrome_driver_url, 'chromedriver-win64.zip')
        zip = zipfile.ZipFile('chromedriver-win64.zip') 
        zip.extractall()
        zip.close()
        os.remove('chromedriver-win64.zip')


def handle_recursive_scrapping(dict: dict, driver):
    vids = WebDriverWait(driver = driver, timeout=3).until(
        expected_conditions.presence_of_all_elements_located((By.CSS_SELECTOR, "section:nth-of-type(2) > div:nth-of-type(2) > div:nth-of-type(1)"))
    )
    vids = driver.find_elements(By.CSS_SELECTOR, "section:nth-of-type(2) > div:nth-of-type(2) > div:nth-of-type(1) > a")
    for vid in vids:
        label = vid.find_element(By.CSS_SELECTOR, "p").text
        thumbs_url = vid.find_element(By.CSS_SELECTOR, "img").get_attribute("src")
        video_id = thumbs_url.replace("https://qipedc.moet.gov.vn/thumbs/", "").replace(".png", "")
        video_url = f"{BASE_URL}/videos/{video_id}.mp4"
        item = {'label': label, 'video_url': video_url}
        dict.append(item)

def csv_init():
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='', encoding= 'utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(["ID", "VIDEO", "LABEL"])

def add_to_csv(id, video, label):
    with csv_lock:
        with open(csv_path, 'a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow([id, video, label])

def download_video(video_data):
    video_url = video_data.get('video_url')
    label = video_data.get('label')
    filename = os.path.basename(urlparse(video_url).path)
    output_path = os.path.join(videos_dir, filename)
    if os.path.exists(output_path):
        print(f"Skip: {filename}")
        return
    try:
        print(f"Downloading: {filename}")
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(output_path, 'wb') as file, tqdm(
            desc=f"Progess {filename}",
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
            ncols=100
        ) as bar:
            for data in response.iter_content(chunk_size=8192):
                size = file.write(data)
                bar.update(size)

        id = sum(1 for _ in open(csv_path, encoding='utf-8'))
        add_to_csv(id, filename, label)                  
        print(f"Completed: {filename}")
        print(f"Updated label.csv: add {label}")

    except Exception as e:
        print(f"Error{filename}: {str(e)}")
        if os.path.exists(output_path):
            os.remove(output_path)  

def crawl_videos():
    print("CRAWLING VIDEOS")
    options = Options().add_argument('--disable-dev-shm-usage')
    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service, options=options)
    videos = []
    try:
        driver.get("https://qipedc.moet.gov.vn/dictionary")
        print("Connected to dictionary website")
        
        handle_recursive_scrapping(videos, driver)

        for i in range(2, 5):
            id = i
            if i != 2: id = i + 1
            button = driver.find_element(By.CSS_SELECTOR, f"button:nth-of-type({id})")
            button.click()
            handle_recursive_scrapping(videos, driver)
            
        for i in range(5, 218):
            id = 6
            button = driver.find_element(By.CSS_SELECTOR, f"button:nth-of-type({id})")
            button.click()
            handle_recursive_scrapping(videos, driver)

        for i in range(218, 220):
            id = 6
            if i != 218: id = 7
            button = driver.find_element(By.CSS_SELECTOR, f"button:nth-of-type({id})")
            button.click()
            handle_recursive_scrapping(videos, driver)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        driver.close()
        return videos

def main():    
    download_chrome_driver()
    videos = crawl_videos()
    if videos:
        print(f"Found {len(videos)} videos\n")
    
    print("STARTING DOWNLOAD VIDEOS")
    csv_init()
    
    if not videos:
        print("Videos not found")
        return
        
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(download_video, videos)
        
    print(f"DOWNLOAD COMPLETED {videos_dir}")

if __name__ == "__main__":
    main()