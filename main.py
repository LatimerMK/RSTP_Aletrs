#!/usr/bin/env python3
# main.py — RTSP motion alert with centroid-center trigger, dynamic sensitivity, monthly logs and photo archive

import cv2
import numpy as np
import requests
import logging
import time
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# ==== ENV / CONFIG ====
RTSP_URL = os.getenv("RTSP_URL")
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ==== Folders ====
RESOURCES_FOLDER = "RESOURCES"
os.makedirs(RESOURCES_FOLDER, exist_ok=True)

# ==== Runtime settings ====
RESTART_INTERVAL = 60 * 60      # періодичний рестарт (сек)
RETRY_DELAY = 5                 # затримка перед перезапуском при помилці (сек)

# Motion params (tweak these)
MOTION_THRESHOLD = 500          # базовий поріг (сума площі контурів)
MIN_BRIGHTNESS = 30             # мінімальна середня яскравість кадру (якщо нижче — вважаємо темно)
MIN_CONTOUR_AREA = 50           # мін. площа одного контуру щоб враховувати його
TARGET_SIZE = (2560, 1440)      # розмір збережених фото
ROI = (737, 534, 175, 146)      # (x, y, w, h) (654, 536, 266, 172)
MOTION_DELAY_FRAMES = 10         # кадри, які об'єкт має бути в центрі перед трігером
ALERT_INTERVAL = 5.0            # мін. інтервал між алертами для того самого об'єкта (сек)
TRIGGER_MEMORY_SECONDS = 8.0    # скільки пам'ятаємо останні тригери (щоб розрізняти авто)
MIN_DISTANCE_FOR_DIFFERENT = 100 # px - мін. дистанція центроїда щоб вважати об'єкт іншим
BRIGHTNESS_TRIGGER_DELTA = 40   # якщо яскравість стрибнула більше за це значення -> миттєвий тригер
DARK_DYNAMIC_FACTOR = 2.5       # наскільки підвищувати поріг у темряві (експериментально)
JPEG_QUALITY = 75               # 95 / 85 / 75  - 1.2 / 0.5 / 0.3 mb

# ==== Logging (monthly folder, UTF-8) ====
now = datetime.now()
month_log_folder = os.path.join(RESOURCES_FOLDER, now.strftime("%Y-%m"))
os.makedirs(month_log_folder, exist_ok=True)
LOG_FILE = os.path.join(month_log_folder, f"{now.strftime('%Y-%m-%d')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.info(f"✅ Логування запущено, файл: {LOG_FILE}")

def select_roi(RTSP_URL):
    cap = cv2.VideoCapture(RTSP_URL)
    ret, frame = cap.read()
    cap.release()
    roi_coords = []
    if not ret:
        print("❌ Не вдалося отримати кадр")
        return None

    clone = frame.copy()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(param) < 2:
                param.append((x, y))

    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", mouse_callback, roi_coords)

    print("🖱 ЛКМ — точка, Backspace — видалити, Enter — підтвердити, ESC — скасувати")

    while True:
        display = clone.copy()
        for pt in roi_coords:
            cv2.circle(display, pt, 5, (0, 0, 255), -1)

        if len(roi_coords) == 2:
            x1, y1 = roi_coords[0]
            x2, y2 = roi_coords[1]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("Frame", display)
        key = cv2.waitKey(1) & 0xFF

        if key == 13:  # Enter
            if len(roi_coords) == 2:
                break
            else:
                print("⚠️ Виберіть дві точки перед підтвердженням!")
        elif key == 8:  # Backspace
            if roi_coords:
                roi_coords.pop()
        elif key == 27:  # ESC
            roi_coords.clear()
            break

    cv2.destroyAllWindows()

    if len(roi_coords) == 2:
        (x1, y1), (x2, y2) = roi_coords
        x, y = min(x1, x2), min(y1, y2)
        w, h = abs(x2 - x1), abs(y2 - y1)
        print(f"✅ ROI = ({x}, {y}, {w}, {h})")
        return (x, y, w, h)
    else:
        print("❌ ROI не вибрано")
        return None


def send_telegram(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
        logging.info(f"✅ Відправлено: {message}")
    except Exception as e:
        logging.info(f"❌ Помилка при відправці: {e}")

# --- Основний цикл ---
last_sent_minute = -1  # щоб не відправляло кілька разів одну і ту ж хвилину

# ==== Helpers ====
def send_photo(photo_path):
    """Відправка фото в Telegram та збереження у папку ресурси/місяць_photo/день"""
    try:
        url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
        with open(photo_path, "rb") as f:
            files = {"photo": f}
            data = {"chat_id": CHAT_ID}
            resp = requests.post(url, files=files, data=data, timeout=15)
        if resp.status_code != 200:
            logging.warning(f"Telegram returned status {resp.status_code}: {resp.text}")
    except Exception as e:
        logging.exception(f"Помилка при відправці фото у Telegram: {e}")

    now = datetime.now()
    month_folder = os.path.join(RESOURCES_FOLDER, now.strftime("%Y-%m") + "_photo")
    os.makedirs(month_folder, exist_ok=True)
    day_folder = os.path.join(month_folder, now.strftime("%Y-%m-%d"))
    os.makedirs(day_folder, exist_ok=True)

    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    new_filename = f"{timestamp}.jpg"
    new_path = os.path.join(day_folder, new_filename)

    try:
        os.replace(photo_path, new_path)
    except Exception:
        os.rename(photo_path, new_path)
    logging.info(f"📸 Фото збережено: {new_path}")

def stretch_to_16_9(img, target_size=TARGET_SIZE):
    """Масштабування 16:9"""
    return cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

# ==== Main detection logic ====
def main():
    global ROI
    global MOTION_THRESHOLD
    global MIN_BRIGHTNESS
    global JPEG_QUALITY

    logging.info("🔄 Підключення до RTSP потоку...")
    cap = cv2.VideoCapture(RTSP_URL)

    ret, prev_frame = cap.read()
    if not ret:
        logging.error("❌ Не вдалося отримати перший кадр з RTSP")
        return
    logging.info("✅ Потік отримано, починаємо обробку кадрів")

    x, y, w, h = ROI
    prev_roi = cv2.cvtColor(prev_frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)

    last_alert_time = 0
    is_dark = False
    trigger_memory = []
    # --- Основний цикл ---
    last_sent_minute = -1  # щоб не відправляло кілька разів одну і ту ж хвилину
    recent_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            error_get_frame_msg = "⚠️ Не вдалося отримати кадр, перезапуск..."
            logging.warning(error_get_frame_msg)
            send_telegram(error_get_frame_msg)
            #continue
            return
        #---------------------------------------------------------
        now = datetime.now()
        minute = now.minute
        if minute % 10 == 0 and minute != last_sent_minute:
            check_online_msg = "Все окей, цикл працює 👍"
            send_telegram(check_online_msg)
            logging.warning(check_online_msg)
            last_sent_minute = minute
        # ---------------------------------------------------------

        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)

        # Динамічна чутливість при темряві
        motion_threshold_dynamic = MOTION_THRESHOLD
        if avg_brightness < MIN_BRIGHTNESS:
            motion_threshold_dynamic *= DARK_DYNAMIC_FACTOR

        # Миттєвий тригер при спалаху яскравості
        brightness_jump = np.mean(gray) - np.mean(prev_roi)
        if brightness_jump > BRIGHTNESS_TRIGGER_DELTA:
            motion_detected = True
        else:
            # Різниця кадрів
            diff = cv2.absdiff(prev_roi, gray)
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            motion_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA)
            motion_detected = motion_area > motion_threshold_dynamic

            motion_detected = False
            centroids_in_frame = []

            for c in contours:
                if cv2.contourArea(c) < MIN_CONTOUR_AREA:
                    continue
                M = cv2.moments(c)
                if M["m00"] == 0:
                    continue
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids_in_frame.append((cx, cy))

                # Центральна зона ROI
                if 0.25 * w < cx < 0.75 * w and 0.25 * h < cy < 0.75 * h:
                #if 0.25 * w < cx < 0.75 * w and 0.1 * h < cy < 0.9 * h:
                # рух всередині центральної горизонтальної та розширеної вертикальної зони
                    # Перевіряємо історію тригерів
                    now_time = time.time()
                    trigger_memory = [t for t in trigger_memory if now_time - t[1] < TRIGGER_MEMORY_SECONDS]
                    new_trigger = True
                    for mem_centroid, ts in trigger_memory:
                        dist = np.linalg.norm(np.array([cx, cy]) - np.array(mem_centroid))
                        if dist < MIN_DISTANCE_FOR_DIFFERENT:
                            new_trigger = False
                            break

                    if new_trigger:
                        motion_detected = True
                        # Відправка фото
                        if now_time - last_alert_time >= ALERT_INTERVAL:
                            stretched = stretch_to_16_9(frame)
                            filename = "alert.jpg"
                            cv2.imwrite(filename, stretched, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                            send_photo(filename)
                            last_alert_time = now_time
                            trigger_memory.append(((cx, cy), now_time))
                            logging.info(f"⚠️ Рух зафіксовано! Фото відправлено")

        prev_roi = gray

    cap.release()

    logging.info("🛑 Потік закрито")

# ==== Run wrapper with auto-restart ====
if __name__ == "__main__":
    #select_roi(RTSP_URL)
    logging.info("🚀 Запуск скрипта детекції руху")
    while True:
        start_time = time.time()
        try:
            main()
        except KeyboardInterrupt:
            logging.info("🔹 Отримано SIGINT — вихід")
            #break
        except Exception as e:
            logging.exception(f"❌ Виникла невідома помилка: {e}")
            logging.info(f"⏳ Перезапуск через {RETRY_DELAY} сек...")
            time.sleep(RETRY_DELAY)

    time.sleep(5)