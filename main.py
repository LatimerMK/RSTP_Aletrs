import cv2
import numpy as np
import requests
import logging
import time
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# ==== Налаштування ====
RTSP_URL = os.getenv("RTSP_URL")
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_BOT_TOKEN")

# Основна папка ресурсів
RESOURCES_FOLDER = "RESOURCES"
if not os.path.exists(RESOURCES_FOLDER):
    os.makedirs(RESOURCES_FOLDER)

# ==== Налаштування  ====
RESTART_INTERVAL = 60 * 60  # Періодичний рестарт кожну годину
RETRY_DELAY = 5              # Затримка перед повторним стартом після збою (сек)
# Мінімальний рівень руху (чим менше число – тим чутливіше)
MOTION_THRESHOLD = 2
MIN_BRIGHTNESS = 30       # мінімальна середня яскравість кадру
MIN_CONTOUR_AREA = 50     # мінімальна площа контуру для врахування руху
# Розмір для 16:9 (FullHD)
TARGET_SIZE = (1920, 1080)
# ROI координати (x, y, w, h) - під себе
ROI = (654, 536, 266, 172)

roi_coords = []

# ==== Налаштування логів ====
now = datetime.now()
month_log_folder = os.path.join(RESOURCES_FOLDER, now.strftime("%Y-%m"))  # папка місяця для логів
if not os.path.exists(month_log_folder):
    os.makedirs(month_log_folder)

today_str = now.strftime("%Y-%m-%d")
LOG_FILE = os.path.join(month_log_folder, f"{today_str}.log")

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

def send_photo(photo_path):
    """Відправка фото в Telegram та збереження у папку ресурси/місяць/день"""
    # Надсилаємо фото
    url = f"https://api.telegram.org/bot{TOKEN}/sendPhoto"
    with open(photo_path, "rb") as f:
        files = {"photo": f}
        data = {"chat_id": CHAT_ID}
        requests.post(url, files=files, data=data)

    now = datetime.now()
    # Папка місяця для фото
    month_folder = os.path.join(RESOURCES_FOLDER, now.strftime("%Y-%m") + "_photo")
    if not os.path.exists(month_folder):
        os.makedirs(month_folder)

    # Папка дня всередині місячної
    day_folder = os.path.join(month_folder, now.strftime("%Y-%m-%d"))
    if not os.path.exists(day_folder):
        os.makedirs(day_folder)

    # Формуємо ім'я файлу за датою та часом
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
    new_filename = f"{timestamp}.jpg"
    new_path = os.path.join(day_folder, new_filename)

    # Переміщаємо файл
    os.rename(photo_path, new_path)
    logging.info(f"📸 Фото збережено: {new_path}")

def stretch_to_16_9(img, target_size=TARGET_SIZE, contrast=1.2, brightness=10, denoise_strength=2):
    """
    Покращує картинку:
    1. Масштабування до 16:9
    2. Підвищення різкості
    3. Зменшення шуму
    4. Корекція контрасту та яскравості
    """
    # Масштабування з високоякісною інтерполяцією
    stretched = cv2.resize(img, target_size, interpolation=cv2.INTER_LANCZOS4)

    # Підвищення різкості
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(stretched, -1, kernel)
    # Зменшення шуму
    denoised = cv2.fastNlMeansDenoisingColored(sharpened, None, denoise_strength, denoise_strength, 7, 21)

    # Корекція контрасту та яскравості
    enhanced = cv2.convertScaleAbs(denoised, alpha=contrast, beta=brightness)

    return enhanced


def main():
    global ROI
    global MOTION_THRESHOLD
    global MIN_BRIGHTNESS

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
    ALERT_INTERVAL = 30  # секунд, мінімальний інтервал між повідомленнями
    is_dark = False  # глобальна або зовнішня змінна перед циклом

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.warning("⚠️ Не вдалося отримати кадр, повтор спроби...")
            time.sleep(1)
            continue

        roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)

        if avg_brightness < MIN_BRIGHTNESS:
            if not is_dark:  # лог тільки при переході в темряву
                logging.info("🌙 Дуже темно, дрібні рухи ігноруємо")
                is_dark = True
            prev_roi = gray
            continue
        else:
            if is_dark:  # лог переходу з темряви в світло
                logging.info("💡 Кадр достатньо освітлений, обробка руху")
                is_dark = False

        # Різниця кадрів
        diff = cv2.absdiff(prev_roi, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        # Фільтруємо дрібні об’єкти
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_area = sum(cv2.contourArea(c) for c in contours if cv2.contourArea(c) > MIN_CONTOUR_AREA)

        logging.debug(f"ℹ️ Рівень руху (сума площ): {motion_area}, яскравість: {avg_brightness}")

        if motion_area > MOTION_THRESHOLD:
            now = time.time()
            if now - last_alert_time >= ALERT_INTERVAL:
                stretched = stretch_to_16_9(frame)
                filename = "alert.jpg"
                cv2.imwrite(filename, stretched, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                send_photo(filename)
                logging.info(f"⚠️ Рух зафіксовано! Фото відправлено")
                last_alert_time = now
            else:
                logging.info("⚠️ Рух зафіксовано, але повідомлення відправляти рано (інтервал)")

        prev_roi = gray

    cap.release()
    logging.info("🛑 Потік закрито")

if __name__ == "__main__":

    #ROI = select_roi(RTSP_URL)
    logging.info("🚀 Запуск скрипта детекції руху")

    while True:
        start_time = time.time()
        try:
            # Якщо хочеш обирати ROI через GUI:
            # ROI = select_roi(RTSP_URL)
            main()  # запускаємо основний цикл
        except Exception as e:
            logging.error(f"❌ Виникла помилка: {e}")
            logging.info(f"⏳ Перезапуск через {RETRY_DELAY} секунд...")
            time.sleep(RETRY_DELAY)

        # Перевірка на періодичний рестарт
        elapsed = time.time() - start_time
        if elapsed < RESTART_INTERVAL:
            sleep_time = RESTART_INTERVAL - elapsed
            logging.info(f"⏳ Періодичний рестарт через {int(sleep_time)} секунд")
            time.sleep(sleep_time)
