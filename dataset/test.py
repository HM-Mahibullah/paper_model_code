import cv2
import os

VIDEO_PATH = "27mahibullah.mp4"
OUTPUT_FOLDER = "27mahibullah_images"
INTERVAL_SECONDS = 0.18 

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * INTERVAL_SECONDS)

count = 0
saved = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if count % frame_interval == 0:
        filename = os.path.join(OUTPUT_FOLDER, f"frame_{saved:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved += 1
    count += 1

cap.release()
print(f"✅ মোট {saved}টি ছবি সেভ হয়েছে।")