import os
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2
import pandas as pd
from pymongo import MongoClient
from datetime import datetime

# Fungsi untuk menghitung kendaraan masuk dan keluar berdasarkan lintasan mereka


def update_vehicle_count(tracks, region_of_interest, counts, model):
    for track in tracks:
        for box in track.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            class_index = int(box.cls[0])
            class_name = model.names[class_index]

            # Periksa apakah titik tengah berada dalam ROI
            if region_of_interest[0][1] < center_y < region_of_interest[1][1]:
                if class_name in counts:
                    counts[class_name]['in'] += 1

# Fungsi untuk menyimpan hasil deteksi ke MongoDB


def save_to_mongodb(counts, collection):
    timestamp = datetime.now()
    for class_name, count in counts.items():
        document = {
            "jenis_kendaraan": class_name,
            "masuk": count['in'],
            "keluar": count['out'],
            "date": timestamp.strftime('%Y-%m-%d'),
            "hari": timestamp.strftime('%A')
        }
        collection.insert_one(document)

# Fungsi untuk mengambil data dari MongoDB dan menyimpannya ke CSV


def export_to_csv(collection):
    # Baca data dari MongoDB
    try:
        data = list(collection.find())
        if not data:
            print("No data found in MongoDB collection.")
            return

        df = pd.DataFrame(data)

        # Check if necessary columns are in dataframe
        required_columns = {'jenis_kendaraan',
                            'date', 'masuk', 'keluar', 'hari'}
        if not required_columns.issubset(df.columns):
            print(
                f"Required columns are missing from data: {required_columns - set(df.columns)}")
            return

    except Exception as e:
        print(f"Error reading from MongoDB: {e}")
        return

    # Simpan hasil ke CSV
    csv_file = 'hasil_deteksi_kendaraan.csv'
    df.to_csv(csv_file, index=False)
    print(f"Data has been exported to {csv_file}")


# Inisialisasi koneksi MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['db_jenis_kendaraan']
collection = db['hasil_deteksi']

# Inisialisasi model YOLO
try:
    model = YOLO('best.pt')
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Path video yang akan digunakan (gunakan jalur absolut)
video_path = os.path.abspath('video2 (1).mp4')

# Check if video file exists
if not os.path.exists(video_path):
    print(f"Video file '{video_path}' not found.")
    exit()

# Buka file video
cap = cv2.VideoCapture(video_path)
assert cap.isOpened(), "Error opening video file"

# Dapatkan informasi video
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                       cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

region_of_interest = [(20, 600), (1700, 604), (1700, 560), (20, 560)]

# Video writer
video_writer = cv2.VideoWriter(
    "object_counting_output.avi", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

# Init Object Counter
counter = object_counter.ObjectCounter()
counter.set_args(view_img=True, reg_pts=region_of_interest,
                 classes_names=model.names, draw_tracks=True)

# Dictionary untuk menyimpan jumlah deteksi per label
counts = {'car': {'in': 0, 'out': 0}, 'truck': {'in': 0, 'out': 0}}

# Proses setiap frame
while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break
    results = model.track(im0, persist=True, show=False)
    im0 = counter.start_counting(im0, results)
    video_writer.write(im0)

    # Update jumlah kendaraan yang masuk dan keluar
    update_vehicle_count(results, region_of_interest, counts, model)

cap.release()
video_writer.release()
cv2.destroyAllWindows()

# Simpan data ke MongoDB setelah selesai memproses video
save_to_mongodb(counts, collection)

# Ekspor data ke CSV
export_to_csv(collection)
