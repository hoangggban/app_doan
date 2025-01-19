from flask import Flask, Response, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import supervision as sv
import math
import json
from time import time
import math
from influxdb_client import InfluxDBClient, Point, WritePrecision
app = Flask(__name__)
url = "https://us-east-1-1.aws.cloud2.influxdata.com"
token = "1XY7Hx6f5xRZOewAYVg_GIptoJ_XvKhlbZ36zVZAC6r-dnfJkpXyCK4JjVXwhK2w_7f39FDen88etQqkDNkfNw=="
org = "Hanoi"
bucket = "hoangban"
client = InfluxDBClient(url=url, token=token, org=org)
write_api = client.write_api()


client.close()
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
def compute_homography(src_points, dst_points):
    """
    Tính toán ma trận biến chuyển phối cảnh (homography matrix) từ các điểm tương ứng.
    
    :param src_points: Danh sách các điểm nguồn (trên hình)
    :param dst_points: Danh sách các điểm đích (trên thực tế)
    :return: Ma trận biến chuyển phối cảnh (3x3)
    """
    # Chuyển đổi các điểm thành mảng numpy
    src_points = np.array(src_points, dtype=np.float32)
    dst_points = np.array(dst_points, dtype=np.float32)
    
    # Tính toán ma trận biến chuyển phối cảnh
    H, status = cv2.findHomography(src_points, dst_points)
    return H

# Các điểm trên hình (ảnh thực tế)
src_points = [(0, 560), (1170, 560), (840, 340), (440, 340)]

# Các điểm trên thực tế
dst_points = [(0, 54), (18, 54), (18, 0), (0, 0)]

# Tính toán ma trận biến chuyển phối cảnh
H = compute_homography(src_points, dst_points)
def transform_point(center_x, center_y, H):
    # Chuyển điểm thành mảng numpy và thêm phần tử w=1 để sử dụng ma trận 3x3
    point_homogeneous = np.array([center_x, center_y, 1.0], dtype=np.float32)
    
    # Biến đổi điểm bằng ma trận H
    transformed_point = np.dot(H, point_homogeneous)
    
    # Chia cho phần tử thứ 3 (w) để chuẩn hóa tọa độ
    # Đây là bước chuẩn hóa nếu w != 1
    transformed_point /= transformed_point[2]
    
    return transformed_point[0], transformed_point[1]
model = YOLO("best.pt")
model1 = YOLO("accident.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()
polygon = np.array(((0, 560), (1170,560) , (840,340) , (440, 340)))
zone= sv.PolygonZone(polygon=polygon)


# Hàm tính vận tốc
uploaded_video_path = None
traffic_alerts = []


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    global uploaded_video_path
    if 'video' not in request.files:
        return redirect(url_for('index'))

    file = request.files['video']
    if file.filename == '':
        return redirect(url_for('index'))

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    uploaded_video_path = file_path

    return redirect(url_for('video_page'))

@app.route('/video_page')
def video_page():
    return render_template('video.html')

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_time_interval = 1 / cap.get(cv2.CAP_PROP_FPS)
    frame_id = 0  # Đếm số frame
    previous_speeds = {}
    tracker_data = {}
    tracker_data_real = {}
    previous_speeds_history = {}
    acceleration_threshold =50
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        results = model(frame)[0]
        detections = sv.Detections.from_ultralytics(results)
        mask1 = zone.trigger(detections=detections)
        detections = detections[mask1]
        detections = tracker.update_with_detections(detections=detections)
        vehicle_speeds = {}
        speeds_in_frame = []
        sudden_changes = []
        collision = False
        abnormal = False
        crowded = False
        congestion = False
        line_y = 450
        line_cross_count = 0  # Tổng số phương tiện vượt qua line (2 giây gần nhất)
        for tracker_id, bbox in zip(detections.tracker_id, detections.xyxy):
            # Tính tọa độ trung tâm
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2

            # Cập nhật lịch sử vị trí
            if tracker_id not in tracker_data:
                tracker_data[tracker_id] = []
                tracker_data_real[tracker_id]=[]
                previous_speeds_history[tracker_id] = [] 
            else:
                prev_x, prev_y = tracker_data[tracker_id][-1]
                if (prev_y < line_y and center_y >= line_y) or (prev_y >= line_y and center_y < line_y):
                        line_cross_count += 1
            x,y = transform_point(center_x,center_y,H)
            tracker_data[tracker_id].append((center_x, center_y))
            tracker_data_real[tracker_id].append((x,y))
         

            # Nếu có ít nhất 2 vị trí, tính vận tốc
            if len(tracker_data[tracker_id]) > 1:
                traces = np.array(tracker_data_real[tracker_id])
                dx = np.median(np.abs(np.diff(traces[:, 0])))
                dy = np.median(np.abs(np.diff(traces[:, 1])))

        # Khoảng cách trung vị và tốc độ
                distance = np.sqrt(dx**2 + dy**2)
                speed = distance / frame_time_interval
                if speed >0:
                   speed = round(speed * 3.6, 2)

        # Lưu vận tốc
                current_frame_id = frame_id
                speeds_in_frame.append(speed)
                vehicle_speeds[tracker_id] = speed   
                previous_speeds_history[tracker_id].append((current_frame_id, speed))
                if len(previous_speeds_history[tracker_id]) > 30:  # Giới hạn danh sách ở 10 phần tử
                       previous_speeds_history[tracker_id].pop(0)

        # Tính gia tốc nếu có đủ dữ liệu từ 10 phần tử
                if len(previous_speeds_history[tracker_id]) == 30:
            # Lấy phần tử đầu tiên và cuối cùng trong danh sách
                  frame_id_1, speed_1 = previous_speeds_history[tracker_id][0]
                  frame_id_10, speed_10 = previous_speeds_history[tracker_id][-1]
            
            # Tính thời gian thực giữa hai frame
                  time_difference = (frame_id_10 - frame_id_1) * frame_time_interval *3.6
            
                  if time_difference > 0:  # Đảm bảo thời gian hợp lệ
                     acceleration = (speed_10 - speed_1) / time_difference  # Gia tốc = Δv / Δt
                     acceleration = round(acceleration, 2)

                # Xác định thay đổi đột ngột dựa trên gia tốc
                     if abs(acceleration) > acceleration_threshold:  # Ngưỡng gia tốc đột ngột
                        sudden_changes.append((tracker_id, speed_1, speed_10, acceleration))
                        abnormal=True
            # Cập nhật vận tốc trước đó
                previous_speeds[tracker_id] = speed



        vehicles_in_roi = len(detections.tracker_id)
        if speeds_in_frame:
           max_speed = max(speeds_in_frame)
           min_speed = min(speeds_in_frame)
           avg_speed = round(sum(speeds_in_frame) / len(speeds_in_frame), 2)
        else:
           max_speed = 0
           min_speed = 0
           avg_speed = 0
        
# Cập nhật line_cross_count mỗi 2 giây
         
        labels = [
        f"#{tracker_id} {int(vehicle_speeds.get(tracker_id, 0)) if tracker_id in vehicle_speeds else 'N/A'} km/h"
        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id)
]
        annotated_frame = box_annotator.annotate(
        frame.copy(), detections=detections)
        annotated_frame = label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels)
        frame= trace_annotator.annotate(
        annotated_frame, detections=detections)
        frame = sv.draw_polygon(scene=frame,polygon=polygon)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        payloaddata = {
            "frame": frame_bytes.hex(),
            "frame_id": frame_id,
            "vehicles_in_roi": vehicles_in_roi,
            "max_speed" : max_speed,
            "min_speed" : min_speed,
            "avg_speed": avg_speed,
            "line_cross_count": line_cross_count
        }
        if vehicles_in_roi>20:
           congestion = True
        if vehicles_in_roi > 10 and vehicles_in_roi <=20:
            crowded=True
            


        alerts = {
            "collision": False,    # Cảnh báo va chạm
            "congestion": False,  # Cảnh báo tắc nghẽn
            "abnormal": False,    # Cảnh báo bất thường
            "crowded": False    # Cảnh báo đông đúc
         }
        


        # Gửi frame và dữ liệu kèm theo
        yield f"data: {json.dumps(payloaddata)}\n\n"

        
        yield f"data: {json.dumps(alerts)}\n\n"
        point = Point("traffic_data")\
                .tag("location", "Intersection A")\
                .field("vehicle_count", vehicles_in_roi)
        write_api.write(bucket=bucket, org=org, record=point)

        frame_id += 1
    cap.release()

@app.route('/traffic_data')
def traffic_data():
    return Response(process_video(uploaded_video_path), content_type='text/event-stream')
@app.route('/alerts')
def alerts():
    return Response(process_video(uploaded_video_path), content_type='text/event-stream')
@app.teardown_appcontext
def close_influx_client(exception):
    client.close()

if __name__ == "__main__":
    app.run(debug=True)
