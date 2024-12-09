import cv2
import numpy as np
import csv
from ultralytics import YOLO

# Tạo mapping từ class ID sang ký tự
class_to_char = {
    0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9",
    10: "A", 11: "B", 12: "C", 13: "D", 14: "E", 15: "F", 16: "G", 17: "H", 18: "I",
    19: "J", 20: "K", 21: "L", 22: "M", 23: "N", 24: "O", 25: "P", 26: "Q", 27: "R",
    28: "S", 29: "T", 30: "U", 31: "V", 32: "W", 33: "X", 34: "Y", 35: "Z"
}

# Hàm sắp xếp bounding box
def sort_boxes(detected_chars, max_line_gap=50):

    if not detected_chars:
        return [], []

    # Sắp xếp ban đầu theo y (tọa độ hàng) trước, rồi theo x (tọa độ cột)
    detected_chars.sort(key=lambda box: (box[1], box[0]))

    lines = []  # Lưu các hàng
    current_line = [detected_chars[0]]  # Hàng hiện tại

    for i in range(1, len(detected_chars)):
        # So sánh tọa độ y để xác định ký tự thuộc cùng hàng
        if abs(detected_chars[i][1] - current_line[0][1]) <= max_line_gap:
            current_line.append(detected_chars[i])
        else:
            # Thêm hàng hiện tại đã sắp xếp từ trái qua phải
            lines.append(sorted(current_line, key=lambda box: box[0]))
            current_line = [detected_chars[i]]

    # Thêm hàng cuối cùng
    lines.append(sorted(current_line, key=lambda box: box[0]))

    # Kết hợp tất cả các hàng theo thứ tự từ trên xuống dưới
    sorted_chars = [char for line in lines for char in line]
    
    return sorted_chars, lines

# Load model YOLOv8
model_plate = YOLO(r"C:\Users\doanv\PycharmProjects\BaiTapLon-XLA\models\license_plate\best.pt")  # Model nhận diện biển số
model_chars = YOLO(r"C:\Users\doanv\PycharmProjects\BaiTapLon-XLA\models\char\best.pt")  # Model nhận diện ký tự

# Xử lý video
def process_video(video_path, output_csv_path, output_video_path, conf_thresh=0.3):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Tạo writer để lưu video kết quả
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Tạo file CSV để lưu kết quả
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Frame", "License_Plate_Text", "Char_X1", "Char_Y1", "Char_X2", "Char_Y2", "Char_Label"])

        frame_index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Dùng model nhận diện biển số
            results_plate = model_plate.predict(source=frame, conf=conf_thresh, save=False, save_txt=False)[0]
            plates = []
            for plate_box in results_plate.boxes.xyxy.tolist():
                x1, y1, x2, y2 = map(int, plate_box[:4])
                plates.append((x1, y1, x2, y2))
                cropped_plate = frame[y1:y2, x1:x2]

                # Dùng model nhận diện ký tự
                results_chars = model_chars.predict(source=cropped_plate, conf=conf_thresh, save=False, save_txt=False)[0]
                chars = []
                for char_box, char_cls in zip(results_chars.boxes.xyxy.tolist(), results_chars.boxes.cls.tolist()):
                    cx1, cy1, cx2, cy2 = map(int, char_box[:4])
                    label = class_to_char[int(char_cls)]  # Mapping class ID thành ký tự
                    chars.append((cx1, cy1, cx2, cy2, label))

                # Sắp xếp ký tự
                sorted_chars, _ = sort_boxes(chars)

                # Tạo chuỗi ký tự cho biển số
                license_plate_text = ''.join([char[4] for char in sorted_chars])

                # Vẽ bounding box ký tự và lưu vào CSV
                for char in sorted_chars:
                    cx1, cy1, cx2, cy2, label = char
                    cv2.rectangle(cropped_plate, (cx1, cy1), (cx2, cy2), (0, 255, 0), 2)
                    cv2.putText(cropped_plate, label, (cx1, cy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    
                    writer.writerow([frame_index, license_plate_text, cx1 + x1, cy1 + y1, cx2 + x1, cy2 + y1, label])

                # Vẽ bounding box biển số và kết quả
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, license_plate_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Ghi khung hình đã xử lý vào video đầu ra
            out.write(frame)

            # Hiển thị khung hình (tùy chọn)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_index += 1

    # Giải phóng tài nguyên
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Gọi hàm xử lý video
process_video(
    video_path=r"C:\Users\doanv\PycharmProjects\BaiTapLon-XLA\video\video2.mp4", 
    output_csv_path=r"C:\Users\doanv\PycharmProjects\BaiTapLon-XLA\output_video\license_plate_results2.csv", 
    output_video_path=r"C:\Users\doanv\PycharmProjects\BaiTapLon-XLA\output_video\video2.mp4", 
    conf_thresh=0.1  # Chỉnh ngưỡng nhận diện ở đây
)
