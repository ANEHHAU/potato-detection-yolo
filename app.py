import cv2
import numpy as np
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image

model = YOLO("best.pt")

vietnamese_names = {
    "Damaged potato": "Khoai nứt",
    "Defected potato": "Khoai biến dạng",
    "Diseased-fungal potato": "Khoai nấm bệnh",
    "Potato": "Khoai tốt",
    "Sprouted potato": "Khoai mọc mầm"
}

font = ImageFont.truetype("arial.ttf", 24)

# ===== IoU FUNCTION =====
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union = area1 + area2 - inter

    return inter/union if union != 0 else 0

video_path = "test6.mp4"
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
height, width, _ = frame.shape
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ===== VÙNG ĐẾM (TĂNG KÍCH CỠ) =====
box_width = 350
box_height = height

box_x1 = width//2 - box_width//2
box_x2 = width//2 + box_width//2
box_y1 = 0
box_y2 = height

SL = 0
id_class_map = {}
counted_ids = set()
id_state = {}   # outside / inside / counted

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(
        frame,
        conf=0.75,
        iou=0.5,
        persist=True,
        tracker="bytetrack.yaml"
    )

    for r in results:
        boxes = r.boxes

        if boxes.id is None:
            continue

        # ===== LỌC TRÙNG BOX =====
        filtered = []

        for box, track_id in zip(boxes, boxes.id):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            current_box = [x1, y1, x2, y2]
            keep = True

            for prev in filtered:
                iou = compute_iou(current_box, prev["box"])

                if iou > 0.6:
                    if conf < prev["conf"]:
                        keep = False
                    else:
                        prev["keep"] = False

            if keep:
                filtered.append({
                    "box": current_box,
                    "track_id": int(track_id),
                    "cls": int(box.cls[0]),
                    "conf": conf,
                    "keep": True
                })

        # ===== XỬ LÝ SAU LỌC =====
        for item in filtered:
            if not item["keep"]:
                continue

            x1, y1, x2, y2 = item["box"]
            track_id = item["track_id"]
            cls = item["cls"]
            conf = item["conf"]

            # LOCK CLASS
            if track_id not in id_class_map:
                id_class_map[track_id] = model.names[cls]

            locked_name = id_class_map[track_id]
            class_name = vietnamese_names.get(locked_name, locked_name)

            # ===== TÍNH TÂM =====
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            inside = (box_x1 <= cx <= box_x2) and (box_y1 <= cy <= box_y2)

            if track_id not in id_state:
                id_state[track_id] = "outside"

            # ===== LOGIC ĐẾM CHUẨN =====
            if id_state[track_id] == "outside" and inside:
                id_state[track_id] = "inside"

            elif id_state[track_id] == "inside" and not inside:
                if track_id not in counted_ids:
                    SL += 1
                    counted_ids.add(track_id)
                id_state[track_id] = "counted"

            # DRAW BOX
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            img_pil = Image.fromarray(frame)
            draw = ImageDraw.Draw(img_pil)
            draw.text((x1, y1 - 30),
                      f"ID:{track_id} {class_name} {conf:.2f}",
                      font=font,
                      fill=(0,255,0))
            frame = np.array(img_pil)

    # ===== VẼ VÙNG ĐẾM =====
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0,0,255), 3)

    # ===== HIỂN THỊ SỐ LƯỢNG =====
    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)
    draw.text((width - 250, 30),
              f"So luong: {SL}",
              font=font,
              fill=(255,0,0))
    frame = np.array(img_pil)

    cv2.imshow("QC Potato System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()














# import cv2
# import time
# import numpy as np
# from ultralytics import YOLO
# from PIL import ImageFont, ImageDraw, Image

# model = YOLO("best.pt")

# vietnamese_names = {
#     "Damaged potato": "Khoai nứt",
#     "Defected potato": "Khoai biến dạng",
#     "Diseased-fungal potato": "Khoai nấm bệnh",
#     "Potato": "Khoai tốt",
#     "Sprouted potato": "Khoai mọc mầm"
# }

# font = ImageFont.truetype("arial.ttf", 24)

# # ===== IoU FUNCTION =====
# def compute_iou(box1, box2):
#     x1 = max(box1[0], box2[0])
#     y1 = max(box1[1], box2[1])
#     x2 = min(box1[2], box2[2])
#     y2 = min(box1[3], box2[3])

#     inter = max(0, x2 - x1) * max(0, y2 - y1)
#     area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
#     area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
#     union = area1 + area2 - inter

#     return inter/union if union != 0 else 0

# video_path = "test3.mp4"
# cap = cv2.VideoCapture(video_path)

# ret, frame = cap.read()
# height, width, _ = frame.shape
# line_x = width // 2
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# SL = 0
# id_class_map = {}
# counted_ids = set()
# previous_positions = {}

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model.track(
#         frame,
#         conf=0.25,
#         iou=0.5,
#         persist=True,
#         tracker="bytetrack.yaml"
#     )

#     for r in results:
#         boxes = r.boxes

#         if boxes.id is None:
#             continue

#         # ===== LỌC TRÙNG BOX =====
#         filtered = []

#         for box, track_id in zip(boxes, boxes.id):
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = float(box.conf[0])

#             current_box = [x1, y1, x2, y2]
#             keep = True

#             for prev in filtered:
#                 iou = compute_iou(current_box, prev["box"])

#                 if iou > 0.6:  # overlap cao -> giữ box conf lớn hơn
#                     if conf < prev["conf"]:
#                         keep = False
#                     else:
#                         prev["keep"] = False

#             if keep:
#                 filtered.append({
#                     "box": current_box,
#                     "track_id": int(track_id),
#                     "cls": int(box.cls[0]),
#                     "conf": conf,
#                     "keep": True
#                 })

#         # ===== XỬ LÝ SAU KHI LỌC =====
#         for item in filtered:
#             if not item["keep"]:
#                 continue

#             x1, y1, x2, y2 = item["box"]
#             track_id = item["track_id"]
#             cls = item["cls"]
#             conf = item["conf"]

#             # LOCK CLASS (giữ nguyên logic bạn)
#             if track_id not in id_class_map:
#                 id_class_map[track_id] = model.names[cls]

#             locked_name = id_class_map[track_id]
#             class_name = vietnamese_names.get(locked_name, locked_name)

#             cx = int((x1 + x2) / 2)

#             # ===== GIỮ NGUYÊN LOGIC ĐẾM CỦA BẠN =====
#             if track_id in previous_positions:
#                 prev_cx = previous_positions[track_id]

#                 if prev_cx < line_x - 5 and cx >= line_x + 5:
#                     if track_id not in counted_ids:
#                         SL += 1
#                         counted_ids.add(track_id)

#             previous_positions[track_id] = cx

#             # DRAW BOX
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             # TEXT
#             img_pil = Image.fromarray(frame)
#             draw = ImageDraw.Draw(img_pil)
#             draw.text((x1, y1 - 30),
#                       f"ID:{track_id} {class_name} {conf:.2f}",
#                       font=font,
#                       fill=(0,255,0))
#             frame = np.array(img_pil)

#     # DRAW LINE
#     cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 255), 3)

#     # SHOW COUNT
#     img_pil = Image.fromarray(frame)
#     draw = ImageDraw.Draw(img_pil)
#     draw.text((width - 250, 30),
#               f"So luong: {SL}",
#               font=font,
#               fill=(255,0,0))
#     frame = np.array(img_pil)

#     cv2.imshow("QC Potato System", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()





# import cv2
# import time
# import numpy as np
# from ultralytics import YOLO
# from PIL import ImageFont, ImageDraw, Image

# # ===== LOAD MODEL =====
# model = YOLO("best.pt")

# # ===== MAP TIẾNG VIỆT ĐÚNG CLASS CỦA BẠN =====
# vietnamese_names = {
#     "Damaged potato": "Khoai nứt",
#     "Defected potato": "Khoai biến dạng",
#     "Diseased-fungal potato": "Khoai nấm bệnh",
#     "Potato": "Khoai tốt",
#     "Sprouted potato": "Khoai mọc mầm"
# }

# # ===== LOAD FONT TIẾNG VIỆT =====
# font = ImageFont.truetype("arial.ttf", 24)

# video_path = "test3.mp4"
# cap = cv2.VideoCapture(video_path)

# if not cap.isOpened():
#     print("Không mở được video")
#     exit()

# ret, frame = cap.read()
# height, width, _ = frame.shape
# line_x = width // 2
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# prev_time = 0
# SL = 0

# id_class_map = {}
# counted_ids = set()
# previous_positions = {}

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model.track(
#         frame,
#         conf=0.25,
#         persist=True,
#         tracker="bytetrack.yaml"
#     )

#     for r in results:
#         boxes = r.boxes

#         if boxes.id is None:
#             continue

#         for box, track_id in zip(boxes, boxes.id):

#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             cls = int(box.cls[0])
#             track_id = int(track_id)

#             # ===== KHÓA CLASS =====
#             if track_id not in id_class_map:
#                 id_class_map[track_id] = model.names[cls]

#             locked_name = id_class_map[track_id]
#             class_name = vietnamese_names.get(locked_name, locked_name)

#             # ===== TÍNH TRỌNG TÂM =====
#             cx = int((x1 + x2) / 2)
#             cy = int((y1 + y2) / 2)

#             cv2.circle(frame, (cx, cy), 5, (255, 0, 0), -1)

#             # ===== ĐẾM CHÍNH XÁC KHI CẮT QUA VẠCH =====
#             if track_id in previous_positions:
#                 prev_cx = previous_positions[track_id]

#                 # Nếu trước ở bên trái và giờ ở bên phải
#                 if prev_cx < line_x and cx >= line_x:
#                     if track_id not in counted_ids:
#                         SL += 1
#                         counted_ids.add(track_id)

#             previous_positions[track_id] = cx

#             # ===== VẼ BOX =====
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#             # ===== VẼ TEXT BẰNG PIL =====
#             img_pil = Image.fromarray(frame)
#             draw = ImageDraw.Draw(img_pil)
#             draw.text((x1, y1 - 30), f"ID:{track_id} {class_name}", font=font, fill=(0,255,0))
#             frame = np.array(img_pil)

#     # ===== VẼ VẠCH =====
#     cv2.line(frame, (line_x, 0), (line_x, height), (0, 0, 255), 3)

#     # ===== HIỂN THỊ SL =====
#     img_pil = Image.fromarray(frame)
#     draw = ImageDraw.Draw(img_pil)
#     draw.text((width - 250, 30), f"Số lượng: {SL}", font=font, fill=(255,0,0))
#     frame = np.array(img_pil)

#     # ===== FPS =====
#     current_time = time.time()
#     fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
#     prev_time = current_time

#     cv2.putText(frame,
#                 f"FPS: {int(fps)}",
#                 (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX,
#                 1,
#                 (0, 255, 0),
#                 2)

#     cv2.imshow("He thong dem va phan loai khoai tay", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()