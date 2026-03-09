import cv2
import numpy as np
from ultralytics import YOLO
from PIL import ImageFont, ImageDraw, Image

model = YOLO("best2.pt")

vietnamese_names = {
    "Damaged potato": "Khoai nứt",
    "Defected potato": "Khoai biến dạng",
    "Diseased-fungal potato": "Khoai nấm bệnh",
    "Potato": "Khoai tốt",
    "Sprouted potato": "Khoai mọc mầm"
}

font = ImageFont.truetype("arial.ttf", 28)

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


video_path = "test2.mp4"
cap = cv2.VideoCapture(video_path)

ret, frame = cap.read()
height, width, _ = frame.shape
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# ===============================
# ===== CHIA 3 VÙNG 1/5 - 3/5 - 1/5 =====
# ===============================

left_zone_x2 = width // 5
right_zone_x1 = width - (width // 5)

# ===== VÙNG ĐẾM (GIỮ NGUYÊN CỦA BẠN - 3/5 GIỮA) =====
box_width = int(width * 3 / 5)
box_height = height

box_x1 = width//2 - box_width//2
box_x2 = width//2 + box_width//2
box_y1 = 0
box_y2 = height

# ===== VÙNG CHỐT (NẰM TRONG VÙNG ĐẾM - NHỎ HƠN) =====
lock_margin = 80
lock_x1 = box_x1 + lock_margin
lock_x2 = box_x2 - lock_margin
lock_y1 = 50
lock_y2 = height - 50


SL = 0
id_class_map = {}
counted_ids = set()
id_state = {}
id_conf_store = {}   # LƯU CONF THEO ID

while True:
    ret, frame = cap.read()
    if not ret:
        break

    total_conf = 0
    total_boxes = 0

    results = model.track(
        frame,
        conf=0.4,
        iou=0.5,
        persist=True,
        tracker="bytetrack.yaml"
    )

    for r in results:
        boxes = r.boxes

        if boxes.id is None:
            continue

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

        for item in filtered:
            if not item["keep"]:
                continue

            x1, y1, x2, y2 = item["box"]
            track_id = item["track_id"]
            cls = item["cls"]
            conf = item["conf"]

            total_conf += conf
            total_boxes += 1

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # ===============================
            # ===== LƯU CONF KHI NẰM TRONG VÙNG CHỐT =====
            # ===============================
            fully_inside_lock = (
                x1 >= lock_x1 and x2 <= lock_x2 and
                y1 >= lock_y1 and y2 <= lock_y2
            )

            if fully_inside_lock:
                if track_id not in id_conf_store:
                    id_conf_store[track_id] = []

                id_conf_store[track_id].append((cls, conf))

                # CHỐT CLASS THEO CONF TRUNG BÌNH CAO NHẤT
                class_scores = {}
                for c, cf in id_conf_store[track_id]:
                    class_scores.setdefault(c, []).append(cf)

                best_class = max(
                    class_scores.items(),
                    key=lambda x: np.mean(x[1])
                )[0]

                id_class_map[track_id] = model.names[best_class]

            # Nếu chưa chốt thì dùng tạm class hiện tại
            if track_id not in id_class_map:
                id_class_map[track_id] = model.names[cls]

            locked_name = id_class_map[track_id]
            class_name = vietnamese_names.get(locked_name, locked_name)

            # ===============================
            # ===== LOGIC ĐẾM (GIỮ NGUYÊN) =====
            # ===============================
            inside = (box_x1 <= cx <= box_x2) and (box_y1 <= cy <= box_y2)

            if track_id not in id_state:
                id_state[track_id] = "outside"

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

    # ===== TÍNH % CHÍNH XÁC =====
    if total_boxes > 0:
        avg_conf = (total_conf / total_boxes) * 100
    else:
        avg_conf = 0

    # ===== VẼ 3 VÙNG =====

    # Vùng trái 1/5
    cv2.rectangle(frame, (0, 0), (left_zone_x2, height), (255,0,0), 2)

    # Vùng phải 1/5
    cv2.rectangle(frame, (right_zone_x1, 0), (width, height), (255,0,0), 2)

    # Vùng đếm giữa (GIỮ NGUYÊN - đỏ)
    cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0,0,255), 3)

    # Vùng chốt (CAM)
    cv2.rectangle(frame, (lock_x1, lock_y1), (lock_x2, lock_y2), (0,165,255), 3)

    img_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(img_pil)

    draw.text((width - 250, 30),
              f"So luong: {SL}",
              font=font,
              fill=(255,0,0))

    draw.text((width//2 - 120, 30),
              f"Accuracy: {avg_conf:.1f}%",
              font=font,
              fill=(0,255,255))

    frame = np.array(img_pil)

    cv2.imshow("QC Potato System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()






# import cv2
# import numpy as np
# from ultralytics import YOLO
# from PIL import ImageFont, ImageDraw, Image

# model = YOLO("best2.pt")

# vietnamese_names = {
#     "Damaged potato": "Khoai nứt",
#     "Defected potato": "Khoai biến dạng",
#     "Diseased-fungal potato": "Khoai nấm bệnh",
#     "Potato": "Khoai tốt",
#     "Sprouted potato": "Khoai mọc mầm"
# }

# font = ImageFont.truetype("arial.ttf", 28)

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


# video_path = "test2.mp4"
# cap = cv2.VideoCapture(video_path)

# ret, frame = cap.read()
# height, width, _ = frame.shape
# cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

# # ===== VÙNG ĐẾM =====
# box_width = 350
# box_height = height

# box_x1 = width//2 - box_width//2
# box_x2 = width//2 + box_width//2
# box_y1 = 0
# box_y2 = height

# SL = 0
# id_class_map = {}
# counted_ids = set()
# id_state = {}

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     total_conf = 0
#     total_boxes = 0

#     results = model.track(
#         frame,
#         conf=0.4,
#         iou=0.5,
#         persist=True,
#         tracker="bytetrack.yaml"
#     )

#     for r in results:
#         boxes = r.boxes

#         if boxes.id is None:
#             continue

#         filtered = []

#         for box, track_id in zip(boxes, boxes.id):
#             x1, y1, x2, y2 = map(int, box.xyxy[0])
#             conf = float(box.conf[0])

#             current_box = [x1, y1, x2, y2]
#             keep = True

#             for prev in filtered:
#                 iou = compute_iou(current_box, prev["box"])
#                 if iou > 0.6:
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

#         for item in filtered:
#             if not item["keep"]:
#                 continue

#             x1, y1, x2, y2 = item["box"]
#             track_id = item["track_id"]
#             cls = item["cls"]
#             conf = item["conf"]

#             total_conf += conf
#             total_boxes += 1

#             if track_id not in id_class_map:
#                 id_class_map[track_id] = model.names[cls]

#             locked_name = id_class_map[track_id]
#             class_name = vietnamese_names.get(locked_name, locked_name)

#             cx = int((x1 + x2) / 2)
#             cy = int((y1 + y2) / 2)

#             inside = (box_x1 <= cx <= box_x2) and (box_y1 <= cy <= box_y2)

#             if track_id not in id_state:
#                 id_state[track_id] = "outside"

#             if id_state[track_id] == "outside" and inside:
#                 id_state[track_id] = "inside"

#             elif id_state[track_id] == "inside" and not inside:
#                 if track_id not in counted_ids:
#                     SL += 1
#                     counted_ids.add(track_id)
#                 id_state[track_id] = "counted"

#             # DRAW BOX
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

#             img_pil = Image.fromarray(frame)
#             draw = ImageDraw.Draw(img_pil)
#             draw.text((x1, y1 - 30),
#                       f"ID:{track_id} {class_name} {conf:.2f}",
#                       font=font,
#                       fill=(0,255,0))
#             frame = np.array(img_pil)

#     # ===== TÍNH % CHÍNH XÁC =====
#     if total_boxes > 0:
#         avg_conf = (total_conf / total_boxes) * 100
#     else:
#         avg_conf = 0

#     # ===== VẼ VÙNG ĐẾM =====
#     cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0,0,255), 3)

#     img_pil = Image.fromarray(frame)
#     draw = ImageDraw.Draw(img_pil)

#     # SỐ LƯỢNG (góc phải)
#     draw.text((width - 250, 30),
#               f"So luong: {SL}",
#               font=font,
#               fill=(255,0,0))

#     # % CHÍNH XÁC (GIỮA MÀN HÌNH)
#     draw.text((width//2 - 120, 30),
#               f"Accuracy: {avg_conf:.1f}%",
#               font=font,
#               fill=(0,255,255))

#     frame = np.array(img_pil)

#     cv2.imshow("QC Potato System", frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
