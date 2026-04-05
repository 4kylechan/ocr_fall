import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret == False:
        break

    frame_height, frame_width, _ = frame.shape

    results = model.track(frame, persist=True)

    boxes = results[0].boxes
    names = results[0].names

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            class_id = int(box.cls[0])
            class_name = names[class_id]

            if class_name != "person":
                continue

            track_id = None

            if box.id is not None:
                track_id = int(box.id[0])

            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(frame_width, int(x2))
            y2 = min(frame_height, int(y2))

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            crop = frame[y1:y2, x1:x2]

            # bbox çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # center çiz
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # track id yaz
            if track_id is not None:
                cv2.putText(
                    frame,
                    f"id: {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # crop göster
            if crop.size > 0:
                cv2.imshow("crop", crop)

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()