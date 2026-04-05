import cv2
import easyocr
from ultralytics import YOLO

# model yükleme
model = YOLO("yolov8n.pt")

# OCR başlatma
reader = easyocr.Reader(["en"], gpu=False)

# kamera açma
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret == False:
        break

    frame_height, frame_width, _ = frame.shape

    # tracking çalıştır
    results = model.track(frame, persist=True)

    boxes = results[0].boxes
    names = results[0].names

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            class_id = int(box.cls[0])
            class_name = names[class_id]

            # sadece person
            if class_name != "person":
                continue

            # track id alma
            track_id = None
            if box.id is not None:
                track_id = int(box.id[0])

            # koordinatları sınırla
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(frame_width, int(x2))
            y2 = min(frame_height, int(y2))

            # center hesaplama
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # crop alma
            crop = frame[y1:y2, x1:x2]

            # OCR sonucu
            detected_text = ""

            if crop.size > 0:
                # crop küçült (ekrana sığsın)
                small_crop = cv2.resize(crop, (150, 150))
                cv2.imshow("crop", small_crop)

                # OCR çalıştır
                ocr_results = reader.readtext(crop)

                # ilk sonucu al (varsa)
                if len(ocr_results) > 0:
                    detected_text = ocr_results[0][1]

            # bbox çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # center çiz
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # track id yaz
            if track_id is not None:
                cv2.putText(
                    frame,
                    f"id: {track_id}",
                    (x1, y1 - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # OCR sonucu yaz
            if detected_text != "":
                cv2.putText(
                    frame,
                    f"OCR: {detected_text}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()