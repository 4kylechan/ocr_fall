import cv2
import easyocr
from ultralytics import YOLO

# model yükleme
model = YOLO("yolov8n.pt")

# OCR başlatma
reader = easyocr.Reader(["en"], gpu=False)

# kamera açma
cap = cv2.VideoCapture(0)

# frame sayacı
frame_count = 0

# son OCR sonucu
last_ocr_text = ""

while True:
    ret, frame = cap.read()

    if ret == False:
        break

    frame_count = frame_count + 1

    frame_height, frame_width, _ = frame.shape

    # tracking çalıştır
    results = model.track(frame, persist=True, verbose=False)

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

            # koordinatları güvenli hale getir
            x1 = max(0, int(x1))
            y1 = max(0, int(y1))
            x2 = min(frame_width, int(x2))
            y2 = min(frame_height, int(y2))

            # merkez nokta
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # tam person crop
            crop = frame[y1:y2, x1:x2]

            # kutu çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # merkez çiz
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # track id yaz
            if track_id is not None:
                cv2.putText(
                    frame,
                    f"track_id: {track_id}",
                    (x1, y1 - 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            if crop.size > 0:
                # crop penceresi çok büyük olmasın
                small_crop = cv2.resize(crop, (150, 150))
                cv2.imshow("crop", small_crop)

                # OCR'yi her framede değil, 3 framede bir çalıştır
                if frame_count % 3 == 0:
                    # OCR için ayrı boyutlandırma
                    resized_for_ocr = cv2.resize(crop, (300, 300))

                    ocr_results = reader.readtext(resized_for_ocr)

                    # ilk sonucu al
                    if len(ocr_results) > 0:
                        last_ocr_text = ocr_results[0][1]

            # son OCR sonucunu ekranda göster
            if last_ocr_text != "":
                cv2.putText(
                    frame,
                    f"OCR: {last_ocr_text}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

            # şimdilik sadece ilk person üzerinde çalış
            break

    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()