# USAGE
# python detect_mask_video.py

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os

MASK = "MASK"
WITHOUT_MASK = "NO MASK"

def detect_and_predict_mask(frame, faceNet, maskNet):
    # Lấy kích thước của khung hình và xây dựng BLOG (Binary Large Object) từ nó

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

    # Truyền BLOB thông qua mạng để tiến hành nhận dạng
	faceNet.setInput(blob)
	detections = faceNet.forward()

    # Khởi tạo danh sách khuôn mặt, vị trí tương ứng và danh sách các dự đoán từ model đã được train
	faces = []
	locs = []
	preds = []

    # Lặp để tiến hành nhận dạng
	for i in range(0, detections.shape[2]):
        # Trích xuất biến cố (xác suất) được liên kết từ quy trình nhận dạng

		confidence = detections[0, 0, i, 2]

		 # Lọc ra các thành phần liên thông yếu bằng cách đảm bảo confidence > min confidence
        # Hay confidence > args["confidenece"]
		if confidence > args["confidence"]:
            # Tính tọa độ (x,y) lấy từ hộp bao (bounding box) của vật thể

			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
            # Đảm bảo hộp bao nằm trong kích thước khung hình

			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

		
            # Trích xuất gương mặt ROI ( gương mặt có trong boundingbox), chuyển từ
            # kênh màu BGR sang RGB, sắp xếp và resize về 224x224, rồi tiến hành tiền xử lý
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

            # Thêm khuôn mặt và bounding box vào danh sách

			faces.append(face)
			locs.append((startX, startY, endX, endY))

    # Tiến hành dự đoán nếu có ít nhất một khuôn mặt được nhận dạng
	if len(faces) > 0:
	  # Để tăng tốc quá trình suy luận, ta tạo ra kích thước lô để nhận dạng cho tất cả
        # khuôn mặt tại cùng 1 thời điểm sẽ tốt hơn là nhận dạng 1-1 
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
	return (locs, preds)

# Xây dựng tham số và load model
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--face", type=str,
	default="face_detector",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="facemask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

#Lặp qua các khung hình có trong video
while True:
    # Lấy các khung hình từ luồng của video và resize kích thước (maxWidth = 400px)

	frame = vs.read()
	frame = imutils.resize(frame, width=1920)

    # Nhận diện khuôn mặt từ khung hình

	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # Lặp qua các khuôn mặt đã nhận dạng và lấy locations

	for (box, pred) in zip(locs, preds):
        # Lấy các tọa đồ từ bounding box để nhận dạng
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

        # Xác định nhãn mà màu để vẽ ra khung hình

		label = MASK if mask > withoutMask else WITHOUT_MASK
		color = (0, 255, 0) if label == MASK else (0, 0, 255)
			
        # Hiển thị độ chính xác (xác suất)
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # Hiển thị nhãn ra roudingbox cv

		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    # Hiển thị output
	cv2.imshow("COVID-19 FACEMASK DETECTOR", frame)
	key = cv2.waitKey(1) & 0xFF

    # Nhấn C để break loop
	if key == ord("c"):
		break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
