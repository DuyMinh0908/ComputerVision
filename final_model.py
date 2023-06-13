from keras.utils.image_utils import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# Thiết lập detect_smile.py [-v file video]
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the (optional) video file")
args = vars(ap.parse_args())
age_ranges = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116']
gender_ranges = ['male', 'female']
emotion_ranges= ['positive','negative','neutral']
export_dir='/media/duyminh/0C38568D400A903A/ComputerVision/1.3_emotion_input_output-20230612T072635Z-001/1.3_emotion_input_output/output/emotion_model_pretrained.h5'
emotion_model = load_model(export_dir)
# Nạp model dò tìm khuôn mặt
# face detector cascade (tích hợp trong OpenCV)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# # Nạp model lenet (dò tìm mặt có cười/không cười)
# model = load_model("lenet.hdf5")

# Nếu không sử dụng video thì mở Webcam
if not args.get("video", False):
    camera = cv2.VideoCapture(0) # Mở Webcam
# Trường hợp khác mở file video
else:
    camera = cv2.VideoCapture(args["video"]) # Sử dụng trường hợp đọc file video

while True:
    # Lấy khung hình hiện tại
    (grabbed, frame) = camera.read()
    # Nếu chúng ta đang xem một video và chúng ta không lấy được khung hình,
    # thì kết thúc video
    if args.get("video") and not grabbed:
        break

    # Thay đổi kích thước frame
    frame = imutils.resize(frame, width=600)
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # sao chép frame để vẽ trên nó sau này
    frame_clone = frame.copy()

    # Dò mặt người trong frame, lấy frame để vẽ trên nó sau này
    faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)

    # Lặp qua hộp xung quanh khuông mặt
    for (fX, fY, fW, fH) in faces:
        # rích xuất ROI của khuôn mặt từ ảnh xám,thay đổi kích thước thành 28x28 pixel
        # chuẩn bị ROI để phân loại mặt cười/không cười bằng CNN sau này
        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # # Dự đoán "smiling" và "not smiling" để gán nhãn cho nó
        # (not_smiling, smiling) = model.predict(roi)[0]
        # label = "Smiling" if smiling > not_smiling else "Not Smiling"
        output_emotion= emotion_ranges[np.argmax(emotion_model.predict(roi))]
        # # Hiển thị nhãn ("Smiling" hoặc "Not Smiling") và hộp hình vuông trên frame
        cv2.putText(frame_clone, output_emotion, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frame_clone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

    # Hiển thị mặt người và nhãn "smiling" hoặc "not smiling"
    cv2.imshow("Demo thu nghiem Face", frame_clone)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Xóa camera và đóng ứng dụng
camera.release()
cv2.destroyAllWindows()