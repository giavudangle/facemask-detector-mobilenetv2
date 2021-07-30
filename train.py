# Import các thư viện cần thiết

# Image Data Generator - Khởi tạo các lô chứa các tensor image theo thời gian thực
# Generate batches of tensor image data with real-time data augmentation.
from re import split
from sys import path
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Converts a PIL Image instance to a Numpy array.
from tensorflow.keras.preprocessing.image import img_to_array

# Loads an image into PIL format.
from tensorflow.keras.preprocessing.image import load_img

# Converts a class vector (integers) to binary class matrix.
from tensorflow.keras.utils import to_categorical

# MobileNetV2 - Kiến trúc MobileNetV2 dùng để áp dụng train
# Instantiates the MobileNetV2 architecture.
from tensorflow.keras.applications import MobileNetV2

# Tiền xử lý input cho MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Input Layer - Một loại lớp trong mạng CNN để khởi tạo 1 tensor input
# Input() is used to instantiate a Keras tensor.
from tensorflow.keras.layers import Input
# AveragePooling2D Layer - Một loại lớp trong mạng CNN để giảm kích thước đầu vào bằng lấy lấy trung bình cộng khi chập
# Average pooling operation for spatial data.
from tensorflow.keras.layers import AveragePooling2D
# Dropout Layer - Một loại lớp trong mạng CNN để áp dụng giảm tải tới input tránh overfitting ( lan truyền tiến và lan truyền ngược)
# Applies Dropout to the input.
from tensorflow.keras.layers import Dropout
# Dense Layer (Fully-connected layter) - Một loại lớp trong mạng CNN để kết hợp cấc đặc điểm đã tổng hợp được từ Pooling và Convolutional Layer
from tensorflow.keras.layers import Dense
# Flatten Layer - Một loại lớp trong mạng CNN để làm phằng đầu vào nhưng không gây ảnh hưởng BatchSize
# Flattens the input. Does not affect the batch size.
from tensorflow.keras.layers import Flatten

# Model - Gom nhóm các lớp thành một object với các tính năng dùng để đào tạo và suy luận
# Model groups layers into an object with training and inference features.
from tensorflow.keras.models import Model
# Hàm kích hoạt (Activation Function) - Dùng để tối ưu ( tìm cặp weight và bias phù hợp) Adam =  Momentum + RMSprop
# Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order moments.
from tensorflow.keras.optimizers import Adam

# Gán nhãn nhị phân
# Binarize labels in a one-vs-all fashion.
from sklearn.preprocessing import LabelBinarizer
# Phân chia dữ liệu 
# Split arrays or matrices into random train and test subsets
from sklearn.model_selection import train_test_split
# Build a text report showing the main classification metrics.
from sklearn.metrics import classification_report

from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

from tensorflow.python.training.tracking import base

# Xây dựng quá trình đọc tham số 
ap = argparse.ArgumentParser()
ap.add_argument("-d","--dataset",required=True,help="Dataset Path")
ap.add_argument("-p","--plot",type=str,default="loss_accuracy_plot.png",help="Loss/Accuracy Plot Path")
ap.add_argument("-m","--model",type=str,default="facemask_detector.model",help="Output Facemask Detector Path")
args = vars(ap.parse_args())

# Khởi tạo các hyperparameters ( siêu tham số ) lần lượt là : 
# Tốc độ học (Learning Rate), Số lượng chu kỳ (Number Of Epochs) và Kích thước lô (Batch Size)

LEARNING_RATE = 1e-4
NUMBER_OF_EPOCHS = 20
BATCH_SIZE = 32

# Lấy danh sách các hình ảnh từ dataset và khởi tạo rồi phân lớp
print("[INFO] Loading images from Dataset ")
imagePaths = list(paths.list_images(args["dataset"]))
data = []
labels = []

# Lặp qua mỗi đường dẫn ảnh
for imagePath in imagePaths:
    # Trích xuất nhãn của hình ảnh từ tên file
    label = imagePath.split(os.path.os.sep)[-2]

    # Nạp hình ảnh kích thước (224x224px) và tiến hành tiền xử lý
    image = load_img(imagePath,target_size=(224,224))
    # Chuyển hình ảnh thành numpy array
    image = img_to_array(image)
    # Tiền xử lý input cho mạng MobileNetV2
    image = preprocess_input(image)

    # Cập nhật dữ liệu và gán nhãn danh sách tương ứng
    data.append(image)
    labels.append(label)


# Chuyển đổi danh sách dữ liệu và nhãn thành Numpy Array
data = np.array(data,dtype="float32")
labels = np.array(labels)

# Ánh xạ labels sang gãn nhãn nhị phân
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
# Chuyển đổi sang ma trận nhị phân
labels = to_categorical(labels)

# Phân chia dữ liệu thành 2 phần là : tập huấn luyện và tập kiểm tra
# 75% để huấn luyện và 25% còn lại để đánh giá kiểm tra
(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=0.2,stratify=labels,random_state=42)

# Xây dựng quy trình khởi tạo hình ảnh cho quá trình tăng dữ liệu (data augmentation)

aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Luồng đi như sau :
# 1. Nạp tensor(224,224,3) vào MobileNetV2
# 2. Cho tensor đi qua AveragePooling2D (pool_size = 7x7)
# 3. Làm phẳng tensor
# 4. Cho tensor đi qua FullyConnected (output = 128 , activation function = relu)
# 5. Dropout model 

# Nạp Input tạo kiến trúc MobileNetV2, bảo đảm các lớp Fully Connected Layer được lấy ra
baseModel = MobileNetV2(weights="imagenet",include_top=False,input_tensor=Input(shape=(224,224,3)))
# Tạo mô hình đầu (cấu hình mô hình) từ output của baseModel
headModel= baseModel.output
# Biến đổi từ việc đi qua 1 lớp Average có PoolSize là 7x7
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
# Làm phẳng mô hình bằng cách đi qua flatten layer
headModel = Flatten(name="flatten")(headModel)
# Biến đổi từ việc đi qua 1 lớp Fully Connected (Dense Layer) 
headModel = Dense(128,activation="relu")(headModel)
# Tiến hành DropOut
headModel = Dropout(0.5)(headModel)
# Làm phẳng mô hình bằng cách đi qua softmax layer
headModel = Dense(2,activation="softmax")(headModel)

# Đặt mô hình headModel lên trên baseModel nó là mô hình thực tế đem đi huấn luyện
model = Model(inputs = baseModel.input,outputs = headModel)

# Tiến hành compile
print("[INFO] compiling model...")
# Tiến hành tối ưu hóa mô hình
optimizer = Adam(lr=LEARNING_RATE, decay=LEARNING_RATE / NUMBER_OF_EPOCHS)
model.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])

# Huấn luyện mô hình đầu của mạng
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BATCH_SIZE),
	steps_per_epoch=len(trainX) // BATCH_SIZE,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BATCH_SIZE,
	epochs=NUMBER_OF_EPOCHS)

# Dự đoán mô hình trên tập dữ liệu kiểm tra
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BATCH_SIZE)

# Với mỗi hình ảnh trong tập kiểm tra, tìm index của label có xác xuất lớn nhất tương ứng
predIdxs = np.argmax(predIdxs, axis=1)

# Show kết quả báo cáo
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# Lưu mô hình 
print("[INFO] saving mask detector model...")
model.save(args["model"], save_format="h5")

N = NUMBER_OF_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
plt.show()
plt.table()