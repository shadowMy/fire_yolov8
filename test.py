from ultralytics import YOLO

# Load a model
model = YOLO('runs/train/yolov8s/weights/best.pt')
# model = YOLO('runs/train/yolov8s_rfcbam_ccfm_dfhead/weights/best.pt')

# Predict with the model
# results = model(r'D:\workspace\PycharmProjects\datasets\yolov8_mydata\test\images')  # predict on an image

model.predict(source=r'D:\workspace\PycharmProjects\datasets\yolov8_mydata\test\images',
              save=True,save_conf=True,save_txt=True,name='yolov8s')
# Export the model
# model.export(format='onnx')

